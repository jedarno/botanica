""""
Functions related to combineing or altering model acrchitectures
"""
import numpy as np
import torch
import torch.nn as nn

from torchvision import models
from torchinfo import summary
from tqdm import tqdm

class IllegalArgumentError(ValueError):
    pass

class transformer_ensemble_avg(nn.Module):
  """
  Class to combine architectures to produce a mean average of model outputs.

  attributes
  ----------
  models : list
    list of torch.nn.module objects containing the models to be used in the ensemble

  sftmx: instance of torch.nn.Softmax

  """

  def __init__(self, models):
    """
    parameters
    ----------
    models : list
      list of torch.nn.module objects
    """
    super(transformer_ensemble_avg, self).__init__()
    self.models = models
    self.sftmx = nn.Softmax(dim=1)

  def forward(self, x):
    """
    parameters
    ----------
    x : torch.Tensor
      Input for forward pass

    returns
    -------
    torch.Tensor
      Output of forward pass
    """
    first_out = self.models[0](x)
    output = self.sftmx(first_out)

    for i in range(1, len(self.models)):
      base_out = self.models[i](x)
      output = output + self.sftmx(base_out)

    return output

class transformer_ensemble_weighted(nn.Module):
    """
    nn.module class for weighted ensemble models. 

    attributes
    ----------
    models : nn.module 
        Models to be comvined in ensemble

    weights : np.ndarray 
        numpy array of influence weights to be applied to the softmax of model outputs before combining for ensemble output 
    """

    def __init__(self, models, weights=None):
        super(transformer_ensemble_weighted, self).__init__() 
        self.models = models
        self.sftmx = nn.Softmax(dim=1)
        if weights:
                self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):

        if len(weights) != len(self.models):
            raise ValueError("List of weights needs to be the same length as the list of models")

        if type(weights) == list:
            weights = np.array(weights)

        if not np.issubdtype(weights.dtype, np.number):
            raise ValueError("weights should be numeric")
        print("New model influence weights: {}".format(weights))
        self._weights = weights

    def forward(self, x):
        first_out = self.models[0](x)
        output = self.sftmx(first_out)

        for i in range(1, len(self.models)):
            base_out = self.models[i](x)
            output = output + self.weights[i] * self.sftmx(base_out)

        return output

class VitWithAtt(nn.Module):

  def __init__(self, vit_model, attention_mechanism, n_classes, att_input_type="one tensor"):
    super(VitWithAtt, self).__init__()
    self.vit_model = vit_model
    self.class_token = vit_model.class_token
    self.conv_projection = vit_model.conv_proj
    self.encoder = vit_model.encoder
    self.attention_mechanism = attention_mechanism
    self.patch_size = vit_model.patch_size
    self.image_size = vit_model.image_size
    self.hidden_dim = vit_model.hidden_dim

    if att_input_type == "one tensor":
      self.att_input_type = att_input_type

    elif att_input_type == "three tensors":
      self.att_input_type = att_input_type

    else: raise IllegalArgumentError("att_input type not recognised.")

    self.classifier = nn.Sequential(
      nn.Linear(768, 512),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(512, n_classes)
      )

  def _process_input(self, x: torch.Tensor) -> torch.Tensor:
    """
    From torch implementation of VisionTransformer class: https://pytorch.org/vision/0.20/_modules/torchvision/models/vision_transformer.html#vit_b_16
    """
    n, c, h, w = x.shape
    p = self.patch_size
    torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    n_h = h // p
    n_w = w // p

    # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    x = self.conv_projection(x)
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x

  def forward(self, x):
    x = self._process_input(x)
    n = x.shape[0]
    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = self.encoder(x)

    if self.att_input_type == "one tensor":
      x = self.attention_mechanism(x)

    elif self.att_input_type == "three tensors":
      x = self.attention_mechanism(x,x,x) #Self attention can take three inputs, not used in this circumstance

    else: raise IllegalArgumentError("att_input type not recognised.")
      
    # Classifier "token" as used by standard language architectures
    x = x[:, 0]
    x = self.classifier(x)
    return x


class VitAttHead(nn.Module):

  def __init__(self, vit_model, attention_head, n_classes):
    super(VitAttHead, self).__init__()
    self.vit_model = vit_model
    self.class_token = vit_model.class_token
    self.conv_projection = vit_model.conv_proj
    #TODO: self.encoder should just be the encoder layers, LN layer is used elsewhere
    self.encoder = vit_model.encoder
    self.attention_head = attention_head
    self.patch_size = vit_model.patch_size
    self.image_size = vit_model.image_size
    self.hidden_dim = vit_model.hidden_dim

  def _process_input(self, x: torch.Tensor) -> torch.Tensor:
    """
    From torch implementation of VisionTransformer class: https://pytorch.org/vision/0.20/_modules/torchvision/models/vision_transformer.html#vit_b_16
    """

    p = self.patch_size
    torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    n_h = h // p
    n_w = w // p

    # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    x = self.conv_projection(x)
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x

  def forward(self, x):
    x = self._process_input(x)
    n = x.shape[0]
    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = self.encoder(x)
    x = self.attention_head(x)
    return x

class VitEarlyAtt(nn.Module):

  def __init__(self, vit_model, local_att, n_classes, att_input_type = "one tensor"):
    """
    Initialise components:
    vit encoder
    vit class token
    attention mechanism
    classification_head
    """
    super(VitEarlyAtt, self).__init__()
    self.vit_model = vit_model
    self.class_token = self.vit_model.class_token
    self.conv_projection = self.vit_model.conv_proj
    self.conv_projection = self.vit_model.conv_proj
    self.encoder = self.vit_model.encoder
    self.patch_size = self.vit_model.patch_size
    self.image_size = self.vit_model.image_size
    self.hidden_dim = self.vit_model.hidden_dim
    
    self.local_att = local_att

    if att_input_type == "one tensor":
      self.att_input_type = att_input_type

    elif att_input_type == "three tensors":
      self.att_input_type = att_input_type

    else: raise IllegalArgumentError("att_input type not recognised.")

    self.classifier = nn.Sequential(
      nn.Linear(768, 512),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(512, n_classes)
      )
    
  def _initial_projection(self, x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    p = self.patch_size
    torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    n_h = h // p
    n_w = w // p

    # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    x = self.conv_projection(x)
    return x

  def _reshape_and_permute(self, x: torch.Tensor) -> torch.Tensor:
    n, hidden_dim, n_h, n_w = x.shape
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)
    return x

  def forward(self, x):
    """
    Forward pass
    1. Process input 
    1.5 convolutional projection
    2. Local attention 
    3. Add cls tokens
    4. Pass to ViT Encoder
    5. Pass to classifier
    """
    #convolutional projection and get batch size
    x = self._initial_projection(x)
    n = x.shape[0]

    #Local attention
    if self.att_input_type == "one tensor":
      x = self.local_att(x)
    elif self.att_input_type == "three tensors":
      x = self.local_att(x, x, x)
    
    x = self._reshape_and_permute(x)

    #Add cls tokens
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    #ViT Encoder 
    x = self.encoder(x)
    #Classifier
    # Classifier "token" as used by standard language architectures
    x = x[:, 0]
    x = self.classifier(x)
    return x

class cnn_vit(nn.Module):
  """
  input -> CNN -> (Resize to conv proj) -> ViT_encoder -> class head
  """

  def __init__(self, cnn, vit_model, n_classes):
    super(cnn_vit, self).__init__()
    self.cnn = cnn
    self.vit_model = vit_model
    self.class_token = self.vit_model.class_token
    self.encoder = self.vit_model.encoder
    self.patch_size = self.vit_model.patch_size
    self.image_size = self.vit_model.image_size
    self.hidden_dim = self.vit_model.hidden_dim
        
    self.classifier = nn.Sequential(
      nn.Linear(768, 512),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(512, n_classes)
      )

  def _reshape_and_permute(self, x: torch.Tensor) -> torch.Tensor:
    n, hidden_dim, n_h, n_w = x.shape
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)
    return x

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # CNN encoder
    n = x.shape[0]
    x = self.cnn(x)
    # Patching
    x = self._reshape_and_permute(x)
    # ViTEncoder
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = self.encoder(x)
    # Head
    x = x[:, 0]
    x = self.classifier(x)
    return x

class concat_vitl_regnety16gf(nn.Module):

  def __init__(self, vitl, regnety16gf, n_classes):
    super(concat_vitl_regnety16gf, self).__init__()
    self.vitl = vitl
    self.regnety16gf = regnety16gf

    #remove classification heads
    self.vitl.heads.head = nn.Identity()
    self.regnety16gf.fc = nn.Identity()

    #define new classifier
    self.classifier = nn.Sequential(
      nn.Linear(1024 + 3024, 2048),
      nn.ReLU(),
      nn.Dropout(p=0.3, inplace=False),
      nn.Linear(2048, 1024),
      nn.ReLU(),
      nn.Dropout(p=0.3, inplace=False),
      nn.Linear(1024, n_classes)
     )

  def forward(self, x):
    vitl_features = self.vitl(x.clone())
    vitl_features = vitl_features.view(vitl_features.size(0), -1)
    regnety_features = self.regnety16gf(x)
    regnety_features = regnety_features.view(regnety_features.size(0), -1)
    x = torch.cat((vitl_features, regnety_features), dim=1)
    x = self.classifier(x)

    return x
    
class regnety16gf_hybrid_fmap(nn.Module):
  
  def __init__(self, models, n_classes, weights=None):

    super(regnety16gf_hybrid_fmap, self).__init__()
    self.models = models

    if weights:
      self.weights = weights

    head_n_inputs = model.fc.in_features

    for model in self.models:
      model.fc = nn.Identity()	

    #Define new calssifier
    self.classifier = nn.Sequential(
      nn.Linear(head_n_inputs,512),
      nn.ReLU(),
      nn.Dropout(p=0.3),
      nn.Linear(512, n_classes)
     ) 

  @property
  def weights(self):
    return self._weights

  @weights.setter
  def weights(self, weights):
    
    if type(weights) == list:
      weights = np.array(weights)

    if len(weights) != len(self.models):
      raise ValueError("The number of weight values provided should match the number of models")

    elif not np.issubtype(weights.dtype, np.number):
      raise ValueError("Weights should be numeric")

    else:
      print("New model influence weights: {}".format(weights))
      self.weights = weights

  def forward(self, x):
    out = self.weights[0] * self.models[0](x)

    for i in range(1, len(self.models)):
      out += self.models[i](x) * self.weights[i]

    out = self.classifier(out)

    return out
