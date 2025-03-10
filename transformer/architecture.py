""""
Functions related to combineing or altering model acrchitectures
"""
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

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

  def __init__(self, vit_model, attention_mechanism, n_classes):
    super(VitWithAtt, self).__init__()
    self.conv_projection = vit_model.conv_proj
    self.encoder = vit_model.encoder
    self.attention_mechanism = attention_mechanism
    n_inputs = vit_model.heads.head.in_features

    self.classifier = nn.Sequential(
      nn.Linear(n_inputs, 512),
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
    x = self.conv_proj(x)
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x

  def forward(self, x):
    x = self.conv_projection(x)
    x = self.encoder(x)
    x = self.attention_mechanism(x)
    x = self.classifier(x)
    return x


