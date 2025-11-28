import torch.optim as optim
import torch.nn as nn

from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy
from .test_functions import run_topk_test
"""
Functions for using swarm optimiser to select a model suite
"""

def _model_wrapper(swarm_values, ensemble_arch,  models, threshold):
  """
  Function to return the ensemble model using swarm values and threshold
  
  ------------------------------------------------------------------
  Parameters

  + swarm_values: iterable of floats
  + ensemble_arch:nn.Module
  + models: list of torch nn.Module instances 
  + threshold:float Only inlude models whos swarm value is above the threshold

  Returns

  Ensemble model using swarm selected suite
  """
  
  modelset = [] 
  chosen_index = []

  for i, model in enumerate(models):

    if swarm_values[i] >= threshold:
      modelset.append(model)
      chosen_index.append(i)

  if len(modelset) < 1:
    print("ERROR: no models selected")
    return None

  if len(modelset) == 1:
    print("WARNING: only one model selected")
  
  ensemble_model = ensemble_arch(modelset)  
  print("chosen models: ", chosen_index)

  return ensemble_model


def fitness_wrapper(swarm_values, ensemble_arch, models, threshold, trainloader, trainset, valloader, valset, device):
  """
  Function to take the swarm position and return fitness using the model wrapper

  parameters
  position:np.ndarray

  returns 
  fitness:iterable:float
  """

  classes = trainset.classes
  ensemble = _model_wrapper(swarm_values, ensemble_arch,  models, threshold)
  criterion = LabelSmoothingCrossEntropy()

  if ensemble == None:
    return float('inf')

  if device:
    criterion = criterion.to(device)
    ensemble = ensemble.to(device)

    for model in ensemble.models:
      print(model.__class__.__name__)
      model = model.to(device)
  
  #train_acc and loss 
  train_scores = run_topk_test(model, classes, trainloader, trainset, criterion, device)
  print("train scores: ", train_scores)
  
  #val_acc and loss 
  val_scores = run_topk_test(model, classes, valloader, valset, criterion, device)
  val_loss = val_scores[3]
  print("val scores: ", val_scores)

  return val_loss


def get_vit_l_arch(n_classes):
  model = models.vit_l_16(weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

  for param in model.parameters():
    param.requires_grad = False

  n_inputs = model.heads.head.in_features
  model.heads.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, n_classes)
  )

  for param in model.encoder.layers.encoder_layer_23.parameters():
    param.requires_grad = True

  for param in model.encoder.ln.parameters():
    param.requires_grad = True

  for param in model.heads.head.parameters():
    param.requires_grad = True

  return model

def get_regnet_arch(n_classes):
  model = models.regnet_y_16gf(weights = models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1)

  for param in model.trunk_output.block3.parameters():
    param.requires_grad = True

  for param in model.trunk_output.block4.parameters():
    param.requires_grad = True

  return model


