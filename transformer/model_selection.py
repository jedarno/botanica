import torch.optim as optim
import torch.nn as nn

from torch import cat
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

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


def fitness_wrapper(swarm_values, ensemble_arch, models, threshold, k_vals, trainloader, trainset, valloader, valset, device, weights = None):
  """
  Function to take the swarm position and return fitness using the model wrapper

  parameters
  position:np.ndarray

  returns 
  fitness:iterable:float
  """

  k1, k2 = k_vals

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
  train_scores = run_topk_test(ensemble, classes, trainloader, trainset, criterion, device)
  
  #val_acc and loss 
  val_scores = run_topk_test(ensemble, classes, valloader, valset, criterion, device)
  val_acc = val_scores[0].item()
  val_loss = val_scores[3]

  #fitness = (k1 * val_loss) + (k2/len(models) *  len(ensemble.models))
  fitness = -((k1 * val_acc) + (k2 * 1/(len(ensemble.models))))

  return fitness


def _get_model_output(model, dataloader, device):

  model.eval()
  batched_output = []

  for data, _ in tqdm(dataloader):
    data = data.to(device)
    output= model(data)
    batched_output.append(output)

  output = cat(batched_output, dim=0)
  return output

def get_suite_outputs(modelsuite, dataloader, device):
  """   
  For each model store output in a pandas data frame indexed by batch. Then save to a csv file. 
  """

  outputs = []

  for model in modelsuite: 
    model_output = _get_model_output(model, dataloader, device)
    outputs.append(model_output)

  return outputs

def get_ensemble_output(output_vals, weights):

  sftmx = nn.Softmax(dim=1)
  ensemble_out = weights[0] * sftmx(output_vals[0])
  print(ensemble_out)

  for i in range(1, len(output_vals)):
    ensemble_out += weights[i] * sftmx(output_vals[i])
    print(ensemble_out)

  return ensemble_out

def get_model_suite(swarm_values, models, threshold):
  model_suite = []

  for i, model in enumerate(models):
    
    if swarm_values[i] >= threshold:
      model_suite.append(model)

  return model_suite

def fitness_wrapper_ensemble_out(model_suite, k_vals, device, classes, trainloader, valloader, weights=None):
  
  k1, k2 = k_vals
  
  criterion = LabelSmoothingCrossEntropy()

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

  for param in model.parameters():
    param.requires_grad = False

  n_inputs = model.fc.in_features
  model.fc = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, n_classes)
  )

  for param in model.trunk_output.block3.parameters():
    param.requires_grad = True

  for param in model.trunk_output.block4.parameters():
    param.requires_grad = True

  return model


