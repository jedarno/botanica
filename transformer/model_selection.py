import os
import torch.optim as optim
import torch.nn as nn

from torch import cat, load, stack
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

  fitness = (k1 * val_loss) + (k2/len(models) *  len(ensemble.models))

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

  for i in range(1, len(output_vals)):
    ensemble_out += weights[i] * sftmx(output_vals[i])

  return ensemble_out

def get_model_suite(swarm_values, model_outputs, threshold):
  model_suite = []

  for i, model in enumerate(model_outputs):
    
    if swarm_values[i] >= threshold:
      model_suite.append(model)

  return model_suite

def topk_acc(output, targets, topk):
  
  maxk = max(topk)
  _, y_pred = output.topk(k = maxk, dim=1)
  y_pred = y_pred.t()
  target_reshaped = targets.view(1, -1).expand_as(y_pred)
  correct = (y_pred == target_reshaped)

  list_topk_accs = [] 
  for k in topk:
    ind_which_topk_matched_truth = correct[:k]
    flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
    tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
    list_topk_accs.append(tot_correct_topk)

  return list_topk_accs

def _get_output_scores(logits, dataset, criterion, device, classes):
  
  data_loss = 0.0
  class_correct = list(0 for i in range(len(classes)))
  sum_top1 = 0
  sum_top3 = 0
  sum_top5 = 0 
  targets = dataset.targets
  targets = targets.to(device)
  loss = criterion(logits, targets)
  data_loss += loss.item()
  top_k = topk_acc(logits, targets, (1,3,5))
  top_1 = top_k[0][0] / len(dataset)
  top_3 = top_k[1][0] / len(dataset)
  top_5 = top_k[2][0] / len(dataset)
  print('Loss: {:.4f}'.format(loss))
  print('top_k acc: {:.4f}, {:.4f}, {:.4f}'.format(top_1, top_3, top_5))

  return (top_1, top_3, top_5, loss)

def fitness_wrapper_ensemble(train_output, val_output, num_models, k_vals, device, classes, trainset, vallset):
  
  k1, k2 = k_vals
  
  criterion = LabelSmoothingCrossEntropy()
  
  criterion = criterion.to(device)

  train_scores = _get_output_scores(train_output, trainset, criterion, device, classes)
  val_scores = _get_output_scores(val_output, vallset, criterion, device, classes)
  val_acc = val_scores[0]

  fitness = -((k1 * val_acc) + (k2 * 1/num_models))
  return fitness 

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

def load_output_from_dir(path, num_models):
  """
  Loads validation output tensors for a group of models in a single folder and returns as a single tensor of stacked outputs
  ---------------------------------------------------
  parameters
  string:path
  int:num_models
  --------------------------------------------------
  returns
  torch.Tensor:model_outputs
  """

  output_list = [] 
  
  for i in range(num_models):
    model_path = os.path.join(path, f"model_{i+1}_val_logits.pt")
    model_output = load(model_path)
    output_list.append(model_output)

  output_stack = stack(output_list, dim=1)
  
  return output_stack
