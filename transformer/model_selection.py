import torch.optim as optim
from timm.loss import LabelSmoothingCrossEntropy
from test_functions import run_topk_test
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


  
