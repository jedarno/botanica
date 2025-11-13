"""
Functions for using swarm optimiser to select a model suite
"""

def model_wrapper(swarm_values, ensemble_arch,  models, threshold):
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

  for i, model in enumerate(models):

    if swarm_values[i] >= threshold:
      modelset.append(model)


  print(modelset)
  #ensemble_model = ensemble_arch(modelset)  
  pass


def swarm_wrapper(position):
  """
  Function to take the swarm position and return fitness using the model wrapper

  parameters
  position:np.ndarray

  returns 
  fitness:iterable:float
  """

  pass

