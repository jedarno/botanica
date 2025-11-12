"""
Functions for using swarm optimiser to select a model suite
"""

def model_wrapper(swarm_values, models, thershold, trainloader, trainset, valloader, vallset, device, num_spochs):
  """
  Function to return the fitness of the current swarm position
  
  ------------------------------------------------------------------
  Parameters

  + swarm_values: iterable of floats
  + models: list of torch nn.Module instances 
  + threshold:float Only inlude models whos swarm value is above the threshold
  + trainloader:Dataloader
  + trainset 
  + valloader:Dataloader
  + vallset
  + device
  + num_epochs:int

  Returns

  Ensemble model using swarm selected suite
  """

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

