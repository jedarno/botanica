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


