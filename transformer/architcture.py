""""
Functions related to combineing or altering model acrchitectures
"""

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
    super(transformer_ensemble, self).__init__()
    self.models = []
    self.sftmx = torch.nn.Softmax(dim=1)
    for model in models:
      self.models.append(model)

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

    output = self.sftmx(output)

    return output
