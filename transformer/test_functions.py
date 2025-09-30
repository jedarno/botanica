import torch
import torch.nn as nn
from tqdm import tqdm


def topk_accuracy(output, target, topk):
  """
  Helper function to calculate top-k accuracy of pytorch model output tensor.

  parameters
  ----------
  output : torch.Tensor
    model output to be evaluated

  target : Iterable : Int
    True labels for test data

  topk : Iterable : Int 
    integer values of k

  output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]

  """
  with torch.no_grad():
    #get top k labels
    maxk = max(topk)
    batch_size  = target.size(0)
    #get maxk indices
    _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
    y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
    target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
    correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth

    #get top-k accuracy
    list_topk_accs = []  
    for k in topk:
      #get tensor of wich topk is correct
      ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
      # flatten it to help compute if we got it correct for each example in batch
      flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
      # get if we got it right for any of our top k prediction for each example in batch
      tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
      list_topk_accs.append(tot_correct_topk)

    return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

def run_topk_test(model, classes, testloader, testset, criterion, device):
  """
  Evaluate topk accuracy and cross entropy for a model over a test data set given as a Pytorch DataLoader object

  parameters
  ----------
  model : torch.nn.module

  testloader : torch.utils.data.DataLoader
    DataLoader for the test data to evaluate the model

  returns : Tuple
    Tuple contains top k accuracies and the test loss
  """
  criterion = criterion.to(device)
  test_loss = 0.0
  class_correct = list(0 for i in range(len(classes)))
  class_total = list(0 for i in range(len(classes)))
  model.eval()

  sum_top_1 = 0
  sum_top_3 = 0
  sum_top_5 = 0

  for data, target in tqdm(testloader):
      data, target = data.to(device), target.to(device)
      with torch.no_grad(): # turn off autograd for faster testing
          output = model(data)
          loss = criterion(output, target)
      test_loss = loss.item() * data.size(0)
      #_, pred = torch.max(output, 1)
      top_k = topk_accuracy(output, target, (1,3,5))
      sum_top_1 += top_k[0][0]
      sum_top_3 += top_k[1][0]
      sum_top_5 += top_k[2][0]
      #print(top_k)

  top_1 = sum_top_1 / len(testset)
  top_3 = sum_top_3 / len(testset)
  top_5 = sum_top_5 / len(testset)

  test_loss = test_loss / len(testset)
  print(type(top_1))
  print('Test Loss: {:.4f}'.format(test_loss))
  print('top_k acc: {:.4f}, {:.4f}, {:.4f}'.format(top_1, top_3, top_5))

  return (top_1, top_3, top_5, test_loss)

def few_shot_test(model, classes, testloader, testset, device, n_shot=3):
  #TODO: remove support set generation and take as a parameter
  #TODO: get n_shot from sup set size and remove redundant parameter
  #TODO: Check support set dims match
  #TODO: add loss tracking using triplet margin loss
  test_loss = 0.0
  running_corrects = 0
  pdist = nn.PairwiseDistance(p=2)
  size = len(testset)

  support_set1, support_set2 = testset.get_support_set(n_shot)

  if device:
    support_set1, support_set2 = support_set1.to(device), support_set2.to(device)

  for anchor, _, _, label in tqdm(testloader):
    anchor, label = anchor.to(device), label.to(device)

    with torch.no_grad(): 
      anchor_embeddings = model.tower(anchor)
      support_embeddings1 = model.tower(support_set1)
      support_embeddings2 = model.tower(support_set2)
      batch_scores = []
      batch_size = anchor_embeddings.size(0)
      
      for i in range(batch_size):
        anchor_embedding = anchor_embeddings[i]
        dist_class1 = 0
        dist_class2 = 0

        for n in range(n_shot):
          dist_class1 += pdist(anchor_embedding, support_embeddings1[n])
          dist_class2 += pdist(anchor_embedding, support_embeddings2[n])

        score_class1 = -dist_class1
        score_class2 = -dist_class2
        scores = torch.stack((score_class1, score_class2))
        batch_scores.append(scores)

    batch_scores = torch.stack(batch_scores)
    _, pred = torch.max(batch_scores,1)
    running_corrects += torch.sum(pred == label.data)

  epoch_acc = running_corrects / size
  print(epoch_acc)
        

