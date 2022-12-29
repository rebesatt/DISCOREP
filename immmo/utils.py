import torch
import numpy as np
import random

def set_seed(seed):
  torch.use_deterministic_algorithms(True)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def get_unique_first_occurences(values):
  unique_values, idx, counts = values.unique(
      sorted=True, return_inverse=True, return_counts=True)

  _, ind_sorted = torch.sort(idx, stable=True)
  cum_sum = counts.cumsum(0)
  cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
  first_indicies = ind_sorted[cum_sum]

  assert (values[first_indicies] == unique_values).all()

  return unique_values, first_indicies

def batch_wise_dot(a, b):
  batch_size = a.shape[0]
  dim = a.shape[1]
  X = a.reshape(batch_size, 1, dim)
  Y = b.reshape(batch_size, dim, 1)
  return torch.matmul(X, Y).squeeze(1)


def get_f1_of_matches(ground_truth, matches):
  truth_set = set(ground_truth.tolist())

  matched_set = set(matches.tolist())

  TP = len(matched_set & truth_set)
  P = len(matched_set)
  T = len(truth_set)

  if P == 0 or TP:
    return 0, 0, 0

  prec = TP / P
  recall = TP / T
  f1 = 2*(prec*recall)/(prec+recall)
  return prec, recall, f1

TORCH_EPS = torch.tensor(np.finfo(float).eps)
