import numpy as np
import torch

def initialEmbeddingVectors(dim, times=1, device='cpu', dtype=torch.float32):
  # returns a normalized random uniform vector
  # epsilon is necessary to ensure that all values of the vector are within the open interval (-1, 1)
  epsilon = np.finfo(float).eps
  embedding = (-2 + epsilon) * torch.rand(times, dim, device=device, dtype=dtype) + 1
  embedding = embedding.div(embedding.norm(dim=1, p=2)[:, None])
  return embedding
