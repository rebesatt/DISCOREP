import torch

class Metric:
  @staticmethod
  def hit_at_k(predictions, ground_truth_idx, device, k = 10):
    return Metric.hit_at_k_per_edge(predictions, ground_truth_idx, device, k=k).sum().item()

  @staticmethod
  def hit_at_k_per_edge(predictions, ground_truth_idx, device, k=10):
    assert predictions.shape[0] == ground_truth_idx.shape[0]

    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)

    _, indices = predictions.topk(k=k, largest=False)
    return torch.where(indices == ground_truth_idx, one_tensor, zero_tensor)

  @staticmethod
  def mrr(predictions, ground_truth_idx):
    assert predictions.shape[0] == ground_truth_idx.shape[0]

    indices = predictions.argsort()
    return (1.0 / (indices == ground_truth_idx).nonzero()[:, 1].float().add(1.0)).sum().item()
