from cgi import test
from statistics import mode
import torch

from immmo.helpers.metric import Metric

class Test:
  @staticmethod
  def test_model(model, key_event_test_triplets, timing_test_triplets):
    pass

  @staticmethod
  def test_key_event(model, test_triplet, k=10):
    examples_count = 0.0
    hits_at_10 = 0.0
    mrr = 0.0

    head = test_triplet[:, 0]
    tail = test_triplet[:, 2]

    triplets_tails, triplets_heads = Test.all_key_event_triplet_variations_of(
        model, test_triplet)

    tails_predictions = model.dist_predict(triplets_tails).reshape(1, -1)
    heads_predictions = model.dist_predict(triplets_heads).reshape(1, -1)

    ground_truth_tails = torch.bucketize(tail, model.event_nodes).reshape(-1, 1)
    hits_at_10 += Metric.hit_at_k(tails_predictions,
                                  ground_truth_tails, device=model.device, k=k)
    mrr += Metric.mrr(tails_predictions, ground_truth_tails)

    ground_truth_heads = torch.bucketize(head, model.key_nodes).reshape(-1, 1)
    hits_at_10 += Metric.hit_at_k(heads_predictions,
                                  ground_truth_heads, device=model.device, k=k)
    mrr += Metric.mrr(heads_predictions, ground_truth_heads)

    examples_count += ground_truth_tails.shape[0] + ground_truth_heads.shape[0]

    return hits_at_10, mrr, examples_count

  @staticmethod
  def test_timing(model, test_triplet, k=10):
    head = test_triplet[:, 0]
    tail = test_triplet[:, 2]

    examples_count = 0.0
    hits_at_10 = 0.0
    mrr = 0.0

    triplets = Test.all_timing_triplet_variations_of(model, test_triplet)
    predictions = model.dist_predict(triplets).reshape(2, -1)

    ground_truth_entity_id = torch.cat((torch.bucketize(tail, model.state_holder_nodes).reshape(-1, 1),
                                        torch.bucketize(head, model.state_holder_nodes).reshape(-1, 1)))

    hits_at_10 += Metric.hit_at_k(predictions, ground_truth_entity_id, device=model.device, k=k)
    mrr += Metric.mrr(predictions, ground_truth_entity_id)

    examples_count += predictions.size()[0]

    return hits_at_10, mrr, examples_count

  @staticmethod
  def all_timing_triplet_variations_of(model, positive_triplet, sub_stream_scoped=False):
    head = positive_triplet[:, 0]
    relation = positive_triplet[:, 1]
    tail = positive_triplet[:, 2]

    if sub_stream_scoped:
      substream_id = model.graph._sub_stream_ids[head[0]]
      all_entities = (model.graph._sub_stream_ids ==
                        substream_id).nonzero(as_tuple=True)[0]
    else:
      all_entities = model.state_holder_nodes

    # keys
    heads = head.expand(all_entities.shape[0])
    relations = relation.expand(all_entities.shape[0])
    tails = tail.expand(all_entities.shape[0])

    triplets_tails = torch.stack(
        (heads, relations, all_entities), dim=1)
    triplets_heads = torch.stack(
        (all_entities, relations, tails), dim=1)
    return torch.cat((triplets_tails, triplets_heads))

  @staticmethod
  def all_key_event_triplet_variations_of(model, positive_triplet):
    head = positive_triplet[:, 0]
    relation = positive_triplet[:, 1]
    tail = positive_triplet[:, 2]

    all_keys = model.key_nodes
    all_events = model.event_nodes

    # keys
    heads = head.expand(all_events.shape[0])
    relation_event = relation.expand(all_events.shape[0])
    relation_key = relation.expand(all_keys.shape[0])
    tails = tail.expand(all_keys.shape[0])

    triplets_tails = torch.stack(
        (heads, relation_event, all_events), dim=1)
    triplets_heads = torch.stack(
        (all_keys, relation_key, tails), dim=1)
    return triplets_tails, triplets_heads
