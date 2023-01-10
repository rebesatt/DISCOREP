from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import git

from ..helpers.test import Test
from ..constants.GraphTypes import GraphTypes
from ..KnowledgeGraph import KnowledgeGraph
from immmo.utils import *

class RepresentationLearning(nn.Module):
  def __init__(self, graph: KnowledgeGraph, margin=3.0, lr=0.01, embedding_dim=20, gamma=1):
    super(RepresentationLearning, self).__init__()

    self.graph = graph
    self.embedding_dim = embedding_dim
    self.device = self.graph._device

    self.verify = False
    self.use_criterion = False
    self.pow = True
    self.norm_edges_except_event_state = True
    self.norm_event_state_edges = True
    self.norm_f_sigma = True
    self.additional_attentions_on_edges = False
    self.use_mean_loss = False
    self.use_single_timing_label = False

    # config
    self.norm = 2
    self.lr = lr
    self.gamma = gamma
    self.model_output_path = None
    self.training_start = None
    self.loss_report_file = None
    self.test_report_files = None

    # load required nodes and edges
    self.key_nodes = self.graph.keyNodeIndices()
    self.event_nodes = self.graph.eventNodeIndices()
    self.key_event_relations = self.graph.keyEventRelationIndices()
    self.state_holder_nodes = self.graph.stateHolderNodeIndices()
    self.timing_relations = self.graph.timeRelationIndices()
    self.event_state_relations = self.graph.eventStateRelationIndices()
    self.relation_tuples = self.graph._edges

    # index
    self.event_state_relations_per_state = self.graph.event_state_relations_by_state(self.state_holder_nodes)
    self.event_nodes_per_state = self.relation_tuples[self.event_state_relations_per_state, 0]
    self.key_event_relations_per_key = self.graph.key_event_relations_by_key(self.key_nodes)

    self.entity_count = len(self.graph._node_types)
    self.relation_count = len(self.graph._relation_types)

    # maps the total index of relations and node indices to the scoped index within the category
    # Example:
    # - let 1500 be a the index of the first event node: self.node_to_scope[1500] -> 0 (self.event_nodes[0] -> 1500)
    self.relation_to_scope = self._init_relation_to_scope()
    self.node_to_scope = self._init_node_to_scope()

    self.critical_state_node_indices = self.graph.critical_states()
    self.entity_is_critical = torch.zeros(self.entity_count, device=self.device, dtype=torch.bool)
    self.entity_is_critical[self.critical_state_node_indices] = True

    self.critical_embedding = self._init_critical_embedding()
    self.entities_emb = self._init_entity_emb()
    self.node_attentions, self.relation_attentions = self._init_attentions()
    self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    # l_e * |e|, l_a * |a|, 1 * l_t
    if self.use_single_timing_label:
      num_of_timing_labels = 1
    else:
      num_of_timing_labels = self.timing_relations.shape[0]
    self.num_of_labels = self.event_nodes.shape[0] + len(self.graph._attributes) + num_of_timing_labels
    self.event_labels = torch.arange(self.event_nodes.shape[0], device=self.device)
    self.attribute_labels = torch.arange(len(self.graph._attributes), device=self.device) + self.event_nodes.shape[0]
    self.timing_labels = torch.arange(num_of_timing_labels, device=self.device) + self.event_nodes.shape[0] + len(self.graph._attributes)

    # init label embeddings
    self.labels_emb = self._init_label_emb()

    self.relation_to_label = self._init_relation_to_label()

    positive_key_event_edges = self.relation_tuples[self.key_event_relations]
    self.positive_key_event_triplets = torch.stack((positive_key_event_edges[:, 0], self.key_event_relations, positive_key_event_edges[:, 1]), dim=1)

    positive_timing_edges = self.relation_tuples[self.timing_relations]
    self.positive_timing_triplets = torch.stack((positive_timing_edges[:, 0], self.timing_relations, positive_timing_edges[:, 1]), dim=1)

    # better GPU index
    self.triplet_node_index = torch.tensor([0,2], dtype=self.graph._index_dtype, device=self.device)
    self.triplet_relation_index = torch.tensor([1], dtype=self.graph._index_dtype, device=self.device)
    

  def _init_critical_embedding(self):
    embedding = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)
    embedding.uniform_(-1, 1)
    embedding.data.div_(embedding.data.norm(p=2, dim=0))
    critical_embedding = nn.Parameter(embedding)
    critical_embedding.requires_grad_(True)

    return critical_embedding

  def _init_entity_emb(self):
    entity_embs = nn.Embedding(num_embeddings=self.entity_count,
                                embedding_dim=self.embedding_dim,
                                device=self.device,
                                dtype=torch.float32)
    entity_embs.requires_grad_(False)
    uniform_range = 1
    entity_embs.weight.data.uniform_(-uniform_range, uniform_range)
    entity_embs.weight.data.div_(entity_embs.weight.data.norm(p=2, dim=1, keepdim=True))
    return entity_embs

  def _init_label_emb(self):
    label_embs = nn.Embedding(num_embeddings=self.num_of_labels,
                                 embedding_dim=self.embedding_dim,
                                 device=self.device,
                                 dtype=torch.float32)
    label_embs.requires_grad_(False)
    uniform_range = 1
    label_embs.weight.data.uniform_(-uniform_range, uniform_range)
    label_embs.weight.data.div_(label_embs.weight.data.norm(p=2, dim=1, keepdim=True))
    return label_embs

  def _init_relation_to_label(self):
    relation_to_label = torch.zeros(self.relation_count, dtype=self.graph._index_dtype, device=self.device)

    if self.use_single_timing_label:
      relation_to_label[self.timing_relations] = self.num_of_labels - 1
    else:
      relation_to_label[self.timing_relations] = self.timing_labels

    for attr_id in range(self.key_event_relations_per_key.shape[1]):
      key_event_relations_of_attr = self.key_event_relations_per_key[:, attr_id]
      relation_to_label[key_event_relations_of_attr] = self.attribute_labels[attr_id]
    
    label_per_event_state_relation = self.event_labels[self.node_to_scope[self.relation_tuples[self.event_state_relations, 0]]]
    relation_to_label[self.event_state_relations] = label_per_event_state_relation

    return relation_to_label

  def _init_attentions(self):
    assert self.graph._vertex_count == self.entity_count
    relation_attentions = torch.zeros(self.relation_count, device=self.device, dtype=torch.float32)
    relation_attentions[self.event_state_relations] = 1 / len(self.graph._attributes)
    relation_att = nn.Parameter(relation_attentions)
    relation_att.requires_grad_(False)

    node_attentions = torch.zeros(self.entity_count, device=self.device, dtype=torch.float32)
    node_attentions[self.event_nodes] = 1 / len(self.graph._attributes)
    node_att = nn.Parameter(node_attentions)
    node_att.requires_grad_(False)
    return node_att, relation_att

  def _init_relation_to_scope(self):
    assert self.graph._edges.shape[0] == self.relation_count
    relation_scopes = torch.zeros(self.relation_count, dtype=self.graph._index_dtype, device=self.device)
    relation_scopes[self.key_event_relations] = torch.arange(self.key_event_relations.shape[0], dtype=self.graph._index_dtype, device=self.device)
    relation_scopes[self.timing_relations] = torch.arange(self.timing_relations.shape[0], dtype=self.graph._index_dtype, device=self.device)
    relation_scopes[self.event_state_relations] = torch.arange(self.event_state_relations.shape[0], dtype=self.graph._index_dtype,  device=self.device)

    return relation_scopes

  def _init_node_to_scope(self):
    assert self.graph._vertex_count == self.entity_count
    node_scopes = torch.zeros(self.entity_count, dtype=self.graph._index_dtype, device=self.device)
    node_scopes[self.key_nodes] = torch.arange(self.key_nodes.shape[0], dtype=self.graph._index_dtype, device=self.device)
    node_scopes[self.event_nodes] = torch.arange(self.event_nodes.shape[0], dtype=self.graph._index_dtype, device=self.device)
    node_scopes[self.state_holder_nodes] = torch.arange(self.state_holder_nodes.shape[0], dtype=self.graph._index_dtype,  device=self.device)

    return node_scopes

  def set_output_dir(self, path):
    self.model_output_path = path
    self.training_start = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    Path(self.model_output_path).mkdir(parents=True, exist_ok=True)
    config_path = self.model_output_path + '/config.json'

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    config = {
      'git-commit': sha,
      'graph': {
        'delta': self.graph._delta,
        'attributes': self.graph._attributes,
        'time_key': self.graph._time_key,
        'critical_key': self.graph._critical_key,
        'sub_stream_key': self.graph._sub_stream_key
      },
      'embedding_dim': self.embedding_dim,
      'learning_rate': self.lr,
      'training_start': self.training_start,
    }
    with open(config_path, 'w') as f:
      json.dump(config, f, indent=2)
    
    print("model config successfully stored at ", config_path)

  def save_model(self, epoch=-1):
    model_path = self.model_output_path + '/model'
    if epoch != -1:
      model_path += '_' + str(epoch)
    model_path += '.bin'

    torch.save(self.state_dict(), model_path)

    print("model successfully stored at ", model_path)
    if epoch == -1:
      self.loss_report_file_handle().close()
  
  def load_model(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def loss_report(self, epoch, timing_test_set_triplets, key_event_test_set_triplets):
    torch.cuda.empty_cache()
    (total_loss, s_ke_loss_pos, s_t_loss_pos) = self.total_loss(
        timing_test_set_triplets, key_event_test_set_triplets)

    loss_report = ['{0: 05d}'.format(epoch), str(total_loss.item()), str(
        s_ke_loss_pos.item()), str(s_t_loss_pos.item())]
    self.loss_report_file_handle().write('\t'.join(loss_report) + '\n')
    self.loss_report_file_handle().flush()

  def loss_report_file_handle(self):
    if self.loss_report_file is None:
      loss_report_path = self.model_output_path + '/train_results.csv'
      self.loss_report_file = open(loss_report_path, 'w')
      self.loss_report_file.write('\t'.join(['epoch', 'total_loss', 's_ke_loss', 's_t_loss']) + '\n')
    return self.loss_report_file

  def test_report_file_handles(self):
    if self.test_report_files is None:
      total_report_file = self.model_output_path + '/total_test.csv'
      timing_report_file = self.model_output_path + '/timing_test.csv'
      key_event_report_file = self.model_output_path + '/key_event_test.csv'

      test_report_files = (total_report_file, timing_report_file, key_event_report_file)
      result = []
      for report_file in test_report_files:
        report_file_handle = open(report_file, 'w')
        report_file_handle.write(
            '\t'.join(['epoch', 'hits_at_k', 'mrr_score']) + '\n')
        result.append(report_file_handle)
      self.test_report_files = result
    return self.test_report_files

  def total_test_report_file_handle(self):
    return self.test_report_file_handles()[0]
  
  def timing_test_report_file_handle(self):
    return self.test_report_file_handles()[1]

  def key_event_test_report_file_handle(self):
    return self.test_report_file_handles()[2]

  def f_sigma(self, state_indices, events_per_state_embs, events_per_state_atts, event_state_per_state_atts, event_state_label_embs, return_tail_ends=False):
    state_is_critical = self.entity_is_critical[state_indices]

    if self.verify:
      event_state_relations = self.event_state_relations_per_state[self.node_to_scope[state_indices]]
      assert (self.labels_emb(self.relation_to_label[event_state_relations]) == event_state_label_embs).all()

    normed_events_per_state_atts = events_per_state_atts.div(events_per_state_atts.sum(dim=1, keepdim=True))
    normed_event_state_per_state_atts = event_state_per_state_atts.div(event_state_per_state_atts.sum(dim=1, keepdim=True))

    if self.additional_attentions_on_edges:
      attention_per_state = events_per_state_atts * event_state_per_state_atts
      normed_attention_per_state = attention_per_state.div(attention_per_state.sum(dim=1, keepdim=True))
    else:
      normed_attention_per_state = normed_events_per_state_atts

    tail_ends = (normed_attention_per_state[:, :, None]
                 * (events_per_state_embs + event_state_label_embs))

    if return_tail_ends:
      return tail_ends

    f_sigma_non_critical = tail_ends.sum(dim=1)
    
    if self.norm_f_sigma:
      normed_f_sigma_non_critical = f_sigma_non_critical.div(f_sigma_non_critical.norm(p=2, dim=1, keepdim=True))

      f_sigma = torch.where(state_is_critical.view(-1, 1),
                self.critical_embedding.expand(state_indices.shape[0], self.embedding_dim),
                normed_f_sigma_non_critical)
    else:
      f_sigma = torch.where(state_is_critical.view(-1, 1),
                self.critical_embedding.expand(state_indices.shape[0], self.embedding_dim),
                f_sigma_non_critical)

    return f_sigma

  def f_sigma_by_index(self, state_indices, return_tail_ends=False):
    state_idx = torch.bucketize(state_indices, self.state_holder_nodes)

    events_per_state = self.event_nodes_per_state[state_idx]
    event_state_relations_per_state = self.event_state_relations_per_state[state_idx]
    event_state_label_embs = self.labels_emb(self.relation_to_label[event_state_relations_per_state])

    events_per_state_embs = self.entities_emb(events_per_state)
    events_per_state_atts = self.node_attentions[events_per_state]
    event_state_per_state_atts = self.relation_attentions[event_state_relations_per_state]

    return self.f_sigma(state_indices, events_per_state_embs, events_per_state_atts, event_state_per_state_atts, event_state_label_embs, return_tail_ends=return_tail_ends)

  def f_sigma_by_index_shuffled(self, state_indices):
    state_idx = torch.bucketize(state_indices, self.state_holder_nodes)

    events_per_state = self.event_nodes_per_state[state_idx]
    event_state_relations_per_state = self.event_state_relations_per_state[state_idx]

    state_is_critical = self.entity_is_critical[state_indices]

    events_per_state_copy = events_per_state[state_is_critical.logical_not()]
    event_state_relations_per_state_copy = event_state_relations_per_state[state_is_critical.logical_not(
    )]

    for i in range(events_per_state_copy.shape[0]):
      tuple_to_swap_with_per_attr = torch.randint(
          i, events_per_state_copy.shape[0], (events_per_state_copy.shape[1], ))
      for j in range(events_per_state_copy.shape[1]):
        a_node = events_per_state_copy[i, j]
        b_node = events_per_state_copy[tuple_to_swap_with_per_attr[j], j]
        events_per_state_copy[i, j] = b_node
        events_per_state_copy[tuple_to_swap_with_per_attr[j], j] = a_node

        a_rel = event_state_relations_per_state_copy[i, j]
        b_rel = event_state_relations_per_state_copy[tuple_to_swap_with_per_attr[j], j]
        event_state_relations_per_state_copy[i, j] = b_rel
        event_state_relations_per_state_copy[tuple_to_swap_with_per_attr[j], j] = a_rel
    
    events_per_state[state_is_critical.logical_not()] = events_per_state_copy
    event_state_relations_per_state[state_is_critical.logical_not()] = event_state_relations_per_state_copy

    event_state_label_embs = self.labels_emb(self.relation_to_label[event_state_relations_per_state])
    events_per_state_embs = self.entities_emb(events_per_state)
    events_per_state_atts = self.node_attentions[events_per_state]
    event_state_per_state_atts = self.relation_attentions[event_state_relations_per_state]

    return self.f_sigma(state_indices, events_per_state_embs, events_per_state_atts, event_state_per_state_atts, event_state_label_embs)

  # lower values -> more important
  # measure distance e+r_e if attention_would be 100%
  def attribute_influence_per_state(self, state_indices):
    state_idx = torch.bucketize(state_indices, self.state_holder_nodes)

    tail_ends = (self.entities_emb(self.event_nodes_per_state[state_idx]) + self.labels_emb(
        self.relation_to_label[self.event_state_relations_per_state[state_idx]]))
    tail_ends_normed = tail_ends.div(tail_ends.norm(p=2, dim=1, keepdim=True))

    state_embs = self.f_sigma_by_index(state_indices)

    return (tail_ends_normed - state_embs[:, None, :]).norm(p=2, dim=2)

  def _distance(self, head_embs, relation_embs, tail_embs):
    return (head_embs + relation_embs - tail_embs).norm(dim=1, p=self.norm)

  def dist_predict(self, triplets):
    with torch.no_grad():
      assert triplets.shape[1] == 3
      heads = triplets[:, 0]
      relations = triplets[:, 1]
      tails = triplets[:, 2]

      is_key_event_relation = (
          self.graph._relation_types[relations] == GraphTypes.KEY_EVENT_RELATION_TYPE)
      is_timing_relation = (
          self.graph._relation_types[relations] == GraphTypes.TIME_RELATION_TYPE)
      assert is_key_event_relation.sum() + is_timing_relation.sum() == len(relations)

      keys = heads[is_key_event_relation]
      key_event_relations = relations[is_key_event_relation]
      events = tails[is_key_event_relation]

      state_heads = heads[is_timing_relation]
      timing_relations = relations[is_timing_relation]
      state_tails = tails[is_timing_relation]

      result = torch.zeros(
          triplets.shape[0], device=self.device, dtype=torch.float32)
      key_embs = self.entities_emb(keys)
      attribute_label_embs = self.labels_emb(self.relation_to_label[key_event_relations])
      event_embs = self.entities_emb(events)

      result[is_key_event_relation] = self._distance(
          key_embs, attribute_label_embs, event_embs)

      state_head_embs = self.f_sigma_by_index(state_heads)
      timing_label_embs = self.labels_emb(self.relation_to_label[timing_relations])
      state_tail_embs = self.f_sigma_by_index(state_tails)

      result[is_timing_relation] = self._distance(state_head_embs, timing_label_embs, state_tail_embs)

    return result

  def _corrupt_state_indices(self, positive_state_indices):
    negative_state_indices = positive_state_indices.clone()

    substream_id = self.graph._sub_stream_ids[positive_state_indices[0, 0]]
    substream_states = (self.graph._sub_stream_ids == substream_id).nonzero(as_tuple=True)[0]
    corrupt_the_source = torch.randint(high=2, size=(negative_state_indices.shape[0],), device=self.device, dtype=torch.bool)
    corrupt_the_target = corrupt_the_source.logical_not()
    corrupt_the_source_amount = torch.sum(corrupt_the_source)
    corrupt_the_target_amount = negative_state_indices.shape[0] - corrupt_the_source_amount

    random_sources = torch.randint(low=0, high=len(
        substream_states), size=(corrupt_the_source_amount,))
    random_targets = torch.randint(low=0, high=len(
        substream_states), size=(corrupt_the_target_amount,))

    negative_state_indices[corrupt_the_source, 0] = substream_states[random_sources]
    negative_state_indices[corrupt_the_target, 1] = substream_states[random_targets]
    
    return negative_state_indices

  def _corrupt_key_event_indices(self, positive_key_event_indices):
    negative_key_event_indices = positive_key_event_indices.clone()
    corrupt_the_source = torch.randint(high=2, size=(
        positive_key_event_indices.shape[0],), device=self.device, dtype=torch.bool)
    corrupt_the_target = corrupt_the_source.logical_not()
    corrupt_the_source_amount = torch.sum(corrupt_the_source)
    corrupt_the_target_amount = positive_key_event_indices.shape[0] - corrupt_the_source_amount

    random_sources = torch.randint(low=0, high=len(
        self.key_nodes), size=(corrupt_the_source_amount,))
    random_targets = torch.randint(low=0, high=len(
        self.event_nodes), size=(corrupt_the_target_amount,))

    negative_key_event_indices[corrupt_the_source, 0] = self.key_nodes[random_sources]
    negative_key_event_indices[corrupt_the_target, 1] = self.event_nodes[random_targets]

    return negative_key_event_indices

  def forward(self, timing_triplet_indices):
    timing_indices = self.positive_timing_triplets[timing_triplet_indices,
                                                   self.triplet_relation_index]
    positive_state_indices = self.positive_timing_triplets[timing_triplet_indices.view(-1, 1),
                                                           self.triplet_node_index]

    negative_state_indices = self._corrupt_state_indices(positive_state_indices)
    negative_state_indices = negative_state_indices.view(-1)
    positive_state_indices = positive_state_indices.view(-1)

    all_state_indices = torch.cat(
        (positive_state_indices, negative_state_indices))
    state_indices = torch.unique(all_state_indices)
    state_indices_scoped = self.node_to_scope[state_indices]

    key_event_indices = torch.unique(self.key_event_relations_per_key[self.node_to_scope[positive_state_indices]])
    key_event_indices_scoped = self.relation_to_scope[key_event_indices]

    if self.verify:
      assert (self.relation_tuples[key_event_indices, 1] == self.event_nodes_per_state[self.node_to_scope[positive_state_indices]].view(-1)).all()

    positive_key_event_indices = self.positive_key_event_triplets[key_event_indices_scoped][:, self.triplet_node_index]
    negative_key_event_indices = self._corrupt_key_event_indices(positive_key_event_indices)
    positive_key_indices = positive_key_event_indices[:, 0].contiguous()
    positive_event_indices_ske = positive_key_event_indices[:, 1].contiguous()
    
    negative_key_indices = negative_key_event_indices[:, 0].contiguous()
    negative_event_indices_ske = negative_key_event_indices[:, 1].contiguous()
    
    all_key_indices = torch.cat((positive_key_indices, negative_key_indices))
    key_indices = torch.unique(all_key_indices)

    event_state_indices = self.event_state_relations_per_state[self.node_to_scope[state_indices]].view(-1)
    event_indices_st = self.relation_tuples[event_state_indices, 0]
    all_event_indices = torch.cat(
        (positive_event_indices_ske, negative_event_indices_ske, event_indices_st))
    event_indices = torch.unique(all_event_indices)

    event_atts = self.node_attentions[event_indices]
    event_state_atts = self.relation_attentions[event_state_indices]

    key_embs = self.entities_emb(key_indices)
    event_embs = self.entities_emb(event_indices)

    event_state_labels = self.relation_to_label[event_state_indices]
    event_state_label_indices = event_state_labels.unique()
    event_state_labels_local_scoped = torch.bucketize(event_state_labels, event_state_label_indices)
    event_state_label_embs = self.labels_emb(event_state_label_indices)

    if self.verify:
      assert (event_state_label_indices[event_state_labels_local_scoped] == event_state_labels).all()

    key_event_labels = self.relation_to_label[key_event_indices]
    key_event_label_indices = key_event_labels.unique()
    key_event_label_indices_local_scoped = torch.bucketize(
        key_event_labels, key_event_label_indices)
    key_event_label_embs = self.labels_emb(key_event_label_indices)

    if self.verify:
      assert (key_event_label_indices[key_event_label_indices_local_scoped] == key_event_labels).all()

    timing_labels = self.relation_to_label[timing_indices]
    timing_label_indices = timing_labels.unique()
    timing_label_indices_local_scoped = torch.bucketize(
        timing_labels, timing_label_indices)
    timing_label_embs = self.labels_emb(timing_label_indices)

    if self.verify:
      assert (timing_label_indices[timing_label_indices_local_scoped] == timing_labels).all()

    needs_grad = [event_atts, event_state_atts, key_embs, event_embs,
                  event_state_label_embs, key_event_label_embs, timing_label_embs]

    events_per_state = self.event_nodes_per_state[state_indices_scoped]
    events_per_state_local_scope = torch.bucketize(
        events_per_state, event_indices)
    event_state_relations_per_state = self.event_state_relations_per_state[state_indices_scoped]
    event_state_relations_per_state_local_scope = torch.bucketize(
        event_state_relations_per_state, event_state_indices)

    if self.verify:
      assert (events_per_state == event_indices[events_per_state_local_scope]).all()
      assert (event_state_relations_per_state == event_state_indices[event_state_relations_per_state_local_scope]).all()

    for param in needs_grad:
      param.requires_grad_(True)
      param.grad = None
    self.critical_embedding.grad = None

    if self.verify:
      assert (self.node_attentions[self.event_nodes_per_state[self.node_to_scope[state_indices]]] == event_atts[events_per_state_local_scope]).all()
      assert (self.relation_attentions[self.event_state_relations_per_state[self.node_to_scope[state_indices]]] == event_state_atts[event_state_relations_per_state_local_scope]).all()
      assert (self.entities_emb(self.event_nodes_per_state[self.node_to_scope[state_indices]]) == event_embs[events_per_state_local_scope]).all()
      assert (self.labels_emb(self.relation_to_label[self.event_state_relations_per_state[self.node_to_scope[state_indices]]]) == event_state_label_embs[event_state_labels_local_scoped[event_state_relations_per_state_local_scope]]).all()
      
      assert (event_indices[events_per_state_local_scope] == self.event_nodes_per_state[self.node_to_scope[state_indices]]).all()
      assert (event_state_indices[event_state_relations_per_state_local_scope] == self.event_state_relations_per_state[self.node_to_scope[state_indices]]).all()


    state_embs = self.f_sigma(
        state_indices,
        event_embs[events_per_state_local_scope],
        event_atts[events_per_state_local_scope],
        event_state_atts[event_state_relations_per_state_local_scope],
        event_state_label_embs[event_state_labels_local_scoped[event_state_relations_per_state_local_scope]])

    if self.verify:
      assert (state_embs == self.f_sigma_by_index(state_indices)).all()

    positive_timing_tuples_local_scoped = torch.bucketize(
        positive_state_indices, state_indices).view(-1, 2)
    positive_state_head_embs = state_embs[positive_timing_tuples_local_scoped[:, 0]]
    positive_state_tail_embs = state_embs[positive_timing_tuples_local_scoped[:, 1]]

    negative_timing_tuples_local_scoped = torch.bucketize(
        negative_state_indices, state_indices).view(-1, 2)
    negative_state_head_embs = state_embs[negative_timing_tuples_local_scoped[:, 0]]
    negative_state_tail_embs = state_embs[negative_timing_tuples_local_scoped[:, 1]]

    if self.verify:
      assert (state_indices[negative_timing_tuples_local_scoped] == negative_state_indices).all()
      assert (state_indices[positive_timing_tuples_local_scoped] == positive_state_indices).all()

      assert (self.labels_emb(self.relation_to_label[timing_indices]) == timing_label_embs[timing_label_indices_local_scoped]).all()
      assert (self.f_sigma_by_index(self.positive_timing_triplets[self.relation_to_scope[timing_indices], 0]) == positive_state_head_embs).all()
      assert (self.f_sigma_by_index(self.positive_timing_triplets[self.relation_to_scope[timing_indices], 2]) == positive_state_tail_embs).all()

      assert (self.f_sigma_by_index(positive_state_indices[0].view(1)) == positive_state_head_embs).all()
      assert (self.f_sigma_by_index(positive_state_indices[1].view(1)) == positive_state_tail_embs).all()
      assert (self.f_sigma_by_index(negative_state_indices[0].view(1)) == negative_state_head_embs).all()
      assert (self.f_sigma_by_index(negative_state_indices[1].view(1)) == negative_state_tail_embs).all()

    s_t_pos_loss = (positive_state_head_embs + timing_label_embs[timing_label_indices_local_scoped] -
                    positive_state_tail_embs).norm(p=self.norm, dim=1)
    s_t_neg_loss = (negative_state_head_embs + timing_label_embs[timing_label_indices_local_scoped] -
                    negative_state_tail_embs).norm(p=self.norm, dim=1)

    s_t_loss = self.loss(s_t_pos_loss, s_t_neg_loss)

    positive_keys_local_scoped = torch.bucketize(
        positive_key_indices, key_indices)
    positive_events_local_scoped = torch.bucketize(
        positive_event_indices_ske, event_indices)
    positive_key_embs = key_embs[positive_keys_local_scoped]
    positive_event_embs = event_embs[positive_events_local_scoped]

    negative_keys_local_scoped = torch.bucketize(
        negative_key_indices, key_indices)
    negative_events_local_scoped = torch.bucketize(
        negative_event_indices_ske, event_indices)
    negative_key_embs = key_embs[negative_keys_local_scoped]
    negative_event_embs = event_embs[negative_events_local_scoped]

    if self.verify:
      assert (key_indices[positive_keys_local_scoped] == positive_key_indices).all()
      assert (event_indices[positive_events_local_scoped] == positive_event_indices_ske).all()
      assert (key_indices[negative_keys_local_scoped] == negative_key_indices).all()
      assert (event_indices[negative_events_local_scoped] == negative_event_indices_ske).all()

      assert (self.labels_emb(self.relation_to_label[key_event_indices]) == key_event_label_embs[key_event_label_indices_local_scoped]).all()
      assert (self.entities_emb(self.positive_key_event_triplets[self.relation_to_scope[key_event_indices], 0]) == positive_key_embs).all()
      assert (self.entities_emb(self.positive_key_event_triplets[self.relation_to_scope[key_event_indices], 2]) == positive_event_embs).all()

    s_ke_pos_loss = (positive_key_embs + key_event_label_embs[key_event_label_indices_local_scoped] -
                    positive_event_embs).norm(p=self.norm, dim=1)
    s_ke_neg_loss = (negative_key_embs + key_event_label_embs[key_event_label_indices_local_scoped] -
                    negative_event_embs).norm(p=self.norm, dim=1)

    s_ke_loss = self.loss(s_ke_pos_loss, s_ke_neg_loss)

    if self.use_mean_loss:
      cat_loss = torch.cat((s_ke_loss, s_t_loss))
      total_loss = cat_loss.mean()
    else:
      total_loss = s_ke_loss.sum() + self.gamma * s_t_loss.sum()
    total_loss.backward()

    with torch.no_grad():
      event_state_label_embs -= event_state_label_embs.grad * self.lr
      key_event_label_embs -= key_event_label_embs.grad * self.lr
      timing_label_embs -= timing_label_embs.grad * self.lr
      event_atts -= event_atts.grad * self.lr
      if self.additional_attentions_on_edges:
        event_state_atts -= event_state_atts.grad * self.lr
      if self.norm_edges_except_event_state:
        key_event_label_embs.div_(
            key_event_label_embs.norm(p=2, dim=1, keepdim=True))
        timing_label_embs.div_(timing_label_embs.norm(p=2, dim=1, keepdim=True))
      if self.norm_event_state_edges:
        event_state_label_embs.div_(
            event_state_label_embs.norm(p=2, dim=1, keepdim=True))

      # norm attentions
      event_atts.clamp_(min=0)  # ensure attentions are bigger than 0
      event_state_atts.clamp_(min=0)  # ensure attentions are bigger than 0
      # normed_attentions_per_state = event_state_atts[event_state_relations_per_state_local_scope].div(
      #     event_state_atts[event_state_relations_per_state_local_scope].sum(dim=1, keepdim=True))
      # event_state_atts[event_state_relations_per_state_local_scope] = normed_attentions_per_state

      key_embs -= key_embs.grad * self.lr
      event_embs -= event_embs.grad * self.lr

      key_embs.data.div_(key_embs.data.norm(p=2, dim=1, keepdim=True))
      event_embs.data.div_(event_embs.data.norm(p=2, dim=1, keepdim=True))

      if self.verify:
        assert (self.relation_to_label[timing_indices] == timing_label_indices).all()
        assert (self.relation_to_label[key_event_indices].unique() == key_event_label_indices).all()
        assert (self.relation_to_label[event_state_indices].unique() == event_state_label_indices).all()

      self.labels_emb.weight[key_event_label_indices] = key_event_label_embs
      self.labels_emb.weight[event_state_label_indices] = event_state_label_embs
      self.labels_emb.weight[timing_label_indices] = timing_label_embs

      self.node_attentions[event_indices] = event_atts
      self.relation_attentions[event_state_indices] = event_state_atts

      self.entities_emb.weight[key_indices] = key_embs
      self.entities_emb.weight[event_indices] = event_embs

      self.critical_embedding -= self.critical_embedding.grad * self.lr
      self.critical_embedding.data.div_(self.critical_embedding.data.norm(p=2, dim=0))

    for param in needs_grad:
      param.requires_grad_(False)

  def loss(self, positive_distances, negative_distances):
    target = torch.tensor([-1], dtype=torch.long, device=self.device)
    if self.pow:
      if self.use_criterion:
        return self.criterion(positive_distances.pow(2), negative_distances.pow(2), target)
      else:
        return positive_distances.pow(2) - negative_distances.pow(2)
    else:
      if self.use_criterion:
        return self.criterion(positive_distances, negative_distances, target)
      else:
        return positive_distances - negative_distances

  def total_loss(self, timing_test_set_triplets, key_event_test_set_triplets):
    with torch.no_grad():
      positive_distances_s_t = self.dist_predict(timing_test_set_triplets)
      positive_distances_s_ke = self.dist_predict(key_event_test_set_triplets)

      results_s_t = torch.zeros(
          (timing_test_set_triplets.shape[0]), dtype=torch.float32, device=self.device)
      results_s_ke = torch.zeros((key_event_test_set_triplets.shape[0]), dtype=torch.float32, device=self.device)

      for i, test_triplets in tqdm(enumerate(torch.split(timing_test_set_triplets, 1)), desc="timing_loss", total=timing_test_set_triplets.shape[0]):
        corrupted_triplets = Test.all_timing_triplet_variations_of(self, test_triplets, sub_stream_scoped=True)

        predictions = self.dist_predict(corrupted_triplets)

        results_s_t[i] = self.loss(positive_distances_s_t[i].view(1), predictions).sum()
      
      for i, test_triplets in tqdm(enumerate(torch.split(key_event_test_set_triplets, 1)), desc="key_event_loss", total=key_event_test_set_triplets.shape[0]):
        corrupted_triplets_key, corrupted_triplets_event = Test.all_key_event_triplet_variations_of(self, test_triplets)

        predictions_key = self.dist_predict(corrupted_triplets_key)
        predictions_event = self.dist_predict(corrupted_triplets_event)
        predictions = torch.cat((predictions_key, predictions_event))

        results_s_ke[i] = self.loss(positive_distances_s_ke[i].view(1), predictions).sum()

      s_ke_loss = results_s_t
      s_t_loss = results_s_ke

      if self.use_mean_loss:
        cat_loss = torch.cat((s_ke_loss, s_t_loss))
        total_loss = cat_loss.mean()
      else:
        total_loss = s_ke_loss.sum() + self.gamma * s_t_loss.sum()

    return total_loss, positive_distances_s_ke.mean(), positive_distances_s_t.mean()

  def sample_negative_timing_edges(self, triplets):
    result = triplets.clone()

    substream_ids = torch.arange(len(self.graph._sub_stream_bag), dtype=self.graph._index_dtype, device=self.device)

    total_edges = 0

    for substream_id in substream_ids:
      substream_states = (self.graph._sub_stream_ids == substream_id).nonzero(as_tuple=True)[0]
      substream_timing_edges = (self.graph._sub_stream_ids[triplets[:, 0]] == substream_id)
      substream_negative_samples = self.sample_negative(triplets[substream_timing_edges], substream_states, substream_states)
      result[substream_timing_edges] = substream_negative_samples

      total_edges += substream_negative_samples.shape[0]
    
    assert total_edges == triplets.shape[0]

    return result

  def sample_negative(self, triplets, source_base, target_base):
    result = triplets.clone()

    corrupt_the_source = torch.randint(
        low=0, high=2, size=(triplets.shape[0],), dtype=torch.bool)
    corrupt_the_target = corrupt_the_source.logical_not()
    corrupt_the_source_amount = torch.sum(corrupt_the_source)
    corrupt_the_target_amount = triplets.shape[0] - corrupt_the_source_amount

    random_sources = torch.randint(low=0, high=len(
        source_base), size=(corrupt_the_source_amount,))
    random_targets = torch.randint(low=0, high=len(
        target_base), size=(corrupt_the_target_amount,))

    result[corrupt_the_source, 0] = source_base[random_sources]
    result[corrupt_the_target, 2] = target_base[random_targets]

    return result
  def test(self, epoch, timing_test_set_triplets, key_event_test_set_triplets, event_k=1, state_k=1):
    with torch.no_grad():
      timing_chunk_size = 1
      key_event_chunk_size = 1

      timing_hits_at_k_score = 0
      timing_mrr_score = 0
      timing_total_examples_count = 0

      key_event_hits_at_k_score = 0
      key_event_mrr_score = 0
      key_event_total_examples_count = 0

      for test_triplets in tqdm(torch.split(timing_test_set_triplets, timing_chunk_size), desc="timing_test"):
        hits_at_k, mrr, examples_count = Test.test_timing(self, test_triplets, k=state_k)

        timing_hits_at_k_score += hits_at_k
        timing_mrr_score += mrr
        timing_total_examples_count += examples_count

      for test_triplets in tqdm(torch.split(key_event_test_set_triplets, key_event_chunk_size), desc="key_event_test"):
        hits_at_k, mrr, examples_count = Test.test_key_event(self, test_triplets, k=event_k)

        key_event_hits_at_k_score += hits_at_k
        key_event_mrr_score += mrr
        key_event_total_examples_count += examples_count

    total_hits_at_k_score = timing_hits_at_k_score + key_event_hits_at_k_score
    total_mrr_score = timing_mrr_score + key_event_mrr_score
    total_examples_count = timing_total_examples_count + key_event_total_examples_count

    total_scores = (epoch, total_hits_at_k_score / total_examples_count *
                    100, total_mrr_score / total_examples_count * 100)

    timing_scores = (epoch, timing_hits_at_k_score / timing_total_examples_count *
                    100, timing_mrr_score / timing_total_examples_count * 100)

    key_event_scores = (epoch, key_event_hits_at_k_score / key_event_total_examples_count *
                        100, key_event_mrr_score / key_event_total_examples_count * 100)

    self.total_test_report_file_handle().write('\t'.join([str(x) for x in total_scores]) + '\n')
    self.total_test_report_file_handle().flush()

    self.timing_test_report_file_handle().write('\t'.join([str(x) for x in timing_scores]) + '\n')
    self.timing_test_report_file_handle().flush()

    self.key_event_test_report_file_handle().write('\t'.join([str(x) for x in key_event_scores]) + '\n')
    self.key_event_test_report_file_handle().flush()

    return total_mrr_score / total_examples_count * 100
