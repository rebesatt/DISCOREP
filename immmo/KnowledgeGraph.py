#!/usr/bin/env python3

from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import torch
from ordered_set import OrderedSet
from scipy.sparse import csc_matrix
from operator import itemgetter
from .constants.GraphTypes import GraphTypes


def index_of_a_in_b(a, b):
  b_indices = torch.where(torch.isin(b, a))[0]
  b_values = b[b_indices]
  return b_indices[b_values.argsort()[a.argsort().argsort()]]

def sizeof_fmt(num, suffix="B"):
  for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
    if abs(num) < 1024.0:
      return f"{num:3.1f}{unit}{suffix}"
    num /= 1024.0
  return f"{num:.1f}Yi{suffix}"

class KnowledgeGraph():
  def __init__(self, time_key='timestamp', critical_key='critical', sub_stream_key=None, attributes=[], delta=0, device='cpu') -> None:
    self._time_key = time_key
    self._critical_key = critical_key
    self._sub_stream_key = sub_stream_key
    self._attributes = list(attributes)

    self._index_dtype = torch.int64
    self._device = torch.device(device if torch.cuda.is_available() else 'cpu')

    self._vertex_count = 0
    self._edges = torch.zeros((0, 2), dtype=self._index_dtype, device=self._device)
    self._edge_index_map = None

    if time_key in self._attributes:
      # ensure that the time key is not part of the attributes
      self._attributes.remove(time_key)
    
    if critical_key in self._attributes:
      # ensure that the critical key is not part of the attributes
      self._attributes.remove(critical_key)

    if sub_stream_key in self._attributes:
      # ensure that the sub stream key is not part of the attributes
      self._attributes.remove(sub_stream_key)

    self._delta = delta

    self._sub_stream_bag = OrderedSet()
    self._sub_stream_ids = torch.zeros(0, device=self._device, dtype=torch.int32)

    self._value_bag = OrderedSet()
    self._attr_values = {
      attr_id: OrderedSet() for attr_id in range(len(self._attributes))
    }

    self._event_state_relations_by_state = {}
    self._timing_relations_by_tail = {}
    self._key_event_relations_by_key = {}

    self._key_by_state = {}
    self._state_by_key = {}

    self._critical_indices = []

    self._node_values = torch.zeros(0, device=self._device, dtype=torch.int32)
    self._node_attr_types = torch.zeros(0, device=self._device, dtype=torch.int16)
    self._timestamps = torch.zeros(0, device=self._device, dtype=torch.int32)

    # node and relation types
    self._node_types = torch.zeros(0, device=self._device, dtype=torch.int8)
    self._relation_types = torch.zeros(0, device=self._device, dtype=torch.int8)

  def _add_vertices(self, count):
    result = torch.arange(self._vertex_count, self._vertex_count + count, device=self._device, dtype=self._index_dtype)
    self._vertex_count += count
    return result

  def _add_edges(self, edges):
    self._edges = torch.cat([self._edges, edges], dim=0)

    self._edge_index_map = csc_matrix((np.arange(1, len(self._edges) + 1), (self._edges[:, 0].numpy(), self._edges[:, 1].numpy())))

  def update_state_event_map(self):
    state_holder_nodes = self.stateHolderNodeIndices()
    event_state_relations = self.eventStateRelationIndices()
    event_state_tuples = self._edges[event_state_relations]
    self._event_state_relations_by_state = { state_node: event_state_relations[(
        event_state_tuples[:, 1] == state_node)].tolist() for state_node in tqdm(state_holder_nodes.tolist(), desc='event_state_map')}

  def update_timing_map(self):
    state_holder_nodes = self.stateHolderNodeIndices()
    timing_relations = self.timeRelationIndices()
    timing_tuples = self._edges[timing_relations]
    self._timing_relations_by_tail = { state_node: timing_relations[(
        timing_tuples[:, 1] == state_node)].tolist() for state_node in tqdm(state_holder_nodes.tolist(), desc='timing_map') }

  def update_key_event_map(self):
    key_nodes = self.keyNodeIndices()
    key_event_relations = self.keyEventRelationIndices()
    key_event_tuples = self._edges[key_event_relations]
    self._key_event_relations_by_key = { key_node: key_event_relations[(
        key_event_tuples[:, 0] == key_node)].tolist() for key_node in tqdm(key_nodes.tolist(), desc='key_event_map') }

  def update_key_state_map(self):
    state_holder_nodes = self.stateHolderNodeIndices().tolist()
    key_nodes = self.keyNodeIndices().tolist()

    assert len(state_holder_nodes) == len(key_nodes)

    for state, key in tqdm(zip(state_holder_nodes, key_nodes), total=len(key_nodes), desc='key_state_map'):
      self._key_by_state[state] = key
      self._state_by_key[key] = state

  def update_relation_maps(self):
    self.update_state_event_map()
    self.update_timing_map()
    self.update_key_event_map()
    self.update_key_state_map()

  def state_node_values(self, state_holder_nodes):
    event_state_relations_per_state = self.event_state_relations_by_state(state_holder_nodes)

    event_nodes_per_state = self._edges[event_state_relations_per_state, 0]
    assert event_nodes_per_state.shape == torch.Size(
        [len(state_holder_nodes), len(self._attributes)])

    cols = {}

    for i, attr_name in enumerate(self._attributes):
      event_nodes = event_nodes_per_state[:, i]
      value_ids = self._node_values[event_nodes].tolist()
      attr_values = self._value_bag[value_ids]

      cols[attr_name] = attr_values

    return pd.DataFrame(cols)

  def critical_states(self):
    return torch.tensor(self._critical_indices, device=self._device, dtype=self._index_dtype)

  def event_state_relations_by_state(self, states):
    result = torch.tensor(itemgetter(*states.tolist())(
      self._event_state_relations_by_state), dtype=self._index_dtype, device=self._device)
    if len(states) == 1:
      return result[None, :]
    return result

  def event_nodes_by_state(self, states):
    result = torch.tensor(itemgetter(*states.tolist())(
        self._event_nodes_by_state), dtype=self._index_dtype, device=self._device)
    if len(states) == 1:
      return result[None, :]
    return result
  
  def key_event_relations_by_key(self, keys):
    result = torch.tensor(itemgetter(*keys.tolist())(
        self._key_event_relations_by_key), dtype=self._index_dtype, device=self._device)
    if len(keys) == 1:
      return result[None, :]
    return result

  def timing_relation_by_tail(self, tail):
    return torch.tensor(self._timing_relations_by_tail[tail.item()], dtype=self._index_dtype, device=self._device)

  def get_eids(self, edges):
    # non existing edges are marked as -1
    return np.asarray(self._edge_index_map[edges[:, 0].cpu().tolist(), edges[:, 1].cpu().tolist()])[0] - 1

  def timing_edge_from(self, head):
    indices = self.timeRelationIndices()
    return indices[(self._edges[indices, 0] == head).nonzero(as_tuple=True)]

  def timing_edge_to(self, tail):
    indices = self.timeRelationIndices()
    return indices[(self._edges[indices, 1] == tail).nonzero(as_tuple=True)]

  def states_by_keys(self, keys):
    result = torch.tensor(itemgetter(*keys.tolist())(
        self._state_by_key), dtype=self._index_dtype, device=self._device)
    if len(keys) == 1:
      return result[None, :]
    return result

  def keys_by_states(self, states):
    result = torch.tensor(itemgetter(*states.tolist())(
        self._key_by_state), dtype=self._index_dtype, device=self._device)
    if len(states) == 1:
      return result.view(1)
    return result

  def dump_to_file(self, file_name):
    path = file_name + '.kgraph'

    config = {
      'critical_key': self._critical_key,
      'time_key': self._time_key,
      'sub_stream_key': self._sub_stream_key,
      'sub_stream_bag': self._sub_stream_bag,
      'attributes': self._attributes,
      'delta': self._delta,
      'device': self._device,
      'index_dtype': self._index_dtype,
      'value_bag': self._value_bag,
      'attr_values': self._attr_values,
      'critical_indices': self._critical_indices,
      'graph': {
        'vertex_count': self._vertex_count,
        'edges': self._edges.cpu(),
        'edge_index_map': self._edge_index_map,
        'event_state_relation_by_state': self._event_state_relations_by_state,
        'timing_relations_by_tail': self._timing_relations_by_tail,
        'key_event_relations_by_key': self._key_event_relations_by_key,
        'key_by_state': self._key_by_state,
        'state_by_key': self._state_by_key
      },
      'nodes':{
        'sub_stream_ids': self._sub_stream_ids.cpu(),
        'types': self._node_types.cpu(),
        'values': self._node_values.cpu(),
        'attr_types': self._node_attr_types.cpu(),
        'timestamps': self._timestamps.cpu()
      },
      'relations': {
        'types': self._relation_types.cpu()
      }
    }
    print("Store graph to file", path)
    with open(path, 'wb') as f:
      pickle.dump(config, f)
    print("Successfully stored")
  
  @staticmethod
  def load_from_file(file_name, device='cpu'):
    path = file_name + '.kgraph'

    g = KnowledgeGraph()
    print("Load graph from file", path)
    with open(path, 'rb') as f:
      config = pickle.load(f)
      g._critical_key = config['critical_key']
      g._time_key = config['time_key']
      g._sub_stream_key = config['sub_stream_key']
      g._sub_stream_bag = config['sub_stream_bag']
      g._attributes = config['attributes']
      g._delta = config['delta']
      g._device = config['device'] if device is None else torch.device(
          device if torch.cuda.is_available() else 'cpu')
      g._index_dtype = config['index_dtype']
      g._value_bag = config['value_bag']
      g._attr_values = config['attr_values']
      g._critical_indices = config['critical_indices']

      graph = config['graph']
      g._vertex_count = graph['vertex_count']
      g._edges = graph['edges'].to(g._device)
      g._edge_index_map = graph['edge_index_map']
      g._timing_relations_by_tail = graph['timing_relations_by_tail']
      g._key_event_relations_by_key = graph['key_event_relations_by_key']
      g._event_state_relations_by_state = graph['event_state_relation_by_state']
      g._key_by_state = graph['key_by_state']
      g._state_by_key = graph['state_by_key']

      nodes = config['nodes']
      g._sub_stream_ids = nodes['sub_stream_ids'].to(g._device)
      g._node_types = nodes['types'].to(g._device)
      g._node_values = nodes['values'].to(g._device)
      g._node_attr_types = nodes['attr_types'].to(g._device)
      g._timestamps = nodes['timestamps'].to(g._device)

      relations = config['relations']
      g._relation_types = relations['types'].to(g._device)
    print("Successfully loaded")


    return g

  def keyNodeIndices(self) -> torch.tensor:
    return torch.where(self._node_types == GraphTypes.KEY_NODE_TYPE)[0]
  
  def keyNodeIndicesByTimestamps(self, timestamps):
    key_node_indices = self.keyNodeIndices()
    key_node_timestamps = self._timestamps[key_node_indices]
    key_node_indices_of_timestamps = index_of_a_in_b(timestamps, key_node_timestamps)
    return key_node_indices[key_node_indices_of_timestamps]
  
  def eventNodeIndices(self) -> torch.tensor:
    return torch.where(self._node_types == GraphTypes.EVENT_NODE_TYPE)[0]

  def eventNodeIndicesByAttribute(self, attr_id):
    return torch.where((self._node_types == GraphTypes.EVENT_NODE_TYPE) & (self._node_attr_types == attr_id))[0]

  def eventNodeIndicesByAttributeValues(self, attr_id, values):
    value_ids = torch.tensor(self._value_bag.index(values), device=self._device, dtype=self._node_values.dtype)
    event_node_indices = self.eventNodeIndicesByAttribute(attr_id)
    event_node_values = self._node_values[event_node_indices]
    event_node_indices_of_values = index_of_a_in_b(value_ids, event_node_values)
    return event_node_indices[event_node_indices_of_values]

  def stateHolderNodeIndices(self) -> torch.tensor:
    return torch.where(self._node_types == GraphTypes.STATE_HOLDER_NODE_TYPE)[0]
  
  def keyEventRelationIndices(self) -> torch.tensor:
    return torch.where(self._relation_types == GraphTypes.KEY_EVENT_RELATION_TYPE)[0]
  
  def eventStateRelationIndices(self) -> torch.tensor:
    return torch.where(self._relation_types == GraphTypes.EVENT_STATE_RELATION_TYPE)[0]
  
  def timeRelationIndices(self) -> torch.tensor:
    return torch.where(self._relation_types == GraphTypes.TIME_RELATION_TYPE)[0]

  def addKeyNodes(self, df) -> torch.tensor:
    key_nodes = self._add_vertices(len(df))
    assert len(key_nodes) == len(df)
    key_node_type = torch.ones(len(key_nodes), device=self._device, dtype=self._node_types.dtype) * GraphTypes.KEY_NODE_TYPE
    self._node_types = torch.cat([self._node_types, key_node_type])
    key_node_values = torch.ones(len(key_nodes), device=self._device, dtype=self._node_values.dtype) * -1
    self._node_values = torch.cat([self._node_values, key_node_values])
    key_node_attr_types = torch.ones(
        len(key_nodes), device=self._device, dtype=self._node_attr_types.dtype) * -1
    self._node_attr_types = torch.cat([self._node_attr_types, key_node_attr_types])
    key_node_timestamps = torch.tensor(
        df[self._time_key].to_list(), device=self._device, dtype=self._timestamps.dtype)
    self._timestamps = torch.cat([self._timestamps, key_node_timestamps])
    sub_stream_ids = torch.ones(len(key_nodes), device=self._device, dtype=self._sub_stream_ids.dtype) * -1
    self._sub_stream_ids = torch.cat([self._sub_stream_ids, sub_stream_ids])

    return key_nodes
  
  def addStateHolderNodes(self, df) -> torch.tensor:
    state_holder_nodes = self._add_vertices(len(df))
    assert len(state_holder_nodes) == len(df)
    critical = df[self._critical_key].values == True
    self._critical_indices += state_holder_nodes[critical].tolist()
    state_holder_node_type = torch.ones(len(state_holder_nodes), device=self._device, dtype=self._node_types.dtype) * GraphTypes.STATE_HOLDER_NODE_TYPE
    self._node_types = torch.cat([self._node_types, state_holder_node_type])
    state_holder_node_values = torch.ones(len(state_holder_nodes), device=self._device, dtype=self._node_values.dtype) * -1
    self._node_values = torch.cat([self._node_values, state_holder_node_values])
    state_holder_node_attr_types = torch.ones(len(state_holder_nodes), device=self._device, dtype=self._node_attr_types.dtype) * -1
    self._node_attr_types = torch.cat([self._node_attr_types, state_holder_node_attr_types])
    state_holder_node_time_stamps = torch.tensor(
        df[self._time_key].to_list(), device=self._device, dtype=self._timestamps.dtype)
    self._timestamps = torch.cat([self._timestamps, state_holder_node_time_stamps])

    if self._sub_stream_key is not None:
      sub_stream_identifiers = df[self._sub_stream_key].to_numpy()
      unique_sub_stream_identifiers, inverse = np.unique(
          sub_stream_identifiers, return_inverse=True)
      self._sub_stream_bag |= OrderedSet(unique_sub_stream_identifiers)
      sub_stream_ids = np.array(self._sub_stream_bag.index(unique_sub_stream_identifiers))[inverse]
    else:
      self._sub_stream_bag |= OrderedSet([0])
      sub_stream_ids = np.array(self._sub_stream_bag.index([0]) * len(state_holder_nodes))

    sub_stream_ids = torch.tensor(
        sub_stream_ids, device=self._device, dtype=self._sub_stream_ids.dtype)

    assert len(sub_stream_ids) == len(state_holder_nodes)
    self._sub_stream_ids = torch.cat([self._sub_stream_ids, sub_stream_ids])

    return state_holder_nodes

  def addEventNodes(self, attr_id, unique_values):
    # add values to value bag
    self._value_bag |= OrderedSet(unique_values)
    # get value indices
    value_indices = self._value_bag.index(unique_values)
    values_to_create = OrderedSet(value_indices) - self._attr_values[attr_id]

    if len(values_to_create) > 0:
        # add values
        self._attr_values[attr_id] |= values_to_create

        # EVENT NODES
        self._add_vertices(len(values_to_create))
        event_node_type = torch.ones(
            len(values_to_create), device=self._device, dtype=self._node_types.dtype) * GraphTypes.EVENT_NODE_TYPE
        self._node_types = torch.cat([self._node_types, event_node_type])
        event_node_values = torch.tensor(
            values_to_create, device=self._device, dtype=self._node_values.dtype)
        self._node_values = torch.cat([self._node_values, event_node_values])
        event_node_attr_types = torch.ones(len(values_to_create), device=self._device, dtype=self._node_attr_types.dtype) * attr_id
        self._node_attr_types = torch.cat([self._node_attr_types, event_node_attr_types])
        event_node_time_stamps = torch.ones(len(values_to_create), device=self._device, dtype=self._timestamps.dtype) * -1
        self._timestamps = torch.cat(
            [self._timestamps, event_node_time_stamps])
        sub_stream_ids = torch.ones(len(values_to_create), device=self._device, dtype=self._sub_stream_ids.dtype) * -1
        self._sub_stream_ids = torch.cat([self._sub_stream_ids, sub_stream_ids])

    event_node_values = self._node_values[self.eventNodeIndicesByAttribute(attr_id)]
    assert len(event_node_values) == len(torch.unique(event_node_values))
    return len(values_to_create)

  def addDataFrameToGraph(self, df):
    if self._time_key not in df.columns:
      raise Exception("Expected the column {0} to be in the given dataframe".format(self._time_key))
    
    if df[self._time_key].dtype != int:
      raise Exception("Expected the time key {0} to be an integer".format(self._time_key))

    if self._delta <= 0:
      print("Attention, delta <= 0, there will be no timing relationships in the graph")

    # KEY NODES
    key_nodes = self.addKeyNodes(df)

    # STATE HOLDER NODES
    state_holder_nodes = self.addStateHolderNodes(df)

    key_event_edges = torch.zeros(
        (0, 2), device=self._device, dtype=self._index_dtype)
    event_state_edges = torch.zeros(
        (0, 2), device=self._device, dtype=self._index_dtype)
    timing_edges = torch.zeros(
        (0, 2), device=self._device, dtype=self._index_dtype)

    new_event_nodes_count = 0

    event_nodes = []

    for attr_id, label in tqdm(enumerate(self._attributes), desc='attr_step', total=len(self._attributes)):
      values = df[label].to_numpy()
      unique_values, inverse = np.unique(values, return_inverse=True)
      
      added_event_nodes_count = self.addEventNodes(attr_id, unique_values)
      new_event_nodes_count += added_event_nodes_count
    
      # collects the created event nodes
      event_node_indices = self.eventNodeIndicesByAttributeValues(attr_id, unique_values)[inverse]
      assert len(values) == len(event_node_indices)

      event_nodes.append(event_node_indices)

    repeated_key_nodes = torch.repeat_interleave(key_nodes, len(self._attributes))
    repeated_state_holder_nodes = torch.repeat_interleave(state_holder_nodes, len(self._attributes))

    event_nodes = torch.stack(event_nodes).T.flatten()
    
    key_event_edges = torch.stack([repeated_key_nodes, event_nodes], dim=1)
    event_state_edges = torch.stack(
        [event_nodes, repeated_state_holder_nodes], dim=1)

    if self._delta > 0:
      source_nodes = []
      target_nodes = []

      all_state_holder_nodes = self.stateHolderNodeIndices()
      all_state_holder_node_timestamps = self._timestamps[all_state_holder_nodes]
      all_state_holder_node_sub_stream_identifiers = self._sub_stream_ids[all_state_holder_nodes]

      # connect within df
      for state_holder_node in tqdm(state_holder_nodes, desc='timing_relation_step'):
        timestamp = self._timestamps[state_holder_node]
        sub_stream_identifier = self._sub_stream_ids[state_holder_node]

        criteria = (all_state_holder_nodes < state_holder_node) & (all_state_holder_node_timestamps <= timestamp) & (
            all_state_holder_node_timestamps >= timestamp - self._delta)
        if self._sub_stream_key is not None:
          indices = ((all_state_holder_node_sub_stream_identifiers == sub_stream_identifier) & criteria).nonzero(as_tuple=True)[0]
        else:
          indices = criteria.nonzero(as_tuple=True)[0]
        preceding_nodes = all_state_holder_nodes[indices]
        preceding_nodes = preceding_nodes[preceding_nodes != state_holder_node]  # prevent self referencing
        if len(preceding_nodes) == 0:
          continue
        source_nodes += preceding_nodes
        target_nodes += [state_holder_node]*len(preceding_nodes)

      source_nodes = torch.tensor(source_nodes, device=self._device, dtype=self._index_dtype)
      target_nodes = torch.tensor(target_nodes, device=self._device, dtype=self._index_dtype)
      timing_edges = torch.stack([source_nodes, target_nodes], dim=1)

    # add all relations
    all_edges = torch.cat([key_event_edges, event_state_edges, timing_edges], dim=0)
    self._add_edges(all_edges)

    self.create_types(
      key_event_edge_count=len(key_event_edges),
      event_state_edge_count=len(event_state_edges),
      timing_edge_count=len(timing_edges))

    self.update_relation_maps()

  def create_types(self, key_event_edge_count = 0, event_state_edge_count = 0, timing_edge_count = 0):
    relation_count = key_event_edge_count + event_state_edge_count + timing_edge_count

    relation_types = torch.zeros(relation_count, device=self._device, dtype=self._relation_types.dtype)
    relation_types[0:key_event_edge_count] = GraphTypes.KEY_EVENT_RELATION_TYPE
    relation_types[key_event_edge_count:(key_event_edge_count+event_state_edge_count)] = GraphTypes.EVENT_STATE_RELATION_TYPE
    relation_types[(key_event_edge_count+event_state_edge_count):] = GraphTypes.TIME_RELATION_TYPE

    self._relation_types = torch.cat([self._relation_types, relation_types])

