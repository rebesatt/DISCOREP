import torch
import numpy as np
from tqdm import tqdm
from pandas import DataFrame

from ...KnowledgeGraph import KnowledgeGraph
from ...constants.GraphTypes import GraphTypes

# test graph correctness


def testGraphCorrectness(graph: KnowledgeGraph, df: DataFrame):
  # number of key nodes == number of data points in Dataframe
  key_nodes = graph.keyNodeIndices()
  state_holder_nodes = graph.stateHolderNodeIndices()
  event_nodes = graph.eventNodeIndices()
  
  assert len(key_nodes) == len(df)
  assert len(state_holder_nodes) == len(df)

  labels = list(df.columns)
  labels.remove(graph._time_key)
  labels.remove(graph._critical_key)
  if graph._sub_stream_key in labels:
    labels.remove(graph._sub_stream_key)

  total_expected_number = 0

  for i, label in enumerate(labels):
    label_event_nodes = graph.eventNodeIndicesByAttribute(i)
    # there is exactly one event node for each unique attribute value
    assert len(df[label].unique()) == len(
        label_event_nodes), (i, len(df[label].unique()), len(label_event_nodes))
    total_expected_number += len(df[label].unique())

  # for each unique attribute value exists exactly one event node
  assert total_expected_number == len(event_nodes)

  # test if each data point is within the graph
  for n, col in tqdm(enumerate(df.iloc),desc='timing_relation_test_step', total=len(state_holder_nodes)):
    col_key_nodes = graph.keyNodeIndicesByTimestamps(torch.tensor([col[graph._time_key]]))
    assert torch.all(torch.tensor(col[graph._time_key]) == graph._timestamps[col_key_nodes])
    assert len(col_key_nodes) == 1
    for i, label in enumerate(labels):
      value_event_nodes = graph.eventNodeIndicesByAttributeValues(attr_id=i, values=[col[label]])
      assert len(value_event_nodes) == 1
      edge_tuples = torch.stack(
          [col_key_nodes[0], value_event_nodes[0]], dim=0)[None, :]
      key_event_relations = graph.get_eids(edge_tuples)
      assert len(key_event_relations) == 1 and key_event_relations[0] != -1
      assert graph._relation_types[key_event_relations[0]] == GraphTypes.KEY_EVENT_RELATION_TYPE

      edge_tuples = torch.stack(
          [value_event_nodes[0], state_holder_nodes[n]], dim=0)[None, :]
      event_state_relations = graph.get_eids(edge_tuples)
      assert len(event_state_relations) == 1 and event_state_relations[0] != -1
      assert graph._relation_types[event_state_relations[0]] == GraphTypes.EVENT_STATE_RELATION_TYPE
    
    if graph._sub_stream_key is not None:
      sub_stream_identifier = col[graph._sub_stream_key]
      indices = np.where((df[graph._sub_stream_key] == sub_stream_identifier) & (df[graph._time_key] >= col[graph._time_key]) & (df[graph._time_key] - col[graph._time_key] <= graph._delta))[0]
    else:
      indices = np.where((df[graph._time_key] >= col[graph._time_key]) & (df[graph._time_key] - col[graph._time_key] <= graph._delta))[0]
    indices = indices[indices != n]
    edge_tuples = torch.stack(
        [value_event_nodes[0], state_holder_nodes[n]], dim=0)[None, :]

    a = state_holder_nodes[n].repeat(len(state_holder_nodes[indices]))
    edge_tuples = torch.stack([a, state_holder_nodes[indices]], dim=1)
    if len(indices) > 0:
      timing_relations = graph.get_eids(edge_tuples)
      assert len(timing_relations) == len(indices)
    assert -1 not in timing_relations