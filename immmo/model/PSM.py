import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from heapq import heappush, heappop
from sklearn.cluster import KMeans
from immmo.utils import set_seed

class PSM:
  def __init__(self, model) -> None:
    self.model = model
    critical = set(self.model.critical_state_node_indices.tolist())
    self.non_critical = torch.tensor(sorted(list(set(self.model.state_holder_nodes.tolist()) - critical)), dtype=self.model.graph._index_dtype, device=model.device)
  
  def edge_into(self, state):
    edges_into_state = self.model.positive_timing_triplets[(
        self.model.positive_timing_triplets[:, 2] == state)]
    if edges_into_state.shape[0] == 0:
      return None
    next_edge = torch.argmax(edges_into_state[:, 0], dim=0)
    return edges_into_state[next_edge, 1]

  def edge_out_of(self, state):
    edges_out_of = self.model.positive_timing_triplets[(
        self.model.positive_timing_triplets[:, 0] == state)]
    if edges_out_of.shape[0] == 0:
      return None, None
    next_edge = torch.argmin(edges_out_of[:, 2], dim=0)
    return edges_out_of[next_edge, 1], edges_out_of[next_edge, 2]

  def get_state_vectors(self, non_critical=True):
    with torch.no_grad():
      if non_critical:
        states = self.non_critical
      else:
        states = self.model.state_holder_nodes

      D = self.model.f_sigma_by_index(states)

    return D

  def cluster(self, state_vectors, k):
    kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++', n_init=10).fit(state_vectors.cpu().numpy())
    cluster_ids_x, cluster_centers = torch.tensor(kmeans.labels_, dtype=self.model.graph._index_dtype, device=self.model.device), torch.tensor(kmeans.cluster_centers_, device=self.model.device)

    return cluster_ids_x, cluster_centers

  def generate_probability_matrix(self, states_used_for_clustering, clusters, sample_count=10000):
    cluster_ids_x, cluster_centers = clusters
    k = cluster_centers.shape[0]
    all_cluster_centers = torch.cat(
        (cluster_centers, self.model.critical_embedding.view(1, -1)))

    M = torch.zeros((k+1, k+1), device=cluster_centers.device)

    # transition probability between clusters and into the critical state
    for source in range(k):
      cluster_ids = (cluster_ids_x == source).nonzero(as_tuple=True)[0]
      source_nodes = self.non_critical[cluster_ids]

      for i in range(sample_count):
        source_node = source_nodes[torch.randint(low=0, high=source_nodes.shape[0], size=(1,))]
        relation, target_node = self.edge_out_of(source_node)
        if relation is None:
          continue
        
        if self.model.entity_is_critical[target_node]:
          target = k
        else:
          target = cluster_ids_x[(states_used_for_clustering == target_node)]

        M[source, target] += 1
    
    # out of critical state
    source_nodes = self.model.critical_state_node_indices
    for i in range(sample_count):
      source_node = source_nodes[torch.randint(
          low=0, high=source_nodes.shape[0], size=(1,))]
      relation, target_node = self.edge_out_of(source_node)
      if relation is None:
        continue

      if self.model.entity_is_critical[target_node]:
        target = k  
      else:
        target = cluster_ids_x[(states_used_for_clustering== target_node)]
      M[k, target] += 1

    return M / M.norm(p=1, dim=1, keepdim=True)

  def match_states_to_clusters(self, state_vectors, clusters):
    _, cluster_centers = clusters

    result = (state_vectors.view(-1, 1, state_vectors.shape[1]) - cluster_centers.expand(
        state_vectors.shape[0], cluster_centers.shape[0], cluster_centers.shape[1])).norm(p=2, dim=2).argmin(dim=1)
    
    result[(state_vectors == self.model.critical_embedding).all(dim = 1)] = cluster_centers.shape[0]

    return result

  def align_trajectory(self, state_vector_history, clusters, probability_matrix, alpha=2):
    _, cluster_centers = clusters
    k = cluster_centers.shape[0] + 1

    Q = []

    cluster_ids_per_state_in_substream = self.match_states_to_clusters(state_vector_history, clusters)
    reversed_history = cluster_ids_per_state_in_substream.flip(dims=[0]).tolist()

    cost_max = 0

    for state in reversed_history:
      cost_max -= probability_matrix[:, state].max().log()

    for state in range(k):
      if state == reversed_history[0]:
        cost = probability_matrix[:, state].max().log()
        heappush(Q, ((cost_max + cost).item(), (state, reversed_history[1:])))
      else:
        heappush(Q, (cost_max.item(), (state, reversed_history)))

    while len(Q) > 0:
      (cost, (s, H)) = heappop(Q)
      if len(H) == 0:
        return cost

      for source in range(k):
        cost_from_source_to_target = -probability_matrix[source, s].log()
        if source == H[0]:
          new_cost = cost + \
              probability_matrix[:, source].max().log() + \
              cost_from_source_to_target
          heappush(Q, (new_cost.item(), (source, H[1:])))
        else:
          new_cost = cost + cost_from_source_to_target
          heappush(Q, (new_cost.item(), (source, H)))

      removal_cost = cost + \
          probability_matrix[:, H[0]].max().log() - alpha * \
          probability_matrix[:, H[0]].min().log()
      heappush(Q, (removal_cost.item(), (s, H[1:])))


  def get_state_cluster_distribution(self, value_stream, clusters, psm_schema):
    cluster_ids_x, cluster_centers = clusters

    non_critical_cluster_ids = self.match_states_to_clusters(self.get_state_vectors(), clusters)

    critical_symbol = psm_schema.critical_states[0]
    critical_state_index = psm_schema.states.index(critical_symbol)

    k = cluster_centers.shape[0]
    state_cluster_distribution = torch.zeros((k+1, k+1))
    state_cluster_distribution[k, critical_state_index] = 1.0
    
    for state in range(k):
      state_columns = self.model.node_to_scope[self.non_critical[(non_critical_cluster_ids == state)]].tolist()
      values_of_state = list(value_stream[state_columns].values)

      unique, counts = np.unique(values_of_state, return_counts=True)

      part_of_all = []
      for i, value in enumerate(unique):
        part_of_all.append(counts[i]/(value_stream.values ==
                          value).nonzero()[0].shape[0])

      inner_cluster_distribution = counts / counts.sum()
      total_distribution = np.array(part_of_all)

      associated_state_distribution = inner_cluster_distribution * total_distribution

      for i in range(len(unique)):
        associated_state = psm_schema.states.index(unique[i][0])
        state_cluster_distribution[state, associated_state] = associated_state_distribution[i]

    return state_cluster_distribution

  def select_optimal_state_number(self, window, threshold=0.5, initial_k=10, delta=2, pieces_sample_count=5, pieces_size=10, max_iter=100):
    non_critical_window = window[self.model.entity_is_critical[window].logical_not()]

    k = initial_k

    with torch.no_grad():
      state_vectors_of_window = self.model.f_sigma_by_index(window)
      non_critical_state_vectors_of_window = self.model.f_sigma_by_index(
          non_critical_window)
      shuffled_state_vectors_of_window = self.model.f_sigma_by_index_shuffled(
          window)

    for _ in range(max_iter):
      clusters_minus = self.cluster(
          non_critical_state_vectors_of_window, k=int(k - delta/2))
      clusters_plus = self.cluster(
          non_critical_state_vectors_of_window, k=int(k + delta/2))

      probability_matrix_minus = self.generate_probability_matrix(
          non_critical_window, clusters_minus)
      probability_matrix_plus = self.generate_probability_matrix(
          non_critical_window, clusters_plus)

      substream_ids, count_per_substream = self.model.graph._sub_stream_ids[window].unique(
          return_counts=True)
      valid_substreams_ids = substream_ids[count_per_substream >= pieces_size]

      assert valid_substreams_ids.shape[0] > 0, "pieces_size must be bigger than the smallest substeam in the window"

      f_minus = 0
      f_plus = 0

      for i in range(pieces_sample_count):
        substream_id = valid_substreams_ids[torch.randint(
            0, valid_substreams_ids.shape[0], (1,))[0]]

        tuples_of_window_in_substream = (
            self.model.graph._sub_stream_ids[window] == substream_id).nonzero(as_tuple=True)[0]
        begin = torch.randint(
            0, tuples_of_window_in_substream.shape[0] - pieces_size, (1, ))[0]
        pieces = tuples_of_window_in_substream[begin:begin+pieces_size]

        vectors_of_pieces = state_vectors_of_window[pieces]
        shuffled_vectors_of_pieces = shuffled_state_vectors_of_window[pieces]

        f_minus += self.align_trajectory(shuffled_vectors_of_pieces,
                                        clusters_minus, probability_matrix_minus) - self.align_trajectory(vectors_of_pieces,                                                                                        clusters_minus, probability_matrix_minus)

        f_plus += self.align_trajectory(shuffled_vectors_of_pieces,
                                      clusters_plus, probability_matrix_plus) - self.align_trajectory(vectors_of_pieces,
                                                                                                      clusters_plus, probability_matrix_plus)

      grad = -(f_plus-f_minus)/delta
      last_k = k
      k -= grad

      if abs(last_k - k) < threshold:
        return int(k)
    return k

  def select_optimal_state_number_fast(self, window, k_max=100, pieces_size=10, pieces_sample_count=5):
    non_critical_window = window[self.model.entity_is_critical[window].logical_not(
    )]

    with torch.no_grad():
      state_vectors_of_window = self.model.f_sigma_by_index(window)
      non_critical_state_vectors_of_window = self.model.f_sigma_by_index(
          non_critical_window)

    alignments = {}

    for k in range(1, k_max):
      clusters = self.cluster(non_critical_state_vectors_of_window, k=int(k))

      probability_matrix = self.generate_probability_matrix(
          non_critical_window, clusters)

      substream_ids, count_per_substream = self.model.graph._sub_stream_ids[window].unique(
          return_counts=True)
      valid_substreams_ids = substream_ids[count_per_substream >= pieces_size]

      assert valid_substreams_ids.shape[0] > 0, "pieces_size must be bigger than the smallest substeam in the window"

      f = 0

      for _ in range(pieces_sample_count):
        substream_id = valid_substreams_ids[torch.randint(
            0, valid_substreams_ids.shape[0], (1,))[0]]

        tuples_of_window_in_substream = (
            self.model.graph._sub_stream_ids[window] == substream_id).nonzero(as_tuple=True)[0]
        begin = torch.randint(
            0, tuples_of_window_in_substream.shape[0] - pieces_size, (1, ))[0]
        pieces = tuples_of_window_in_substream[begin:begin+pieces_size]

        vectors_of_pieces = state_vectors_of_window[pieces]

        f += self.align_trajectory(vectors_of_pieces, clusters, probability_matrix)

      alignments[k] = f/pieces_sample_count
      print(alignments[k])
    return alignments
