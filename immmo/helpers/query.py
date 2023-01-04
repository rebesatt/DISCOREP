from immmo.helpers.predicate import Predicate
import torch
from immmo.utils import get_f1_of_matches
from heapq import heappush, heappop
import string
from immmo.model.PSM import PSM

ALPHABET = list(string.ascii_lowercase)

class Query:
  @staticmethod
  def match_query(psm, configurations):
    time_window = len(configurations)
    matches_per_configuration = []
    for configuration in configurations:
      match_map = Predicate.match_map_of_configuration(psm, configuration)
      matches_per_configuration.append(match_map)
    
    matched_tuples = []
    match_stream = torch.stack(matches_per_configuration).T
    last_occurences = torch.zeros(len(configurations))
    for i, matches in enumerate(match_stream):
      last_occurences[matches] = i

      if (last_occurences > 0).all() and (i - last_occurences.min()) < time_window:
        matched_tuples.append(i)

    return torch.tensor(matched_tuples, dtype=psm.model.graph._index_dtype, device=psm.model.device)

  @staticmethod
  def generate_query_candidates(psm, probability_matrix, configurations):
    Q = []

    critical_state = probability_matrix.shape[0]-1
    heappush(Q, (0, (critical_state, [])))

    query_accuracy = []
    paths = []

    visited = set()

    while len(Q) > 0:
      (cost, (state, path_to_critical)) = heappop(Q)
      visited.add(state)

      if len(path_to_critical) > 0:
        candidate_configurations = [configurations[state]
                                    for state in path_to_critical]
        query_matches = Query.match_query(psm, candidate_configurations)
        prec, recall, f1 = get_f1_of_matches(
            psm.model.node_to_scope[psm.model.critical_state_node_indices], (query_matches + 1))
        query_accuracy.append((prec, recall, f1))
        paths.append(path_to_critical)

      # calculate query accuracy
      for source in range(probability_matrix.shape[0]):
        if source in visited:
          continue
        cost_from_source_to_target = -probability_matrix[source, state].log()

        if cost_from_source_to_target.isinf():
          continue

        new_cost = cost + cost_from_source_to_target
        new_path = path_to_critical
        if state != critical_state:
          new_path = [state] + new_path
        heappush(Q, (new_cost.item(), (source, new_path)))

    return query_accuracy, paths

  @staticmethod
  def readable_query(psm, configurations, query_path):
    symbols = [ALPHABET[i] for i in range(len(query_path))]
    query_from = ', '.join(symbols)

    query_where = ' and '.join([Predicate.readable_configuration(
        psm, configurations[state], domain_name=symbols[i]) for i, state in enumerate(query_path)])

    from_string = "FROM({query_from})".format(query_from=query_from)
    where_string = "WHERE " + query_where
    within_string = "WITHIN {time_delta}".format(time_delta=len(query_path))

    query_string = "{from_string}\n{where_string}\n{within_string}".format(from_string=from_string, where_string=where_string, within_string=within_string)
    return query_string

  @staticmethod
  def generate_queries(model, k):
    psm = PSM(model)

    non_critical_state_vectors = psm.get_state_vectors()

    clusters = psm.cluster(non_critical_state_vectors, k=k)

    configurations, configurations_scores = Predicate.get_best_predicate_configurations(
        psm, clusters)

    probability_matrix = psm.generate_probability_matrix(
        psm.non_critical, clusters)

    query_accuracy, paths = Query.generate_query_candidates(
        psm, probability_matrix, configurations)
    return query_accuracy, configurations, configurations_scores, probability_matrix, paths
