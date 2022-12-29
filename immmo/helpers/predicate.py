import torch
from immmo.utils import get_unique_first_occurences, get_f1_of_matches
from itertools import chain, combinations

class Predicate:
  @staticmethod
  def match_map_of_configuration(psm, configuration):
    values_per_state = psm.model.graph._node_values[psm.model.event_nodes_per_state]
    current_match_map = None

    for attr_id, value_ids in configuration.items():
      attribute_values = values_per_state[:, attr_id]
      value_is_in_column = torch.isin(attribute_values, value_ids)
      if current_match_map is None:
        current_match_map = value_is_in_column
      else:
        current_match_map = current_match_map.logical_and(value_is_in_column)
    if current_match_map is None:
      current_match_map = torch.zeros(psm.model.state_holder_nodes.shape[0], dtype=bool, device=psm.model.device)
    return current_match_map

  @staticmethod
  def predicate_evaluation(psm, states, configuration):
    match_map = Predicate.match_map_of_configuration(psm, configuration)

    matched_states_true = match_map.nonzero(as_tuple=True)[0]
    matched_states_false = match_map.logical_not().nonzero(as_tuple=True)[
        0]
    
    return get_f1_of_matches(states, matched_states_true), get_f1_of_matches(states, matched_states_false)

  @staticmethod
  def get_importance(psm, candidates, ground_truth_states):
    values, value_idx = get_unique_first_occurences(candidates[0, :])
    attribute_ids = candidates[2, value_idx].int()

    scores = torch.zeros((values.shape[0], 2))

    for i in range(values.shape[0]):
        attr_id = attribute_ids[i]
        pred_val = values[i].view(1)
        (_, _, f1_true), (_, rec_false, _) = Predicate.predicate_evaluation(
            psm, ground_truth_states, { attr_id: pred_val })

        # how good does pred_val matches the cluster
        scores[i, 0] = f1_true
        # how much of the cluster is matched with NOT pred_val
        scores[i, 1] = rec_false

    return scores, value_idx

  @staticmethod
  def filter_predicates(psm, high_candidates, cluster_states):
    importance_per_candidate, candidate_idx = Predicate.get_importance(
        psm, high_candidates, cluster_states)

    f1_recall_ratio = importance_per_candidate[:, 0] / importance_per_candidate[:, 1] 

    threshold = 0.01
    remaining_candidates = candidate_idx[(f1_recall_ratio < threshold).logical_not()]
    remaining_predicates = high_candidates[:, remaining_candidates]

    for i, pred in enumerate(remaining_predicates[0]):
      remaining_predicates[1, i] = high_candidates[1, (high_candidates[0] == pred)].sum()


    #remaining_candidates
    return remaining_predicates

  @staticmethod
  def promising_values(model, clusters, states_used_for_clustering):
    cluster_ids_x, cluster_centers = clusters

    candidates_per_cluster = []

    for c in range(cluster_centers.shape[0]):
      cluster_states = states_used_for_clustering[(cluster_ids_x == c)]

      with torch.no_grad():
        influence = model.attribute_influence_per_state(cluster_states)
      normed_influence = influence/influence.sum(dim=1, keepdim=True)

      event_node_values = model.graph._node_values[model.event_nodes_per_state[model.node_to_scope[cluster_states]]]

      # top attribute count
      a_n = 2

      high_influence_per_state, high_influence_per_state_index = normed_influence.topk(
          k=a_n, dim=1, largest=False)  # higher values -> lower influence
      low_influence_per_state, low_influence_per_state_index = normed_influence.topk(
          k=a_n, dim=1, largest=True)  # smaller values -> higher influence

      highest_influence_values = event_node_values.gather(
          1, high_influence_per_state_index).view(-1)
      lowest_influence_values = event_node_values.gather(
          1, low_influence_per_state_index).view(-1)

      # value
      # influence value
      # attr_id

      high_candidates = torch.stack(
          (highest_influence_values, high_influence_per_state.view(-1), high_influence_per_state_index.view(-1)))
      low_candidates = torch.stack(
          (lowest_influence_values, low_influence_per_state.view(-1), low_influence_per_state_index.view(-1)))

      candidates_per_cluster.append([high_candidates, low_candidates])
    return candidates_per_cluster

  def get_best_predicate_configurations(psm, clusters):
    cluster_ids_x, cluster_centers = clusters

    cluster_configurations = []
    cluster_configuration_scores = []

    candidates_per_cluster = Predicate.promising_values(
        psm.model, clusters, psm.non_critical)
    for cluster_id, (high_candidates, low_candidates) in enumerate(candidates_per_cluster):

      cluster_states = psm.model.node_to_scope[psm.non_critical[(
          cluster_ids_x == cluster_id)]]
      high_candidates = candidates_per_cluster[cluster_id][0]

      remaining_predicates = Predicate.filter_predicates(
          psm, high_candidates, cluster_states)

      remaining_predicates[[0, 2], :].T.tolist()

      pred_ids = torch.arange(remaining_predicates.shape[1]).tolist()

      highest_f1 = 0
      best_configuration = None

      for x in chain.from_iterable(combinations(pred_ids, r) for r in range(len(pred_ids)+1)):
        values_per_attr = {}
        subset_predicates = remaining_predicates[:, x]
        attr_ids = subset_predicates[2, :].unique()

        values_per_attr = {attr_id.int().item(
        ): subset_predicates[0, (subset_predicates[2, :] == attr_id)] for attr_id in attr_ids}
        scores_of_configuration, _ = Predicate.predicate_evaluation(
            psm, cluster_states, values_per_attr)

        if scores_of_configuration[0] > highest_f1:
          best_configuration = values_per_attr
          highest_f1 = scores_of_configuration[0]

      cluster_configurations.append(best_configuration)
      cluster_configuration_scores.append(highest_f1)

    return cluster_configurations, cluster_configuration_scores

  @staticmethod
  def readable_configuration(psm, configuration, domain_name=''):
    attribute_strings = []

    for attr_id, value_ids in configuration.items():
      attribute_name = psm.model.graph._attributes[attr_id]

      attribute_values = [str(x) for x in psm.model.graph._value_bag[value_ids.int().tolist()]]

      if len(attribute_values) == 1:
        attribute_strings.append("{domain_name}.{attribute_name} = {attribute_value}".format(
            attribute_name=attribute_name, attribute_value=attribute_values[0], domain_name=domain_name))
      else:
        attribute_strings.append("{domain_name}.{attribute_name} in [{attribute_values}]".format(
            attribute_name=attribute_name, attribute_values=attribute_values, domain_name=domain_name))
    return " and ".join(attribute_strings)
