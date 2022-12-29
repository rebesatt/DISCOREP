import torch
import random
from pathlib import Path
import re
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
from pathlib import Path
from immmo.utils import set_seed
from immmo.model.RepresentationLearning import RepresentationLearning
from immmo.KnowledgeGraph import KnowledgeGraph
from immmo.model.PSM import PSM
from examples.PSM_Schema import PSM_Schema

RELEVANT_ATTRIBUTES = {
    'PSM_01': {
        'A': [0],
        'B': [0],
        'C': [0],
        'D': [0],
        'E': [0]
    },
    'PSM_02': {
        'A': [0],
        'B': [0],
        'C': [0]
    },
    'PSM_05': {
        'A': [0],
        'B': [1],
        'C': [2],
        'D': [3]
    },
    'PSM_05': {
        'A': [0],
        'B': [1],
        'C': [2],
        'D': [3]
    },
    'PSM_06': {
        'A': [1],
        'B': [1,2],
        'C': [2],
        'D': [1]
    },
    'PSM_07': {
        'A': [1],
        'B': [2],
        'C': [2],
        'D': [1]
    },
    'PSM_08': {
        'A': [1],
        'B': [2],
        'C': [2],
        'D': [1]
    },
    'PSM_10': {
        'A': [0],
        'B': [1],
        'C': [],
        'D': [3]
    }
}

def models_in_dir(dir):
    pathlist = np.array([str(x) for x in Path(
        dir).glob('**/*.bin')])

    # sort by epoch

    pathlist = pathlist[torch.tensor([int(re.search('_([0-9]*)\.bin', x).group(1))
                for x in pathlist]).argsort()]

    if type(pathlist) is np.str_:
        pathlist = np.array([pathlist])
    return pathlist


def validate_custom(args):
  set_seed(args.seed)
  torch.set_printoptions(sci_mode=False, precision=4)

  output_dir = args.model_dir
  model_paths = models_in_dir(output_dir)

  loss_report_path = output_dir + '/validate_result.csv'
  report_file = open(loss_report_path, 'w')
  report_file.write('\t'.join(['epoch', 'matrix_dist', 'cluster_dist', 'min_relevance', 'max_relevance']) + '\n')
  report_file.flush()

  df = pd.read_csv('./examples/data/{psm_model}.csv'.format(psm_model=args.psm_model))
  ground_truth_stream = df['Truth']

  psm_schema = PSM_Schema('./examples/schemas/{psm_model}.json'.format(psm_model=args.psm_model))
  psm_schema.load()
  expected_matrix = psm_schema.get_probability_matrix()

  true_attr_index = RELEVANT_ATTRIBUTES[args.psm_model]

  graph = KnowledgeGraph.load_from_file(args.graph_path, device=args.device)

  for model_path in tqdm(model_paths):
    epoch = int(re.search('_([0-9]*)\.bin', model_path).group(1))
    model = RepresentationLearning(
        graph, embedding_dim=args.embedding_dim)
    model.model_output_path = output_dir
    model.load_model(model_path)

    attribute_influence_per_state = model.attribute_influence_per_state(
        model.state_holder_nodes)

    psm = PSM(model)
    set_seed(0)

    non_critical_state_vectors = psm.get_state_vectors()
    state_vectors = psm.get_state_vectors(non_critical=False)
    clusters = psm.cluster(non_critical_state_vectors, k=expected_matrix.shape[0]-1)
    probability_matrix = psm.generate_probability_matrix(psm.non_critical, clusters)
    
    cluster_dist = psm.get_state_cluster_distribution(ground_truth_stream, clusters, psm_schema)

    k = cluster_dist.shape[0]
    cluster_values = torch.arange(0,k).expand(k, k)
    expected_values = (cluster_values*cluster_dist).sum(dim=1, keepdim=True)
    variance_per_cluster = ((cluster_values - expected_values).pow(2) * cluster_dist).sum(1)

    cluster_states = [psm_schema.states[cluster_dist[i].argmax()]
                      for i in range(cluster_dist.shape[0])]

    perm = [psm_schema.states.index(state) for state in cluster_states]

    sorted_probability_matrix = torch.zeros_like(probability_matrix)

    for i in range(probability_matrix.shape[0]):
      for j in range(probability_matrix.shape[1]):
        sorted_probability_matrix[perm[i], perm[j]] = probability_matrix[i, j]
    
    acc_high_att = 0
    acc_low_att = 0

    non_critical_stream = ground_truth_stream[model.node_to_scope[psm.non_critical].tolist()]
    for index, truth in enumerate(non_critical_stream):
      if truth not in true_attr_index:
        continue
      state = model.node_to_scope[psm.non_critical[index]]
      ground_truth_att = set(true_attr_index[truth])
      _, pred_high_att = attribute_influence_per_state[state].topk(k=len(ground_truth_att), largest=True)
      _, pred_low_att = attribute_influence_per_state[state].topk(k=len(ground_truth_att), largest=False)

      acc_high_att += len(ground_truth_att & set(pred_high_att.tolist()))/len(ground_truth_att)
      acc_low_att += len(ground_truth_att & set(pred_low_att.tolist()))/len(ground_truth_att)
    
    acc_high_att /= len(non_critical_stream)
    acc_low_att /= len(non_critical_stream)

    matrix_metric = (sorted_probability_matrix -
                    expected_matrix).view(-1).norm(p=2)
    
    output_row = [epoch, matrix_metric.item(), variance_per_cluster.sum().item(), acc_high_att, acc_low_att]
    report_file.write('\t'.join([str(x) for x in output_row]) + '\n')
    report_file.flush()
  report_file.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # settings
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cpu')

  # knowledge graph location
  # train custom datasets | train graph given in --graph_path
  parser.add_argument('--psm_model', default='PSM_05')

  # knowledge graph location
  # train custom datasets | train graph given in --graph_path
  parser.add_argument(
      '--model_dir', default='./results/cloud')

  # hyper params
  parser.add_argument('--embedding_dim', type=int, default=20)

  args = parser.parse_args()
  args.graph_path = "./data/graphs/{psm_model}".format(psm_model=args.psm_model)

  validate_custom(args)
