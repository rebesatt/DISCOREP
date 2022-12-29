import torch
import random
from tqdm import tqdm
import argparse
import re
import numpy as np
from pathlib import Path
from immmo.utils import set_seed
from immmo.model.RepresentationLearning import RepresentationLearning
from immmo.KnowledgeGraph import KnowledgeGraph

def models_in_dir(dir):
  pathlist = np.array([str(x) for x in Path(
      dir).glob('**/*.bin')])

  # sort by epoch

  pathlist = pathlist[torch.tensor([int(re.search('_([0-9]*)\.bin', x).group(1))
              for x in pathlist]).argsort()]

  if type(pathlist) is np.str_:
      pathlist = np.array([pathlist])
  return pathlist


def test(args):
  set_seed(args.seed)

  output_dir = args.model_dir
  model_paths = models_in_dir(output_dir)

  graph = KnowledgeGraph.load_from_file(args.graph_path, device=args.device)

  model = RepresentationLearning(
      graph, embedding_dim=args.embedding_dim)
  model.set_output_dir(output_dir)

  for model_path in tqdm(model_paths):
    epoch = int(re.search('_([0-9]*)\.bin', model_path).group(1))
    model.load_model(model_path)

    # test set
    timing_count = model.positive_timing_triplets.shape[0]
    key_event_count = model.positive_key_event_triplets.shape[0]

    TIMING_TEST_SET = model.positive_timing_triplets[random.sample(
        range(timing_count), int(timing_count*args.test_set_percent))]
    KEY_EVENT_TEST_SET = model.positive_key_event_triplets[random.sample(
        range(key_event_count), int(key_event_count*args.test_set_percent))]

    model.test(epoch, TIMING_TEST_SET, KEY_EVENT_TEST_SET, event_k=args.hits_key_event, state_k=args.hits_followed_by)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # settings
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cpu')
  # percentage of training set as test set
  parser.add_argument('--test_set_percent', type=float, default=0.2)

  # hits at k
  parser.add_argument('--hits_key_event', type=int, default=10)
  parser.add_argument('--hits_followed_by', type=int, default=10)

  # knowledge graph location
  parser.add_argument('--mode', default='custom')  # train custom datasets | train graph given in --graph_path
  parser.add_argument('--psm_model', default='PSM_05')
  parser.add_argument('--graph_path', default='')

  # knowledge graph location
  # train custom datasets | train graph given in --graph_path
  parser.add_argument('--model_dir', default='./results/cloud/')

  # hyper params
  parser.add_argument('--embedding_dim', type=int, default=20)

  args = parser.parse_args()

  if args.mode == 'custom':
    args.graph_path = "./data/graphs/{psm_model}".format(psm_model=args.psm_model)

  test(args)
