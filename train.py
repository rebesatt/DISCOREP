import torch
import random
from tqdm import tqdm
import argparse
from pathlib import Path
from immmo.utils import set_seed
from immmo.model.RepresentationLearning import RepresentationLearning
from immmo.KnowledgeGraph import KnowledgeGraph


def train(args):
  set_seed(args.seed)

  dataset_name = Path(args.graph_path).stem
  output_path = "{output_dir}/{dataset_name}".format(
      output_dir=args.output_dir, dataset_name=dataset_name)

  graph = KnowledgeGraph.load_from_file(args.graph_path, device=args.device)

  model = RepresentationLearning(graph, margin=args.margin, lr=args.learning_rate, embedding_dim=args.embedding_dim, gamma=args.gamma)
  model.set_output_dir(output_path)

  # test set
  timing_count = model.positive_timing_triplets.shape[0]
  key_event_count = model.positive_key_event_triplets.shape[0]

  TIMING_TEST_SET = model.positive_timing_triplets[random.sample(range(timing_count), int(timing_count*args.test_set_percent))]
  KEY_EVENT_TEST_SET = model.positive_key_event_triplets[random.sample(range(key_event_count), int(key_event_count*args.test_set_percent))]

  # timing edge samples
  triplet_samples = torch.randint(0, model.positive_timing_triplets.shape[0], (args.epochs, 1), dtype=model.graph._index_dtype, device=model.device)

  for epoch in tqdm(range(args.epochs), desc='steps', total=args.epochs):
    model.forward(triplet_samples[epoch])

    if epoch % args.loss_report_step_size == 0:
      if epoch > 0:
        model.save_model(epoch=epoch)
      model.loss_report(epoch, TIMING_TEST_SET, KEY_EVENT_TEST_SET)

  model.save_model(epoch=args.epochs)
  model.loss_report(args.epochs, TIMING_TEST_SET, KEY_EVENT_TEST_SET)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # settings
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--test_set_percent', type=float, default=0.2) # percentage of training set as test set
  parser.add_argument('--loss_report_step_size', type=int, default=250000) # calculate total loss after as many epochs
  
  # knowledge graph location
  parser.add_argument('--mode', default='custom')  # train custom datasets | train graph given in --graph_path
  parser.add_argument('--psm_model', default='PSM_05')
  parser.add_argument('--graph_path', default='')

  # output location
  parser.add_argument('--output_dir', default='./results')

  # hyper params
  parser.add_argument('--learning_rate', type=float, default=0.01)
  parser.add_argument('--epochs', type=int, default=20000000)
  parser.add_argument('--embedding_dim', type=int, default=20)
  parser.add_argument('--margin', type=float, default=0.0)
  parser.add_argument('--gamma', type=float, default=1.0)
  
  args = parser.parse_args()

  if args.mode == 'custom':
    args.graph_path = "./data/graphs/{psm_model}".format(psm_model=args.psm_model)

  train(args)
