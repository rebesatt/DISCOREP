import random
import torch
import argparse
import numpy as np
import pandas as pd
from immmo.KnowledgeGraph import KnowledgeGraph
from pathlib import Path

def generate_knowledge_graph(args):
  dataset_name = Path(args.csv_file_path).stem

  if args.fix_na:
    df = pd.read_csv(args.csv_file_path, keep_default_na=False)
  else:
    df = pd.read_csv(args.csv_file_path,)

  if args.mode == 'custom':
    attributes = [x for x in df.columns if x != 'Truth']
  else:
    if args.all_attributes:
      attributes = df.columns
    else:
      attributes = args.attributes

  graph = KnowledgeGraph(time_key=args.time_key, critical_key=args.critical_key, sub_stream_key=args.sub_stream_key,
                    attributes=attributes, delta=args.delta, device=args.device)
  graph.addDataFrameToGraph(df)

  output_path = "{output_dir}/{dataset_name}".format(
      output_dir=args.output_dir, dataset_name=dataset_name)
  graph.dump_to_file(output_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # if 'custom': train custom dataset with name: --psm_model
  # if anything else use csv given in --csv_file_path
  parser.add_argument('--mode', default='custom')
  parser.add_argument('--psm_model', default='PSM_10')
  parser.add_argument('--csv_file_path', default='')

  # settings
  # creates timing edge between tuples t1, t2 if t2.time - t1.time <= delta
  parser.add_argument('--delta', type=int, default=1)
  parser.add_argument('--device', default='cpu')

  # output path
  parser.add_argument('--output_dir', default='./data/graphs')

  # if not custom:
  parser.add_argument('--time_key', default='Time')
  parser.add_argument('--critical_key', default='Critical')
  parser.add_argument('--sub_stream_key', default='Substream_ID')
  parser.add_argument('--attributes', nargs='+', default=[])
  parser.add_argument('--all_attributes', type=bool, default=False)
  parser.add_argument('--fix_na', type=bool, default=False)

  args = parser.parse_args()

  if args.mode == 'custom':
    args.csv_file_path = "./examples/data/{psm_model}.csv".format(psm_model=args.psm_model)

  generate_knowledge_graph(args)
