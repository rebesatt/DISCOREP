from immmo.helpers.query import Query
import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

from immmo.KnowledgeGraph import KnowledgeGraph
from immmo.model.RepresentationLearning import RepresentationLearning
from immmo.model.PSM import PSM
from examples.PSM_Schema import PSM_Schema
from immmo.utils import set_seed

psm_model = 'PSM_05' # adjust custom model
#model_path = './results/PSM_05/model_8000000.bin'  # adjust path to model
model_path = 'results/finance_query3_close2/model_4750000.bin'
k = 3 # num of clusters (does not count critical state)

set_seed(0)

# graph_path = './data/graphs/{psm_model}'.format(psm_model=psm_model)
graph_path = 'data/graphs/finance_query3_close2'
g = KnowledgeGraph.load_from_file(graph_path, 'cuda:0')

model = RepresentationLearning(
    g, embedding_dim=20)
model.load_model(model_path)


psm = PSM(model)

non_critical_state_vectors = psm.get_state_vectors()

clusters = psm.cluster(non_critical_state_vectors, k=k)
cluster_ids_x, cluster_centers = clusters

query_accuracy, configurations, configurations_scores, probability_matrix, paths = Query.generate_queries(
    model, k)

for i, (acc, path) in enumerate(zip(query_accuracy, paths)):
  print("Query Candidate", i)
  print("Precision: {0} Recall: {1} F1: {2}".format(*acc))
  print(Query.readable_query(psm, configurations, path))
  print('-----------------')
