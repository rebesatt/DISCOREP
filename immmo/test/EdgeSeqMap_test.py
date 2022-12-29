from igraph import Graph
from .. import KnowledgeGraph
import numpy as np

def kreuzArrays(arr_a, arr_b):
  return np.array(np.meshgrid(arr_a, arr_b)).T.reshape(-1,2)

g = Graph(directed=True)
g.add_vertices(10)
n = 10

kreuz = kreuzArrays(np.arange(n), np.arange(n))
edges = kreuz[np.random.choice(len(kreuz), 10, replace=False)] # picks 50 unique random edges
g.add_edges(edges)

def test_sourceToTargetsMap():
  global g
  sourceToTarget_edges = []
  sourceToTarget = g.es.sourceToTargetsMap()
  for source, targets in sourceToTarget.items():
    for target in targets:
      sourceToTarget_edges.append([source, target])

  assert np.all(np.isin(sourceToTarget_edges, edges) == True) and np.all(np.isin(edges, sourceToTarget_edges) == True)

def test_targetToSourcesMap():
  targetToSource_edges = []
  targetToSource = g.es.targetToSourcesMap()
  for target, sources in targetToSource.items():
    for source in sources:
        targetToSource_edges.append([source, target])

  assert np.all(np.isin(targetToSource_edges, edges) == True) and np.all(np.isin(edges, targetToSource_edges) == True)