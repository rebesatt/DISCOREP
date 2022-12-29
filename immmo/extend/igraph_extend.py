from igraph import EdgeSeq, Graph
import numpy as np

def sourceToTargetsMap(self):
  result = {}
  for edge in self:
    result.setdefault(edge.source, []).append(edge.target)
  return result
def targetToSourcesMap(self):
  result = {}
  for edge in self:
    result.setdefault(edge.target, []).append(edge.source)
  return result

def toNodeIndices(self):
  return np.array([[edge.source, edge.target] for edge in self])

def sourceToEdgesMap(self):
  result = {}
  for edge in self:
    result.setdefault(edge.source, []).append(edge.index)
  return result
def targetToEdgesMap(self):
  result = {}
  for edge in self:
    result.setdefault(edge.target, []).append(edge.index)
  return result

def toIndexMap(self):
  return dict(zip(self.indices, range(len(self))))

def verticesOfAllPathsBetweenEdge(self, edge):
  return self.verticesOfAllPathsBetweenVertices(edge.source, edge.target)

def verticesOfAllPathsBetweenVertices(self, source, target):
  s = set(self.subcomponent(source, mode="out"))
  t = set(self.subcomponent(target, mode="in"))
  return s.intersection(t)

def targetsOfEdgeSeq(self, edges):
  return self.vs[[edge.target for edge in edges]]

def sourcesOfEdgeSeq(self, edges):
  return self.vs[[edge.source for edge in edges]]

EdgeSeq.sourceToTargetsMap = sourceToTargetsMap
EdgeSeq.targetToSourcesMap = targetToSourcesMap
EdgeSeq.sourceToEdgesMap = sourceToEdgesMap
EdgeSeq.targetToEdgesMap = targetToEdgesMap
EdgeSeq.toIndexMap = toIndexMap
EdgeSeq.toNodeIndices = toNodeIndices
Graph.verticesOfAllPathsBetweenEdge = verticesOfAllPathsBetweenEdge
Graph.verticesOfAllPathsBetweenVertices = verticesOfAllPathsBetweenVertices
Graph.targetsOfEdgeSeq = targetsOfEdgeSeq
Graph.sourcesOfEdgeSeq = sourcesOfEdgeSeq