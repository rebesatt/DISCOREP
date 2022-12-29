#!/usr/bin/env python3

class GraphTypes():
  KEY_NODE_TYPE = 0
  EVENT_NODE_TYPE = 1
  STATE_HOLDER_NODE_TYPE = 2

  KEY_EVENT_RELATION_TYPE = 3 # relationship from key node to an event node of attribute a 
  EVENT_STATE_RELATION_TYPE = 4 # relationship from event node e to a state-holder node
  TIME_RELATION_TYPE = 5 # followed-by relationship between two state-holder nodes
