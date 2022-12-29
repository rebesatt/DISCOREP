import json
import numpy as np
from tqdm import tqdm
import torch
import random
import re
from immmo.utils import set_seed

STATE = 's'
PROBABILITY = 'p'
VALUE = 'v'

NUMBER_RANGE_REGEX = '^\[([0-9]+(\.[0-9]+)?)\-([0-9]+(\.[0-9]+)?)\]$'
DISTINCT_VALUES_REGEX = '^\([^|]+(\|[^|]+)*\)$'

class PSM_Schema():
  def __init__(self, schema_path, output_path=None):
    self.schema_path = schema_path
    self.output_path = output_path
    self.schema = {}
    self.states = []
    self.critical_states = []

    self.generative_value_counter = 0

    self.V = {}
    self.T = {}
    self.attr_count = None

    self.output_handle = None
    self.output_counter = 0

  def load(self):
    with open(self.schema_path, 'r') as f:
      self.schema = json.load(f)
    self.states = list(self.schema.keys())
    self.V = {}
    self.T = {}
    self.attr_count = None

    for s in self.states:
      state_config = self.schema[s]
      self.T[s] = { STATE: [], PROBABILITY: [] }
      for x in state_config["T"]:
        self.T[s][STATE].append(x[0])
        self.T[s][PROBABILITY].append(x[1])

      self.V[s] = { VALUE: [], PROBABILITY: [] }
      for x in state_config["V"]:
        self.V[s][VALUE].append(x[0])
        self.V[s][PROBABILITY].append(x[1])
        if self.attr_count is None:
          self.attr_count = len(x[0].split(','))
      
      if 'critical' in state_config:
        self.critical_states.append(s)

  def output_file_handle(self):
    if self.output_handle is None:
      self.output_handle = open(self.output_path, 'w')
      self.output_handle.write(','.join(['Truth', 'Time', 'Substream_ID'] + [
                               'Value' + str(x) for x in range(self.attr_count)] + ['Critical']) + '\n')
      self.output_counter = 0
    return self.output_handle
  
  def output(self, ground_truth_state, substream_id, attributes):
    attribute_values, is_critical = attributes
    values = [self.generate_value(a) for a in attribute_values.split(',')]
    self.output_file_handle().write(','.join([ground_truth_state, str(
        self.output_counter), str(substream_id)] + values + [is_critical]) + '\n')
    self.output_file_handle().flush()
    self.output_counter += 1

  def generate_value(self, value_str):
    if value_str == '$':
      result = self.generative_value_counter
      self.generative_value_counter += 1
      return str(result)

    m = re.match(NUMBER_RANGE_REGEX, value_str)
    if m is not None:
      begin = m.group(1)
      end = m.group(3)

      if m.group(2) is None and m.group(4) is None:
        # integer range
        return str(random.randint(int(begin), int(end)))
      else:
        # float range
        return "{:.3f}".format(random.random() * (float(end)-float(begin)) + float(begin))

    m = re.match(DISTINCT_VALUES_REGEX, value_str)
    if m is not None:
      # select uniform from given items
      values = value_str[1:-1].split('|')
      select = random.randint(0, len(values)-1)
      return str(values[select])

    # does not match any format above, return pure value
    return value_str

  def generate(self, n, begin_state = None, sub_stream_num=1):
    if begin_state is not None:
      assert begin_state in self.states
      current_state = begin_state
    else:
      current_state = self.states[0]

    substream_id = 0
    substream_size = int(n / sub_stream_num)

    for i in tqdm(range(n), total=n):
      if i > 0 and i % substream_size == 0:
        substream_id += 1
      
      transition_probs = self.T[current_state][PROBABILITY]
      transition_states = self.T[current_state][STATE]

      value_probs = self.V[current_state][PROBABILITY]
      values = self.V[current_state][VALUE]

      output = np.random.choice(np.arange(
          0, len(value_probs)), p=value_probs)
      output_value = values[output]

      is_critical = str(current_state in self.critical_states)

      self.output(current_state, substream_id, [output_value, is_critical])

      next_state = np.random.choice(np.arange(
          0, len(transition_probs)), p = transition_probs)
      current_state = transition_states[next_state]
      
  def get_probability_matrix(self):
    matrix = []
    for state_settings in self.T.values():
      row = [0. for _ in self.states]
      for i in range(len(state_settings[STATE])):
        next_state = state_settings[STATE][i]
        trans_prob = state_settings[PROBABILITY][i]
        row[self.states.index(next_state)] = trans_prob
      matrix.append(row)
    return torch.tensor(matrix)

if __name__ == "__main__":
  set_seed(0)
  p = PSM_Schema('./schemas/PSM_01.json', './data/PSM_01.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_02.json', './data/PSM_02.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_02.json', './data/PSM_02_10000.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_03.json', './data/PSM_03.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_03_distinct.json', './data/PSM_03_distinct.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_03_multi_attr.json',
                './data/PSM_03_multi_attr.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_03_multi_attr_single_attr.json',
                 './data/PSM_03_multi_attr_single_attr.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_04_multi_attr.json',
                './data/PSM_04_multi_attr.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_05.json',
                 './data/PSM_05.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_05.json',
                 './data/PSM_05_substream.csv')
  p.load()
  p.generate(10000, "A", sub_stream_num=int(10000/16))

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_06.json',
                 './data/PSM_06.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_06.json',
                 './data/PSM_06_substream.csv')
  p.load()
  p.generate(10000, "A", sub_stream_num=int(10000/100))

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_07.json',
                 './data/PSM_07.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_08.json',
                 './data/PSM_08.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_09.json',
                 './data/PSM_09.csv')
  p.load()
  p.generate(10000, "A")

  set_seed(0)
  p = PSM_Schema('./schemas/PSM_10.json',
                 './data/PSM_10.csv')
  p.load()
  p.generate(10000, "A")
