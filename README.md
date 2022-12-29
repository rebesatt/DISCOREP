# Requirements
- python@3.8.0
- poetry


# Setup
- in Repository run: ```poetry install```
- enter the environment with ```poetry shell```

# Custom Schema Generation
Generates custom datasets. The generated csv files are stored in examples/data
```cd examples & python3 PSM_Schema.py```

# Preprocess
Generates the the knowledge graph of a custom dataset or a given csv file containing the substream. Arguments are specified in preprocess.py
From custom dataset: ```python3 preprocess.py --psm_model <model-name>```
From CSV file: ```python3 preprocess.py --mode 1 --csv_file_path <csv-file>```

# Train
Trains the model and 
Arguments are specified in train.py
From custom knowledge graph: ```python3 train.py --psm_model <model-name>```
From knowledge graph:```python3 train.py --mode 1 --graph_path <graph-path>```

# Test
Calculates the Hits@k and Mrr score for all models in the specified directory
For custom models: ```python3 test.py --psm_model <model-name> --model_dir <model-dir>```
Specify kgraph of for model: ```python3 test.py --mode 1 --graph_path <graph-path> --model_dir <model-dir>```

# Validate Custom
Calculates metrics for know ground truth PSM (only for custom models)
For custom models: ```python3 validate_custom.py --psm_model <model-name> --model_dir <model-dir>```

# Generate Query
Generates promising queries
Arguments must be adjusted in file
```python3 gen_query.py```