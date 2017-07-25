# sampling-benchmark
Benchmark for samplers that sample from posterior distributions over model parameters

## File Structure
```
.
├── data
│   ├── download_datasets.py
│   └──preprocessing
│       ├── format.py
│       ├── separation.py
│       └── transform.py
├── exploring-packages
│   ├── ...
├── models
│   ├── classification.py
│   ├── regression.py
│   ├── unsupervised.py
│   └── utils.py
└── sample_posteriors.py        
```

### [sample_posteriors.py](https://github.com/bradyneal/sampling-benchmark/blob/master/sample_posteriors.py)
* Main file for building "grid"
* Draws samples from the posteriors of all of the supported models, conditioned on all of the specified datasts (double for loop)

### [models](https://github.com/bradyneal/sampling-benchmark/tree/master/models)
* Package for sampling from the posteriors of various models for tasks such as regression, classification, and unsupervised
* *MODEL_NAMES constants: specify supported models for each task
* **TODO:** implement more models

### [data](https://github.com/bradyneal/sampling-benchmark/tree/master/data)
* Package for downloading, preprocessing, saving, and loading of data
* [**download_datasets.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/download_datasets.py) - download and save the raw datasets
* **TODO: preprocess_datasets.py** - preprocess raw datasets and save according to the [**Datasets Format**](https://github.com/bradyneal/sampling-benchmark#datasets-format) below

### [preprocessing](https://github.com/bradyneal/sampling-benchmark/tree/master/data/preprocessing)
* Package for data processing modules
* [**format.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/preprocessing/format.py) - provides function for shifting between data formats (e.g. numpy arrays and Pandas DataFrames)
* [**separation.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/preprocessing/separation.py) - provides functions for separating data based on the types of variables
* [**transform.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/preprocessing/transform.py) - provides functions for transformations of data (e.g. one-hot encoding, standardization, robust standardization, whitening)

### Datasets Format
```
... (e.g. /data/lisa/data/openml)
└── datasets
    ├── raw
    │   └── {dataset_id}.pickle (same format for all folders below)
    ├── one-hot - one-hot encoded categorical features (same for all preprocessing below)
    ├── standardized - standardized non-categorical features
    ├── robust_standardized - robust standardized non-categorical features
    └── whitened - whitened non-categorical features
```

### [exploring-packages](https://github.com/bradyneal/sampling-benchmark/tree/master/exploring-packages)
* Jupyter Notebook code for exploring packages such as OpenML and PyMC3
