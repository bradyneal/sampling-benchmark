# sampling-benchmark
Benchmark for samplers that sample from posterior distributions over model parameters

## File Structure
```
.
├── exploring-packages
│   └── ...
├── models.py
├── data
│   ├── download_datasets.py
│   └── preprocess_datasets.py
├── preprocessing
│   ├── separation.py
│   └── transform.py
└── sample_posteriors.py
        
```

### [sample_posteriors.py](https://github.com/bradyneal/sampling-benchmark/blob/master/sample_posteriors.py)
* Main file for building "grid"
* Draws samples from the posteriors of all of the supported models, conditioned on all of the specified datasts (double for loop)

### [models.py](https://github.com/bradyneal/sampling-benchmark/blob/master/models.py)
* Provides functions for sampling from the posteriors of various
different models
* MODEL_NAMES constant: specifies supported models
* **TODO:** implement more models

### [data](https://github.com/bradyneal/sampling-benchmark/tree/master/data)
* Package for downloading, preprocessing, saving, and loading of data
* [**download_datasets.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/download_datasets.py) - download and save the raw datasets
* **preprocess_datasets.py** - preprocess raw datasets and save according to the [**Datasets Format**](https://github.com/bradyneal/sampling-benchmark#datasets-format) below

### [preprocessing](https://github.com/bradyneal/sampling-benchmark/tree/master/preprocessing)
* Package for data processing modules
* [**separation.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/preprocessing/separation.py) - provides functions for separating data based on the types of variables
* [**transform.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/preprocessing/transform.py) - provides functions for transformations of data (e.g. one-hot encoding, standardization, robust standardization, whitening)

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
