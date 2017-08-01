# sampling-benchmark
Benchmark for samplers that sample from posterior distributions over model parameters

## File Structure
```
.
├── data
│   ├── config.py
│   ├── io.py
│   ├── preprocessing
│   │   ├── format.py
│   │   ├── separation.py
│   │   └── transform.py
│   └── repo.py
├── data_scripts
│   ├── download_datasets.py
│   ├── preprocess_datasets.py
│   ├── purge_bad_datasets.py
│   └── read_datasets.py
├── exploring-packages
│   └── ...
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

### [data_scripts](https://github.com/bradyneal/sampling-benchmark/tree/master/data_scripts)
* Folder for data downloading, preprocessing, validating, etc. scripts
* [**download_datasets.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data_scripts/download_datasets.py) - downloads and saves the raw datasets
* [**purge\_bad\_datasets.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data_scripts/purge_bad_datasets.py) - deletes any datasets that were pickled but don't work (e.g. empty dataset files)
* [**preprocess_datasets.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data_scripts/preprocess_datasets.py) - preprocesses raw datasets and saves according to the [**Datasets Format**](https://github.com/bradyneal/sampling-benchmark#datasets-format) below
* [**read_datasets.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data_scripts/read_datasets.py) - reads all datasets, logging any errors


### [data](https://github.com/bradyneal/sampling-benchmark/tree/master/data)
* Package for downloading, preprocessing, saving, loading, and deleting of data
* [**config.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/config.py) - contains configuration information such as the various folders that the different preprocessed versions of data are stored in
* [**io.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/io.py) - provides functions for reading, writing, and deleting of datasets and any necessary error logging
* [**repo.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/repo.py) - provides functions for interacting with OpenML metadata (e.g. get all datasets by task)

### [preprocessing](https://github.com/bradyneal/sampling-benchmark/tree/master/data/preprocessing)
* Sub-package for data processing modules
* [**format.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/preprocessing/format.py) - provides function for shifting between data formats (e.g. numpy arrays and Pandas DataFrames)
* [**separation.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/preprocessing/separation.py) - provides functions for separating data based on the types of variables
* [**transform.py**](https://github.com/bradyneal/sampling-benchmark/blob/master/data/preprocessing/transform.py) - provides functions for transformations of data (e.g. one-hot encoding, standardization, robust standardization, whitening)

### Datasets Format
```
... (e.g. /data/lisa/data/openml)
└── datasets
    ├── errors - logging of any errors encountered with datasets
    ├── raw
    │   └── {dataset_id}.pickle (same format for all folders below)
    ├── one-hot - one-hot encoded categorical features (same for all preprocessing below)
    ├── standardized - standardized non-categorical features
    ├── robust_standardized - robust standardized non-categorical features
    └── whitened - whitened non-categorical features
```

### [exploring-packages](https://github.com/bradyneal/sampling-benchmark/tree/master/exploring-packages)
* Jupyter Notebook code for exploring packages such as OpenML and PyMC3
