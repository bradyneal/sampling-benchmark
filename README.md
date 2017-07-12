# sampler-benchmark
Benchmark for samplers that sample from posterior distributions over model parameters

## File Structure
```
.
├── exploring-packages
│   └── ...
├── models.py
├── processing
│   └── separation.py
└── sample_posteriors.py
        
```

### sample_posteriors.py
* Main file for building "grid"
* Draws samples from the posteriors of all of the supported models, conditioned on all of the specified datasts (double for loop)

### model.py
* Provides functions for sampling from the posteriors of various
different models
* MODEL_NAMES constant: specifies supported models
* **TODO:** implement more models

### processing
* Package for data processing modules
* **separation.py** - provides functions for separating data based on the types of variables
* **TODO: preprocessing.py** - provides functions for preprocessing of data (e.g. standardization, whitening, etc.)

### TODO: data
* Package for downloading, saving, and loading of data

### exploring-packages
* Jupyter Notebook code for exploring packages such as OpenML and PyMC3