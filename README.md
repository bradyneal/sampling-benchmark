# sampling-benchmark
Benchmark for samplers that sample from posterior distributions over model parameters

## File Structure
```
.
├── exploring-packages
│   └── ...
├── models.py
├── preprocessing
│   ├── separation.py
│   └── transform.py
└── sample_posteriors.py
        
```

### sample_posteriors.py
* Main file for building "grid"
* Draws samples from the posteriors of all of the supported models, conditioned on all of the specified datasts (double for loop)

### models.py
* Provides functions for sampling from the posteriors of various
different models
* MODEL_NAMES constant: specifies supported models
* **TODO:** implement more models

### preprocessing
* Package for data processing modules
* **separation.py** - provides functions for separating data based on the types of variables
* **transform.py** - provides functions for transformations of data (e.g. one-hot encoding, standardization, whitening)

### TODO: data
* Package for downloading, saving, and loading of data

### exploring-packages
* Jupyter Notebook code for exploring packages such as OpenML and PyMC3
