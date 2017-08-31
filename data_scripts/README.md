# data_scripts
Folder for data downloading, preprocessing, and validating scripts

All of these scripts are "main" files. This is simply a folder for scripts, not a package.

* download_datasets.py - This file was used to download the datasets from OpenML. It is the most out of date as code was abstracted into the data package after this file was written and run (it would make use of many of the functions in the data package). Many of the datasets failed to download, however, due to issues with OpenML users uploading datasets in the wrong format and other OpenML server errors. We were able to download around 2200 datasets, but there are many more, which could potentially be downloaded if we contact the OpenML owner to see what's going on.
* preprocess_datasets.py - This file preprocesses all the datasets on disk. This was the next file that was written, but some datasets were throwing errors upon reading. For example, some were completely empty files. I verified that this was not a problem with our code, but rather, OpenML was sending these empty files.
* read\_datasets.py - This file simply reads all of the datasets on disk. It was written to see how widespread the problems that I was detecting when running preprocess\_datasets.py were. This is when a lot of the error logging in the data package was written.
* purge\_bad\_datasets.py - This file attempts to read all the datasets on disk and gets rid of the ones that throw errors.