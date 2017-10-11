#!/bin/bash

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source activate samp-phase3
python main.py $1 $2 $3
