#!/usr/bin/bash
set -e

ROOT=$(dirname $0)
cd $ROOT

module purge
module load pytorch-gpu/py3/2.1.1 
cd qallm
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install .
