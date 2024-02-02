#!/usr/bin/bash
set -e

ROOT=$(dirname $0)
cd $ROOT


module purge
module load pytorch-gpu/py3/2.1.1 
cd qallm
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --no-cache-dir -r requirements.txt
pip install -r requirements.txt
pip install .