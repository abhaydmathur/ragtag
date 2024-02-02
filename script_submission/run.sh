#!/usr/bin/bash
set -e

export SCARF_NO_ANALYTICS=true
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ANONYMIZED_TELEMETRY=False

QUESTIONS=$(realpath $1)
OUTPUT=$(realpath $2)

ROOT=$(dirname $0)
cd $ROOT

module purge
module load pytorch-gpu/py3/2.1.1 
cd qallm
pip install -r requirements.txt
pip install .
export PATH=/linkhome/rech/genner01/ufk76ad/.local/bin:$PATH
python qallm -c "$QUESTIONS" -o "$OUTPUT"
