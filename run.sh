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
qallm -c "$QUESTIONS" -o "$OUTPUT"
