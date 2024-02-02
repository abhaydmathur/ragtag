# ragtag

## Usage

Our repository contains submodules, linking to the datasets made available for the hackathon. To reproduce our results
it is thus required to clone this repository with the recursion flag, as below, and you are located on the ```$WORK``` directory on the Jean-Zay supercomputer.

```bash
git clone --recurse-submodules git@github.com:abhaydmathur/ragtag.git ; cd ragtag
cd qallm
.venv/bin/pip install --no-cache-dir -r requirements.txt
pip install .
```

```bash
usage: qallm [-h] [-c file] [-i question] [-o file] [-v]

Adversarial machine learning robustness evaluations for ML based N-IDS.

options:
  -h, --help            show this help message and exit
  -c file, --csv file   relative path to the csv file configuration the questions to be answered. If not defined, it is required to define an input
                        question as a string.
  -i question, --input question
                        String input question to be answered. If not defined, it is required to define an input csv file.
  -o file, --output file
                        relative path to the file in which the output shall be written to. If not defined, the prediction results will be displayed
                        on the standard output.
  -v, --verbose         Set logging level to INFO to obtain information about the internal program execution. Without the verbose flag, only Errors
                        are diplayed in stdout.
```

## Preproc Dataset

```bash
python3 create_dataset.py --md_path all_mds --output_dir mds_dataset
```

Usage : 

```txt
usage: create_dataset.py [-h] [--md_path MD_PATH] [--rag_model_name RAG_MODEL_NAME]
                         [--dpr_ctx_encoder_model_name DPR_CTX_ENCODER_MODEL_NAME]
                         [--output_dir OUTPUT_DIR] [--num_proc NUM_PROC] [--batch_size BATCH_SIZE]
                         [--d D] [--m M]

options:
  -h, --help            show this help message and exit
  --md_path MD_PATH     Path to directory with all markdown files. (default: /content/test_data)
  --rag_model_name RAG_MODEL_NAME
                        The RAG model to use. Either 'facebook/rag-sequence-nq' or 'facebook/rag-
                        token-nq' (default: facebook/rag-sequence-nq)
  --dpr_ctx_encoder_model_name DPR_CTX_ENCODER_MODEL_NAME
                        The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-
                        single-nq-base' or 'facebook/dpr-ctx_encoder-multiset-base' (default:
                        facebook/dpr-ctx_encoder-multiset-base)
  --output_dir OUTPUT_DIR
                        Path to a directory where the dataset passages and the index will be saved
                        (default: None)
  --num_proc NUM_PROC   The number of processes to use to split the documents into passages.
                        Default is single process. (default: None)
  --batch_size BATCH_SIZE
                        The batch size to use when computing the passages embeddings using the DPR
                        context encoder. (default: 16)
  --d D                 The dimension of the embeddings to pass to the HNSW Faiss index. (default:
                        768)
  --m M                 The number of bi-directional links created for every new element during
                        the HNSW index construction. (default: 128)
```

## Retrieval (Single Sample)

```txt
usage: retrieval.py [-h] [--dataset_path DATASET_PATH] [--question QUESTION]
                    [--rag_model_name RAG_MODEL_NAME]
                    [--dpr_ctx_encoder_model_name DPR_CTX_ENCODER_MODEL_NAME]
                    [--output_dir OUTPUT_DIR] [--context_window CONTEXT_WINDOW]
                    [--num_proc NUM_PROC] [--batch_size BATCH_SIZE] [--d D] [--m M]

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path to directory indexed dataset. (default: mds_dataset/api_dataset)
  --question QUESTION   Question that is passed as input to RAG. Default is 'How many Italian
                        government data requests did LinkedIn receive in 2022? Please provide the
                        URL of the source.' (default: None)
  --rag_model_name RAG_MODEL_NAME
                        The RAG model to use. Either 'facebook/rag-sequence-nq' or 'facebook/rag-
                        token-nq' (default: facebook/rag-sequence-nq)
  --dpr_ctx_encoder_model_name DPR_CTX_ENCODER_MODEL_NAME
                        The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-
                        single-nq-base' or 'facebook/dpr-ctx_encoder-multiset-base' (default:
                        facebook/dpr-ctx_encoder-multiset-base)
  --output_dir OUTPUT_DIR
                        Path to a directory where the dataset passages and the index will be saved
                        (default: None)
  --context_window CONTEXT_WINDOW
                        Context window (default: 10000)
  --num_proc NUM_PROC   The number of processes to use to split the documents into passages.
                        Default is single process. (default: None)
  --batch_size BATCH_SIZE
                        The batch size to use when computing the passages embeddings using the DPR
                        context encoder. (default: 16)
  --d D                 The dimension of the embeddings to pass to the HNSW Faiss index. (default:
                        768)
  --m M                 The number of bi-directional links created for every new element during
                        the HNSW index construction. (default: 128)
```

## Retrieval (Batch)