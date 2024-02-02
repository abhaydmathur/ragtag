import logging
import pandas as pd
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import faiss
import glob 
import torch
import numpy as np
from datasets import Features, Sequence, Value, load_from_disk, Dataset

from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)

CONTEXT_WINDOW = None

logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"




def main(
    rag_example_args: "RagExampleArguments",
    processing_args: "ProcessingArguments",
    index_hnsw_args: "IndexHnswArguments",
):
    ######################################
    logger.info("Step 1 - Create the dataset")
    ######################################

    dataset = load_from_disk(rag_example_args.dataset_path)
    index_path = os.path.join(
        "/".join(rag_example_args.dataset_path.split("/")[:-1]), "hnsw_index.faiss"
    )
    dataset.load_faiss_index("embeddings", index_path)  # to reload the index

    # ######################################
    # logger.info("Step 3 - Load RAG")
    # ######################################

    # # Easy way to load the model
    retriever = RagRetriever.from_pretrained(
        rag_example_args.rag_model_name, index_name="custom", indexed_dataset=dataset
    )
    model = RagSequenceForGeneration.from_pretrained(rag_example_args.rag_model_name, retriever=retriever)
    tokenizer = RagTokenizer.from_pretrained(rag_example_args.rag_model_name)

    # # For distributed fine-tuning you'll need to provide the paths instead, as the dataset and the index are loaded separately.
    # # retriever = RagRetriever.from_pretrained(rag_model_name, index_name="custom", passages_path=passages_path, index_path=index_path)

    # ######################################
    # logger.info("Step 4 - Have fun")
    # ######################################

    question = ["How many Italian government data requests did LinkedIn receive in 2022? Please provide the URL of the source.", "What's your opinion on twitter's content regulations?"] 
    input_ids = tokenizer.question_encoder(question, padding = True, return_tensors="pt")["input_ids"]
    question_hidden_states = model.question_encoder(input_ids)[0]
    _, doc_ids, _ = model.retriever.retrieve(question_hidden_states.detach().numpy(),1)
    #doc_scores = torch.bmm(
     #   question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    #)squeeze(1)
    #print(doc_scores)
#    print(f"Confidence Range : {np.min(doc_scores)}, {np.max(doc_scores)}")
    #doc_ids = np.argmax(doc_scores)
    generated = model.generate(input_ids)
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
    logger.info(f"Q: {question}")
    logger.info(f"A:  {generated_string}")
    logger.info(f"Docs: {doc_ids}")
    logger.info(f"Doc_Titles: {[dataset[id]['title'] for id in doc_ids]}")

@dataclass
class RagExampleArguments:
    dataset_path: str = field(
    # csv_path: str = field(
        default="mds_dataset/api_dataset",
        metadata={
            "help": "Path to directory indexed dataset."
        },
    )
    question: Optional[str] = field(
        default=None,
        metadata={
            "help": "Question that is passed as input to RAG. Default is 'How many Italian government data requests did LinkedIn receive in 2022? Please provide the URL of the source.'"
        },
    )
    rag_model_name: str = field(
        default="facebook/rag-sequence-nq",
        metadata={
            "help": "The RAG model to use. Either 'facebook/rag-sequence-nq' or 'facebook/rag-token-nq'"
        },
    )
    dpr_ctx_encoder_model_name: str = field(
        default="facebook/dpr-ctx_encoder-multiset-base",
        metadata={
            "help": (
                "The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-single-nq-base' or"
                " 'facebook/dpr-ctx_encoder-multiset-base'"
            )
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a directory where the dataset passages and the index will be saved"
        },
    )

    context_window : Optional[int] = field(
      default = 10000,
      metadata={
            "help": "Context window"
        },
    )


@dataclass
class ProcessingArguments:
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use to split the documents into passages. Default is single process."
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )


@dataclass
class IndexHnswArguments:
    d: int = field(
        default=768,
        metadata={
            "help": "The dimension of the embeddings to pass to the HNSW Faiss index."
        },
    )
    m: int = field(
        default=128,
        metadata={
            "help": (
                "The number of bi-directional links created for every new element during the HNSW index construction."
            )
        },
    )


if __name__ == "__main__":
    # global CONTEXT_WINDOW
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser(
        (RagExampleArguments, ProcessingArguments, IndexHnswArguments)
    )
    (
        rag_example_args,
        processing_args,
        index_hnsw_args,
    ) = parser.parse_args_into_dataclasses()

    CONTEXT_WINDOW = rag_example_args.context_window
    with TemporaryDirectory() as tmp_dir:
        rag_example_args.output_dir = rag_example_args.output_dir or tmp_dir
        main(rag_example_args, processing_args, index_hnsw_args)
