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
from datasets import load_from_disk

from transformers import (
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

def loadRAG(dataset_path, rag_model_path):
    dataset = load_from_disk(dataset_path)
    index_path = os.path.join(
        "/".join(dataset_path.split("/")[:-1]), "hnsw_index.faiss"
    )
    dataset.load_faiss_index("embeddings", index_path)  # to reload the index

    retriever = RagRetriever.from_pretrained(
        rag_model_path, index_name="custom", indexed_dataset=dataset
    )
    model = RagSequenceForGeneration.from_pretrained(rag_model_path, retriever=retriever)
    tokenizer = RagTokenizer.from_pretrained(rag_model_path)

    return retriever, model, tokenizer, dataset


def batch_inf(x, retriever, model, tokenizer):
    input_ids = tokenizer.question_encoder(x, padding = True, return_tensors="pt")["input_ids"]
    question_hidden_states = model.question_encoder(input_ids)[0]
    _, doc_ids, _ = model.retriever.retrieve(question_hidden_states.detach().numpy(), 1)
    generated = model.generate(input_ids)
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return generated_string, doc_ids
    
