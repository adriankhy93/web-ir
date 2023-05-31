import pandas as pd
import string
from rank_bm25 import BM25Okapi

import numpy as np


def replace_punct_with_space(text):
    return text.translate(str.maketrans(string.punctuation, " " * 32))


def replace_multiple_spaces_with_single_space(text):
    return " ".join(text.split())


def preprocess_pipeline(text):
    preproc_text = text.lower()
    preproc_text = replace_punct_with_space(preproc_text)
    preproc_text = replace_multiple_spaces_with_single_space(preproc_text)
    return preproc_text


def prepare_paper_data(paper_df):
    paper_df["preproc_paper_title"] = paper_df["paper_title"].apply(
        lambda x: preprocess_pipeline(x)
    )

    paper_df["preproc_paper_abstract"] = paper_df["paper_abstract"].apply(
        lambda x: preprocess_pipeline(x)
    )

    ## construct doc corpus
    #   join title, abstract

    paper_df["doc"] = (
        paper_df["preproc_paper_title"] + " " + paper_df["preproc_paper_abstract"]
    )

    paper_df = paper_df.drop("preproc_paper_title", axis=1)
    paper_df = paper_df.drop("preproc_paper_abstract", axis=1)
    return paper_df


def init_search_engine(corpus):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25
