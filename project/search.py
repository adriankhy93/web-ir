import pandas as pd
import string
from rank_bm25 import BM25Okapi

import numpy as np
import json
import markdown

from utils import preprocess_pipeline


def get_relevant_documents_indexes(search_engine, query, k):
    # Get the top k indexes
    query = preprocess_pipeline(query)
    tokenized_query = query.split(" ")

    doc_scores = search_engine.get_scores(tokenized_query)
    num_of_rel_docs = sum(doc_scores != 0)
    k = min(num_of_rel_docs, k)

    top_k_indexs = np.argsort(doc_scores)[::-1][:k]
    return top_k_indexs


def get_prof_ids_given_paper_id(prof_paper_df, paper_id):
    return prof_paper_df.loc[
        prof_paper_df["paper_id"].apply(lambda x: x == paper_id), "prof_id"
    ].tolist()


def get_search_display_ids(paper_df, prof_paper_df, top_k_indexs):
    retrieved_paper_ids = paper_df.loc[top_k_indexs, "paper_id"].tolist()
    retrieved_prof_ids = [
        get_prof_ids_given_paper_id(prof_paper_df, i) for i in retrieved_paper_ids
    ]
    return retrieved_paper_ids, retrieved_prof_ids


def get_prof_name_from_id(prof_id, prof_df):
    return prof_df.loc[prof_id, "prof_name"]


def get_prof_title_from_id(prof_id, prof_df):
    return prof_df.loc[prof_id, "prof_title"]


def get_prof_email_from_id(prof_id, prof_df):
    return prof_df.loc[prof_id, "prof_email"]


def get_prof_aoi_from_id(prof_id, prof_df):
    return prof_df.loc[prof_id, "prof_area_of_interest"]


def get_prof_link_from_id(prof_id, prof_df):
    return prof_df.loc[prof_id, "prof_link"]


def get_paper_title_from_id(paper_id, paper_df):
    return paper_df.loc[paper_id, "paper_title"]


def get_paper_abstract_from_id(paper_id, paper_df):
    return paper_df.loc[paper_id, "paper_abstract"]


def get_paper_link_from_id(paper_id, paper_df):
    return paper_df.loc[paper_id, "paper_link"]


def get_search_display_data(retrieved_paper_ids, retrieved_prof_ids, prof_df, paper_df):
    prof_names = [
        [get_prof_name_from_id(prof_id, prof_df) for prof_id in prof_ids]
        for prof_ids in retrieved_prof_ids
    ]

    paper_titles = [
        get_paper_title_from_id(paper_id, paper_df) for paper_id in retrieved_paper_ids
    ]

    paper_abstracts = [
        get_paper_abstract_from_id(paper_id, paper_df)
        for paper_id in retrieved_paper_ids
    ]

    paper_links = [
        get_paper_link_from_id(paper_id, paper_df) for paper_id in retrieved_paper_ids
    ]

    N = len(paper_links)

    json_output = [
        {
            "rank": i,
            "prof_names": ", ".join(prof_names[i]),
            "paper_title": paper_titles[i],
            # "paper_abstract": markdown.markdown(paper_abstracts[i]),
            "paper_abstract": paper_abstracts[i],
            "paper_link": paper_links[i],
        }
        for i in range(N)
    ]

    return json_output


def get_prof_display_data(prof_id, prof_df):
    prof_name = get_prof_name_from_id(prof_id, prof_df)
    prof_title = get_prof_title_from_id(prof_id, prof_df)
    prof_email = get_prof_email_from_id(prof_id, prof_df)
    prof_aoi = get_prof_aoi_from_id(prof_id, prof_df)
    prof_link = get_prof_link_from_id(prof_id, prof_df)

    json_output = {
        "prof_name": prof_name,
        "prof_title": prof_title,
        "prof_email": prof_email,
        "prof_aoi": prof_aoi,
        "prof_link": prof_link,
    }
    return json_output


def perform_search(search_engine, query, k, paper_df, prof_df, prof_paper_df):
    top_k_indexs = get_relevant_documents_indexes(search_engine, query, k)

    retrieved_paper_ids, retrieved_prof_ids = get_search_display_ids(
        paper_df, prof_paper_df, top_k_indexs
    )

    json_output = get_search_display_data(
        retrieved_paper_ids, retrieved_prof_ids, prof_df, paper_df
    )

    return json_output


def get_all_profs_info(prof_df):
    N = len(prof_df)
    json_output = []
    for i in range(N):
        row = prof_df.loc[i]
        prof_id = row["prof_id"]
        prof_name = row["prof_name"]
        prof_title = row["prof_title"]
        prof_email = row["prof_email"]
        prof_area_of_interest = row["prof_area_of_interest"]
        prof_link = row["prof_link"]

        # paper_ids = prof_paper_df.loc[
        #     prof_paper_df["prof_id"].apply(lambda x: x == prof_id), "paper_id"
        # ].tolist()

        # paper_titles = paper_df.loc[
        #     paper_df["paper_id"].apply(lambda x: x in paper_ids), "paper_title"
        # ].tolist()

        json_output.append(
            {
                "prof_id": prof_id,
                "prof_name": prof_name,
                "prof_title": prof_title,
                "prof_email": prof_email,
                "prof_area_of_interest": prof_area_of_interest,
                "prof_link": prof_link,
                # "paper_titles": paper_titles,
            }
        )
    return json_output
