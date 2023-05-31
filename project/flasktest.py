from flask import Flask, render_template, url_for, flash, redirect, request
from flaskext.markdown import Markdown

# from search import search
# from filter import Filter
# from storage import DBStorage
import html

from utils import prepare_paper_data, init_search_engine
from search import perform_search, get_all_profs_info

import pandas as pd
from flask_misaka import Misaka
import os

## LOAD DATA
paper_df = pd.read_csv("data/paper_df.csv")
prof_df = pd.read_csv("data/prof_df.csv")
prof_paper_df = pd.read_csv("data/prof_paper_df.csv")
paper_df = prepare_paper_data(paper_df)

# Prepare Search Engine

search_engine = init_search_engine(paper_df["doc"])


IMGS_FOLDER = os.path.join("imgs")
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = IMGS_FOLDER
Misaka(app)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form["query"]
        k = 20

        results = perform_search(
            search_engine, query, k, paper_df, prof_df, prof_paper_df
        )
        return render_template("home.html", results=results)
    else:
        return render_template("home.html")


@app.route("/profs")
def profs():
    results = get_all_profs_info(prof_df)
    return render_template("profs.html", results=results)


app.run(host="0.0.0.0", port=80, debug=False)  # or 5001
