# Plot topics

![Output example 01.](./examples/example01.png)

---

Tested with **Python 3.10.2**.

---
## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)

---

## Installation
```bash
python -m pip install -r requirements.txt
```

---

## Usage

A typical execution is going to look like:
```bash
python plot_topics.py sbert_path data_source_1 data_source_2 ... \
  --output="output.pdf" \
  --dbscan-eps=0.10 \
  --dbscan-min-samples=10 \
  --keywords-per-cluster=3 \
  --banned-keywords="keyword1,keyword2,keyword3" \
  --corpus-dir-ext="txt" \
  --corpus-file-sep="," \
  --corpus-file-col-index=0 \
  --source-sample-factor=0.10 \
  --scatterplot-palette="hsv" \
  --kdeplot-cmap="viridis"
```

Each `data_source_i` can be either a directory or a single file.

- *If it is a directory*, this script will search **recursively** every file with extension specified by `--corpus-dir-ext` (default is `txt`); or
- *If it is a single file*, this script will read it using [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) as follows: `pd.read_csv(data_source_i, sep=corpus-file-sep, usecols=corpus-file-col-index, index_col=False).squeeze()`.

You can access the documentation (which includes more configuration options) by running:
```bash
python plot_topics.py --help
```
