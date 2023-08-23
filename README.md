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

To use this script, you need to provide two required parameters:
- A Sentence Model path (sentence-transformer); and
- At least one text source path.

```bash
python run.py /path/to/sbert /path/to/data/source1 /path/to/data/source2 ...
```

However, you will likely need to specific some parameters that control the quality of your plot plot, in particular the DBSCAN hyperparameters.
A typical run will look like this:
```bash
python run.py /path/to/sbert /path/to/data/source1 /path/to/data/source2 ... \
  --output="output.pdf" \
  --dbscan-eps=0.10 \
  --dbscan-min-samples=10 \
  --keywords-per-cluster=3 \
  --banned-keywords="keyword1,keyword2,keyword3" \
  --scatterplot-palette="hsv" \
  --kdeplot-cmap="viridis"
```

Some settings associated with loading data may also be required:
```bash
python run.py /path/to/sbert /path/to/data/source1 /path/to/data/source2 ... \
  --output="output.pdf" \
  --dbscan-eps=0.10 \
  --dbscan-min-samples=10 \
  --keywords-per-cluster=3 \
  --banned-keywords="keyword1,keyword2,keyword3" \
  --scatterplot-palette="hsv" \
  --kdeplot-cmap="viridis"
  --corpus-dir-ext="txt" \
  --corpus-file-sep="," \
  --corpus-file-col-index=0 \
  --source-sample-factor=0.10
```

Each `/path/to/data/source` can be either a directory or a single file.

- *If it is a directory*, this script will search **recursively** every file with extension specified by `--corpus-dir-ext` (default is `txt`); or
- *If it is a single file*...
   - *and if it has .txt extension*: this script will simply read it;
   - *Otherwise*: this script will read it using [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) as follows: `pd.read_csv(/path/to/data/source, sep=corpus-file-sep, usecols=[corpus-file-col-index], index_col=False).squeeze()`. Note that this implies that a single column is read per file.

You can access the documentation (which includes more configuration options) by running:
```bash
python run.py --help
```
