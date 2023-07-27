import typing as t
import collections
import functools
import argparse
import warnings
import hashlib
import random
import string
import glob
import sys
import os
import re

import matplotlib.pyplot as plt
import sentence_transformers
import numpy.typing as npt
import sklearn.cluster
import seaborn as sns
import scipy.spatial
import pandas as pd
import numpy as np
import colorama
import spacy
import nltk
import tqdm


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import umap


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


STOPWORDS = frozenset(nltk.corpus.stopwords.words("portuguese"))
REG_WHITESPACE_SPAN = re.compile(r"\s+")


def sample_items_(items: list[str], /, source_sample_factor: float | int | None) -> list[str]:
    if source_sample_factor is None:
        return items

    random.shuffle(items)
    n = int(
        source_sample_factor if source_sample_factor >= 1.0 else source_sample_factor * len(items)
    )
    n = max(1, n)
    return items[:n]


def read_corpus(
    corpus_uris: list[str],
    sep: str,
    file_ext: str,
    source_sample_factor: int | float | None,
    sample_random_state: int | None,
) -> list[str]:
    texts: list[str] = []

    if sample_random_state is not None:
        random.seed(sample_random_state)

    for curi in corpus_uris:
        cur_texts: list[str] = []
        print_prefix: str = f"({curi}) {'Read' if source_sample_factor is None else 'Sampled'} "

        if os.path.isdir(curi):
            furis = glob.glob(os.path.join(curi, f"**/*.{file_ext}"), recursive=True)
            furis = sample_items_(furis, source_sample_factor=source_sample_factor)

            print(f"{print_prefix}{len(furis)} files with '.{file_ext}' extension.")

            for furi in furis:
                with open(furi, "r", encoding="utf-8") as f_in:
                    cur_texts.append(f_in.read())

        else:
            df: t.Union[pd.DataFrame, list[t.Any]]
            df = pd.read_csv(curi, sep=sep)
            df = df.iloc[:, 0].tolist()
            df = sample_items_(df, source_sample_factor=source_sample_factor)

            print(f"{print_prefix}{len(cur_texts)} non-empty items from corpus.")

            cur_texts.extend([str(item) for item in cur_texts if item])

        texts.extend(cur_texts)

    print(f"Number of sources: {len(corpus_uris)}")
    print(f"Number of files  : {len(texts)}")

    if not texts:
        raise ValueError("No text has been loaded.")

    return texts


def cluster_embs(
    embs: npt.NDArray[np.float64], dbscan_kwargs: dict[str, t.Any]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    clusterer = sklearn.cluster.DBSCAN(**dbscan_kwargs)
    cluster_ids: npt.NDArray[np.int64] = clusterer.fit_predict(embs)

    clusters = np.unique(cluster_ids)
    clusters = clusters[clusters >= 0]

    n_clusters = clusters.size + 1
    print(f"Number of clusters:", n_clusters)

    if n_clusters == 1:
        raise ValueError(
            "All points were predicted as noise points. Please provide distinct DBSCAN parameters."
        )

    medoids = np.empty((clusters.size, 2))

    for i, cluster_id in enumerate(clusters):
        embs_cur_cluster = embs[cluster_ids == cluster_id, :]
        centroid = np.mean(embs_cur_cluster, axis=0, keepdims=True)
        medoid_id = np.argmin(scipy.spatial.distance.cdist(centroid, embs_cur_cluster)).squeeze()
        medoid = embs_cur_cluster[medoid_id, :]
        medoids[i] = medoid

    return cluster_ids, medoids


def build_cache_uri(resource_name: str, args, /) -> str:
    corpus_urns = [os.path.basename(item.rstrip("/")) for item in args.corpus_uris]
    corpus_urns.sort()

    hasher = hashlib.sha256()
    for urn in corpus_urns:
        hasher.update(urn.encode())

    corpus_urns_sha256 = hasher.hexdigest()

    sbert_urn = os.path.basename(args.sbert_uri.rstrip("/"))

    return os.path.join(args.cache_dir, f"{resource_name}_{sbert_urn}_{corpus_urns_sha256}.npy")


def compute_very_common_tokens(
    texts,
    spacy_model_name: str,
    cutoff: int | float,
    minimum_length: int,
    banned_tokens: set[str] | None,
    disable_progress_bar: bool,
) -> list[str]:
    spacy_model = spacy.load(spacy_model_name)

    fn_spacy_model = functools.partial(
        spacy_model,
        disable=["tok2vec", "morphologizer", "parser", "attribute_ruler", "ner"],
    )

    banned_tokens = banned_tokens or set()

    def fn_preproc(doc):
        doc = doc[: spacy_model.max_length - 1]
        doc = doc.lower()
        doc = REG_WHITESPACE_SPAN.sub(" ", doc)
        return doc

    token_freqs: collections.Counter[str] = collections.Counter()

    for doc in tqdm.tqdm(texts, desc="Computing very common tokens", disable=disable_progress_bar):
        doc = fn_preproc(doc)
        lemmas = [token.lemma_ for token in fn_spacy_model(doc)]
        token_freqs.update(lemmas)

    token_freqs_clean = collections.Counter(
        {
            k: v
            for k, v in token_freqs.items()
            if k not in STOPWORDS
            and k not in banned_tokens
            and k not in string.punctuation
            and len(k) >= minimum_length
        }
    )

    if isinstance(cutoff, float):
        cutoff = int(cutoff * len(token_freqs_clean))

    very_common_tokens = [v for v, _ in token_freqs_clean.most_common(cutoff)]

    return very_common_tokens


def compute_cluster_keywords(
    texts: list[str],
    spacy_model_name: str,
    cluster_ids: npt.NDArray[np.int64],
    banned_tokens: set[str],
    minimum_length: int,
    keywords_per_cluster: int,
    disable_progress_bar: bool,
) -> dict[int, list[str]]:
    keywords: dict[int, list[str]] = {}

    clusters = np.unique(cluster_ids)
    clusters = clusters[clusters >= 0]

    for cluster_id in tqdm.tqdm(
        clusters, desc="Computing cluster keywords", disable=disable_progress_bar
    ):
        texts_cur_cluster = [text for text, cl_id in zip(texts, cluster_ids) if cl_id == cluster_id]
        keywords[cluster_id] = compute_very_common_tokens(
            texts_cur_cluster,
            spacy_model_name=spacy_model_name,
            cutoff=keywords_per_cluster,
            minimum_length=minimum_length,
            banned_tokens=banned_tokens,
            disable_progress_bar=True,
        )

    return keywords


def plot_base(
    fig,
    ax,
    embs: npt.NDArray[np.float64],
    cluster_ids: npt.NDArray[np.int64] | None,
    args,
    remove_labels: bool,
) -> None:
    X, Y = embs.T

    sns.kdeplot(
        x=X,
        y=Y,
        fill=True,
        thresh=args.kdeplot_thresh,
        levels=args.kdeplot_levels,
        alpha=args.kdeplot_alpha,
        cmap=args.kdeplot_cmap,
        ax=ax,
    )

    sns.scatterplot(
        x=X,
        y=Y,
        hue=cluster_ids,
        style=cluster_ids,
        ax=ax,
        legend=False,
        palette="hsv_r" if cluster_ids is not None else None,
        linewidth=0,
    )

    if remove_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

    sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)


def plot_keywords(
    fig,
    ax,
    embs: npt.NDArray[np.float64],
    medoids: npt.NDArray[np.float64],
    cluster_keywords: dict[str, list[str]],
    args,
) -> None:
    xytextcoords = np.empty((len(medoids), 2), dtype=float)
    xytextcoords[:, 0] = medoids[:, 0] >= 0.5 * np.add(*ax.get_xlim())

    for disc in [False, True]:
        cur_inds = np.flatnonzero(xytextcoords[:, 0] == disc)
        coord_ids = np.argsort(np.argsort(medoids[cur_inds, 1]))
        frac_space = np.linspace(-0.025, 0.975, len(cur_inds))
        xytextcoords[cur_inds, 1] = frac_space[coord_ids]

    min_, max_ = -args.keyword_inflate_prop, (1.0 + args.keyword_inflate_prop)
    xytextcoords[:, 0] = xytextcoords[:, 0] * (max_ - min_) + min_

    arrowprops = {
        "arrowstyle": "->",
        "connectionstyle": "arc3,rad=0.25",
        "edgecolor": args.arrow_color,
        "alpha": args.arrow_alpha,
        "linestyle": args.arrow_linestyle,
        "lw": args.arrow_linewidth,
    }

    bbox = {
        "boxstyle": "square",
        "facecolor": "white",
        "lw": 1.0,
        "edgecolor": "black",
        "alpha": 1.0,
    }

    mid_x = 0.50 * float(np.add(*ax.get_xlim()))
    mid_y = 0.50 * float(np.add(*ax.get_ylim()))

    for i, (medoid, xytext) in enumerate(zip(medoids, xytextcoords)):
        cur_arrowprops = arrowprops
        cx, cy = medoid

        if (cx >= mid_x and cy <= mid_y) or (cx <= mid_x and cy >= mid_y):
            cur_arrowprops = arrowprops.copy()
            cur_arrowprops["connectionstyle"] = "arc3,rad=-0.25"

        ax.annotate(
            "\n".join(cluster_keywords[i]),
            xy=tuple(medoid),
            xytext=tuple(xytext),
            fontsize=args.keyword_font_size,
            horizontalalignment="right" if cx >= mid_x else "left",
            arrowprops=cur_arrowprops,
            bbox=bbox,
            xycoords="data",
            textcoords="axes fraction",
        )


def run(args) -> None:
    texts = read_corpus(
        corpus_uris=args.corpus_uris,
        sep=args.corpus_file_sep,
        file_ext=args.corpus_dir_ext,
        source_sample_factor=args.source_sample_factor,
        sample_random_state=args.sample_random_state,
    )

    embs_uri = build_cache_uri("embs", args)
    very_common_tokens_uri = build_cache_uri("very_common_tokens", args)

    if not args.ignore_cache and os.path.exists(embs_uri):
        embs = np.load(embs_uri)
        print(f"{colorama.Fore.YELLOW}Found cached '{embs_uri}'.{colorama.Style.RESET_ALL}")

    else:
        sbert = sentence_transformers.SentenceTransformer(args.sbert_uri, device=args.sbert_device)
        projector = umap.UMAP(n_components=2, metric="cosine", random_state=args.umap_random_state)

        embs = sbert.encode(
            texts, batch_size=args.sbert_batch_size, show_progress_bar=not args.disable_progress_bar
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embs = projector.fit_transform(embs)

        np.save(embs_uri, embs)

    if not args.ignore_cache and os.path.exists(very_common_tokens_uri):
        with open(very_common_tokens_uri, "r", encoding="utf-8") as f_in:
            very_common_tokens = set(f_in.read().split("\n"))

        print(
            f"{colorama.Fore.YELLOW}Found cached '{very_common_tokens_uri}'.{colorama.Style.RESET_ALL}"
        )

    else:
        very_common_tokens = set(
            compute_very_common_tokens(
                texts=texts,
                spacy_model_name=args.spacy_model_name,
                cutoff=args.very_common_tokens_cutoff,
                disable_progress_bar=args.disable_progress_bar,
                banned_tokens=None,
                minimum_length=1,
            )
        )

        with open(very_common_tokens_uri, "w") as f_out:
            f_out.write("\n".join(sorted(very_common_tokens)))

    embs = (embs - embs.min(axis=0)) / np.ptp(embs, axis=0)

    fig, ax = plt.subplots(1, figsize=(args.fig_width, args.fig_height), layout="tight")

    try:
        cluster_ids, medoids = cluster_embs(
            embs, dbscan_kwargs={"eps": args.dbscan_eps, "min_samples": args.dbscan_min_samples}
        )

    except ValueError as err:
        plot_base(fig=fig, ax=ax, embs=embs, cluster_ids=None, args=args, remove_labels=False)
        plt.show()
        raise ValueError from err

    plot_base(fig=fig, ax=ax, embs=embs, cluster_ids=cluster_ids, args=args, remove_labels=True)

    user_banned_tokens = set(args.banned_tokens.split(",")) if args.banned_tokens else set()

    if user_banned_tokens:
        print(f"Found {len(user_banned_tokens)} custom banned tokens.")

    cluster_keywords = compute_cluster_keywords(
        texts=texts,
        spacy_model_name=args.spacy_model_name,
        cluster_ids=cluster_ids,
        keywords_per_cluster=args.keywords_per_cluster,
        banned_tokens=very_common_tokens | user_banned_tokens,
        minimum_length=args.keyword_minimum_length,
        disable_progress_bar=args.disable_progress_bar,
    )

    plot_keywords(
        fig=fig,
        ax=ax,
        embs=embs,
        medoids=medoids,
        cluster_keywords=cluster_keywords,
        args=args,
    )

    if not args.do_not_show:
        plt.show()

    if args.output:
        fig.savefig(args.output, bbox_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("sbert_uri")
    parser.add_argument("corpus_uris", action="extend", nargs="+", type=str)
    parser.add_argument("--output", "-o", default="output.pdf")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")

    parser_plot = parser.add_argument_group("plot arguments")
    parser_plot.add_argument("--scatterplot-palette", default="hsv")
    parser_plot.add_argument("--kdeplot-thresh", default=1e-3, type=float)
    parser_plot.add_argument("--kdeplot-levels", default=12, type=int)
    parser_plot.add_argument("--kdeplot-alpha", default=1.0, type=float)
    parser_plot.add_argument("--kdeplot-cmap", default="viridis")
    parser_plot.add_argument("--arrow-color", default="black")
    parser_plot.add_argument("--arrow-alpha", default=0.80, type=float)
    parser_plot.add_argument("--arrow-linestyle", default="dashed")
    parser_plot.add_argument("--arrow-linewidth", default=2.0, type=float)
    parser_plot.add_argument("--keyword-inflate-prop", default=0.05, type=float)
    parser_plot.add_argument("--fig-width", default=12.0, type=float)
    parser_plot.add_argument("--fig-height", default=8.0, type=float)
    parser_plot.add_argument("--do-not-show", action="store_true")

    parser_sbert = parser.add_argument_group("dbscan arguments")
    parser_sbert.add_argument("--dbscan-eps", default=0.05, type=float)
    parser_sbert.add_argument("--dbscan-min-samples", default=10, type=int)

    parser_control = parser.add_argument_group("keywords arguments")
    parser_control.add_argument("--keywords-per-cluster", default=3, type=int)
    parser_control.add_argument("--banned-tokens", default=None, type=str)
    parser_control.add_argument("--keyword-minimum-length", default=3, type=int)
    parser_control.add_argument("--keyword-font-size", default=10, type=int)
    parser_control.add_argument("--very-common-tokens-cutoff", default=0.01, type=float)
    parser_control.add_argument("--spacy-model-name", default="pt_core_news_sm")

    parser_sbert = parser.add_argument_group("sbert arguments")
    parser_sbert.add_argument("--sbert-device", default=None, type=str)
    parser_sbert.add_argument("--sbert-batch-size", default=128, type=int)

    parserumap = parser.add_argument_group("umap arguments")
    parserumap.add_argument("--umap-random-state", default=182783, type=int)

    parser_corpus = parser.add_argument_group("corpus arguments")
    parser_corpus.add_argument("--corpus-file-sep", default=",")
    parser_corpus.add_argument("--corpus-dir-ext", default="txt")
    parser_corpus.add_argument("--source-sample-factor", default=None, type=float)
    parser_corpus.add_argument("--sample-random-state", default=8101192, type=int)

    parser_seaborn = parser.add_argument_group("seaborn arguments")
    parser_seaborn.add_argument("--seaborn-context", default="paper")
    parser_seaborn.add_argument("--seaborn-style", default="whitegrid")
    parser_seaborn.add_argument("--seaborn-font-scale", default=1.0, type=float)

    args = parser.parse_args()

    args.cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        spacy.load(args.spacy_model_name)

    except OSError:
        import spacy.cli

        spacy.cli.download(args.spacy_model_name)

    sns.set_theme(
        context=args.seaborn_context,
        style=args.seaborn_style,
        palette="colorblind",
        font_scale=args.seaborn_font_scale,
    )

    run(args)
