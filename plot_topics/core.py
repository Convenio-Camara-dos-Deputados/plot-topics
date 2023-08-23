"""Plot topic keywords from semantic embedding clusters."""
import typing as t
import collections
import functools
import argparse
import warnings
import hashlib
import random
import string
import glob
import os
import re

import matplotlib.pyplot as plt
import sentence_transformers
import numpy as np
import numpy.typing as npt
import sklearn.preprocessing
import sklearn.cluster
import seaborn as sns
import scipy.spatial
import pandas as pd
import colorama
import spacy
import nltk
import tqdm

from . import rslp_s


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


def sample_items_(items: list[str], /, source_sample_size: float | int | None) -> list[str]:
    if source_sample_size is None:
        return items

    random.shuffle(items)
    n = int(source_sample_size if source_sample_size >= 1.0 else (source_sample_size * len(items)))
    n = max(1, n)
    return items[:n]


def read_corpus(
    corpus_uris: list[str],
    sep: str,
    file_ext: str,
    keep_index: int,
    source_sample_size: int | float | None,
    sample_random_state: int | None,
    max_chars_per_instance: int | None,
) -> list[str]:
    corpus_uris = [item.strip('"').rstrip("/") for item in corpus_uris]
    texts: list[str] = []

    for curi in corpus_uris:
        if not os.path.exists(curi):
            raise FileNotFoundError(curi)

    if sample_random_state is not None:
        random.seed(sample_random_state)

    if max_chars_per_instance is None:
        max_chars_per_instance = -1

    for curi in corpus_uris:
        print_prefix: str = f"({curi}) {'Read' if source_sample_size is None else 'Sampled'} "

        if os.path.isdir(curi):
            furis = glob.glob(os.path.join(curi, f"**/*.{file_ext}"), recursive=True)
            furis = sample_items_(furis, source_sample_size=source_sample_size)

            print(f"{print_prefix}{len(furis)} files with '.{file_ext}' extension.")

            cur_texts: list[str] = []
            for furi in furis:
                with open(furi, "r", encoding="utf-8") as f_in:
                    cur_texts.append(f_in.read(max_chars_per_instance))

        elif curi.endswith(".txt"):
            with open(curi, "r", encoding="utf-8") as f_in:
                cur_texts = [f_in.read(max_chars_per_instance)]

            print(f"({curi}) Read '.txt' file.")

        else:
            cur_texts = (
                pd.read_csv(curi, usecols=[keep_index], sep=sep, index_col=False).squeeze().tolist()
            )
            cur_texts = sample_items_(cur_texts, source_sample_size=source_sample_size)
            cur_texts = [str(item) for item in cur_texts if item]

            if max_chars_per_instance > 0:
                cur_texts = [item[:max_chars_per_instance] for item in cur_texts]

            print(f"{print_prefix}{len(cur_texts)} non-empty items from corpus.")

        texts.extend(cur_texts)

    print(f"Number of sources: {len(corpus_uris)}")
    print(f"Number of files  : {len(texts)}")

    if not texts:
        raise ValueError("No text has been loaded.")

    return texts


def cluster_embs(
    embs: npt.NDArray[np.float64],
    dbscan_kwargs: dict[str, t.Any],
    n_clusters: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if n_clusters is None:
        clusterer = sklearn.cluster.DBSCAN(**dbscan_kwargs)
        cluster_ids: npt.NDArray[np.int64] = clusterer.fit_predict(embs)
        clusters = np.unique(cluster_ids)

    else:
        eps_min = 0.0
        eps_max = float(np.max(np.ptp(embs, axis=0)) * 0.50 * np.sqrt(2))

        dbscan_kwargs_copy = dbscan_kwargs.copy()

        for i in range(1, 1 + 50):
            if eps_max - eps_min <= 1e-6:
                break

            eps_mid = 0.50 * (eps_min + eps_max)
            dbscan_kwargs_copy["eps"] = eps_mid
            clusterer = sklearn.cluster.DBSCAN(**dbscan_kwargs_copy)
            cluster_ids = clusterer.fit_predict(embs)
            clusters = np.unique(cluster_ids)

            if clusters.size == n_clusters:
                print(
                    f"{colorama.Fore.BLUE}"
                    f"Found the requested number of clusters (#{i} optimization iteration): "
                    f"eps={eps_mid:.4f} (you can avoid reoptimization by using "
                    f"--dbscan-eps={eps_mid:.4f} in future runs).{colorama.Style.RESET_ALL}"
                )
                break

            elif clusters.size > n_clusters:
                eps_min = eps_mid
            else:
                eps_max = eps_mid

    clusters = clusters[clusters >= 0]

    n_clusters = clusters.size + 1
    print("Number of clusters (includes noise 'cluster' if present):", n_clusters)

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


def build_cache_uri(
    resource_name: str, args, /, *, additional_info: t.Sequence[str] | None = None
) -> str:
    corpus_urns = [os.path.basename(item.rstrip("/")) for item in args.corpus_uris]
    corpus_urns.sort()

    hasher = hashlib.sha256()
    for urn in corpus_urns:
        hasher.update(urn.encode())

    if additional_info:
        for item in additional_info:
            hasher.update(item.encode())

    corpus_urns_sha256 = hasher.hexdigest()

    sbert_urn = os.path.basename(args.sbert_uri.rstrip("/"))

    return os.path.join(args.cache_dir, f"{resource_name}_{sbert_urn}_{corpus_urns_sha256}.npy")


def compute_very_common_tokens(
    texts,
    spacy_model_name: str,
    cutoff: int | float,
    minimum_length: int,
    banned_tokens: set[str] | None,
    enable_plural_stemmer: bool,
    do_not_stem_keywords: set[str],
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
        lemmas: list[str] = [token.lemma_ for token in fn_spacy_model(doc)]

        if enable_plural_stemmer:
            lemmas = [
                rslp_s.stem(item) if item not in do_not_stem_keywords else item for item in lemmas
            ]

        token_freqs.update(lemmas)

    token_freqs_clean = collections.Counter(
        {
            k: v
            for k, v in token_freqs.items()
            if len(k) >= minimum_length
            and k not in STOPWORDS
            and k not in banned_tokens
            and k not in string.punctuation
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
    enable_plural_stemmer: bool,
    do_not_stem_keywords: set[str],
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
            enable_plural_stemmer=enable_plural_stemmer,
            do_not_stem_keywords=do_not_stem_keywords,
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
        palette=args.scatterplot_palette if cluster_ids is not None else None,
        linewidth=0,
    )

    if remove_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

    sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)


def build_rotation_matrix(degrees: float) -> npt.NDArray[np.float64]:
    radians = degrees / 360 * 2 * np.pi
    cos, sin = float(np.cos(radians)), float(np.sin(radians))
    rot_matrix = np.array([[cos, -sin], [sin, cos]], dtype=float)
    return rot_matrix


def rotate_embeddings(
    embs: npt.NDArray[np.float64],
    medoids: npt.NDArray[np.float64],
    acceptable_unbalance: float = 0.05,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    degrees_cand = np.arange(0, 90 + 1, 5)
    unbalances = np.full(degrees_cand.size, fill_value=np.inf, dtype=float)

    def fn(
        embs: npt.NDArray[np.float64], medoids: npt.NDArray[np.float64], degrees: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        rot_mat = build_rotation_matrix(degrees)

        rot_embs = embs @ rot_mat
        rot_medoids = medoids @ rot_mat

        rot_embs_median = np.median(rot_embs, axis=0)

        rot_embs -= rot_embs_median
        rot_medoids -= rot_embs_median

        return rot_embs, rot_medoids

    for i, degrees in enumerate(degrees_cand):
        _, rot_medoids = fn(embs=embs, medoids=medoids, degrees=degrees)
        unbalances[i] = right_side_prop = float(np.mean(rot_medoids[:, 0] >= 0.0))
        is_acceptable = abs(right_side_prop - 0.50) <= acceptable_unbalance
        if is_acceptable:
            break

    unbalances = np.abs(unbalances - 0.50)
    best_angle = degrees_cand[np.argmin(unbalances)]

    if best_angle > 0:
        embs, medoids = fn(embs=embs, medoids=medoids, degrees=best_angle)
        print(f"Rotated embeddings {best_angle} degrees to optimize keyword box placement.")

    return embs, medoids


def plot_keywords(
    ax,
    medoids: npt.NDArray[np.float64],
    cluster_keywords: dict[int, list[str]],
    args,
) -> None:
    mid_x = 0

    textcoord_x = medoids[:, 0] >= mid_x
    textcoord_y = np.empty(len(medoids), dtype=float)

    for is_after_midway_x in [False, True]:
        cur_inds = np.flatnonzero(textcoord_x == is_after_midway_x)
        coord_ids = np.argsort(np.argsort(medoids[cur_inds, 1]))
        frac_space = np.linspace(-0.025, 0.975, len(cur_inds))
        textcoord_y[cur_inds] = frac_space[coord_ids]

    min_, max_ = -args.keyword_inflate_prop, (1.0 + args.keyword_inflate_prop)
    textcoord_x = textcoord_x.astype(float) * (max_ - min_) + min_

    xytextcoords = np.vstack((textcoord_x, textcoord_y), dtype=float).T

    arrowprops = {
        "arrowstyle": "->",
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

    mid_y, max_y = np.quantile(ax.get_ylim(), (0.5, 1.0))

    for i, (medoid, xytext) in enumerate(zip(medoids, xytextcoords)):
        cx, cy = medoid

        cur_arrowprops = arrowprops.copy()
        curvature = args.arrow_rad * (cy - mid_y) / (max_y - mid_y)
        if cx <= mid_x:
            curvature *= -1.0
        cur_arrowprops["connectionstyle"] = f"arc3,rad={curvature}"

        ax.annotate(
            args.keyword_sep.join(cluster_keywords[i]),
            xy=tuple(medoid),
            xytext=tuple(xytext),
            fontsize=args.keyword_font_size,
            horizontalalignment="right" if cx >= mid_x else "left",
            arrowprops=cur_arrowprops,
            bbox=bbox,
            xycoords="data",
            textcoords="axes fraction",
        )


def plot(args) -> None:
    texts = read_corpus(
        corpus_uris=args.corpus_uris,
        sep=args.corpus_file_sep,
        keep_index=args.corpus_file_col_index,
        file_ext=args.corpus_dir_ext,
        source_sample_size=args.source_sample_size,
        sample_random_state=args.sample_random_state,
        max_chars_per_instance=args.max_chars_per_instance,
    )

    embs_uri = build_cache_uri("embs", args)

    very_common_tokens_uri = build_cache_uri(
        "very_common_tokens",
        args,
        additional_info=[
            f"vctc_{args.very_common_tokens_cutoff:.4f}",
            f"eps_{args.enable_plural_stemmer}",
            f"smn_{args.spacy_model_name}",
            f"dnskw_{args.do_not_stem_keywords}",
        ],
    )

    if not args.ignore_cache and os.path.exists(embs_uri):
        embs = np.load(embs_uri)
        print(f"{colorama.Fore.YELLOW}Found cached '{embs_uri}'.{colorama.Style.RESET_ALL}")

    else:
        sbert = sentence_transformers.SentenceTransformer(args.sbert_uri, device=args.sbert_device)
        projector = umap.UMAP(n_components=2, metric="cosine", random_state=args.umap_random_state)

        embs = sbert.encode(
            texts,
            batch_size=args.sbert_batch_size,
            show_progress_bar=not args.disable_progress_bar,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embs = projector.fit_transform(embs)

        np.save(embs_uri, embs)

    do_not_stem_keywords: str[str] = (
        {item.strip() for item in args.do_not_stem_keywords.split(",")}
        if args.do_not_stem_keywords
        else set()
    )

    if not args.ignore_cache and os.path.exists(very_common_tokens_uri):
        with open(very_common_tokens_uri, "r", encoding="utf-8") as f_in:
            very_common_tokens = set(f_in.read().split("\n"))

        print(
            f"{colorama.Fore.YELLOW}Found cached '{very_common_tokens_uri}'."
            f"{colorama.Style.RESET_ALL}"
        )

    else:
        very_common_tokens = set(
            compute_very_common_tokens(
                texts=texts,
                spacy_model_name=args.spacy_model_name,
                cutoff=args.very_common_tokens_cutoff,
                disable_progress_bar=args.disable_progress_bar,
                enable_plural_stemmer=args.enable_plural_stemmer,
                do_not_stem_keywords=do_not_stem_keywords,
                banned_tokens=None,
                minimum_length=1,
            )
        )

        with open(very_common_tokens_uri, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(sorted(very_common_tokens)))

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))
    embs = scaler.fit_transform(embs)
    embs -= np.median(embs, axis=0)

    inches_to_pixels = 1.0 / float(plt.rcParams["figure.dpi"])
    fig, ax = plt.subplots(
        1,
        figsize=(args.fig_width * inches_to_pixels, args.fig_height * inches_to_pixels),
        layout="tight",
    )

    try:
        cluster_ids, medoids = cluster_embs(
            embs,
            dbscan_kwargs={
                "eps": args.dbscan_eps,
                "min_samples": args.dbscan_min_samples,
            },
            n_clusters=args.n_clusters,
        )

    except ValueError as err:
        plot_base(fig=fig, ax=ax, embs=embs, cluster_ids=None, args=args, remove_labels=False)
        plt.show()
        raise ValueError from err

    embs, medoids = rotate_embeddings(embs=embs, medoids=medoids)

    plot_base(
        fig=fig,
        ax=ax,
        embs=embs,
        cluster_ids=cluster_ids,
        args=args,
        remove_labels=True,
    )

    user_banned_tokens: str[str] = (
        {item.strip() for item in args.banned_keywords.split(",")}
        if args.banned_keywords
        else set()
    )

    if user_banned_tokens:
        print(f"Found {len(user_banned_tokens)} custom banned tokens.")

    cluster_keywords = compute_cluster_keywords(
        texts=texts,
        spacy_model_name=args.spacy_model_name,
        cluster_ids=cluster_ids,
        keywords_per_cluster=args.keywords_per_cluster,
        banned_tokens=very_common_tokens | user_banned_tokens,
        minimum_length=args.keyword_minimum_length,
        enable_plural_stemmer=args.enable_plural_stemmer,
        do_not_stem_keywords=do_not_stem_keywords,
        disable_progress_bar=args.disable_progress_bar,
    )

    plot_keywords(
        ax=ax,
        medoids=medoids,
        cluster_keywords=cluster_keywords,
        args=args,
    )

    if not args.do_not_show:
        plt.show()

    if args.output:
        fig.savefig(args.output, bbox_inches=0)
        print(f"Saved plot as '{args.output}'.")
