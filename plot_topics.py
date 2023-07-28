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
        source_sample_factor if source_sample_factor >= 1.0 else (source_sample_factor * len(items))
    )
    n = max(1, n)
    return items[:n]


def read_corpus(
    corpus_uris: list[str],
    sep: str,
    file_ext: str,
    keep_index: int,
    source_sample_factor: int | float | None,
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
        print_prefix: str = f"({curi}) {'Read' if source_sample_factor is None else 'Sampled'} "

        if os.path.isdir(curi):
            furis = glob.glob(os.path.join(curi, f"**/*.{file_ext}"), recursive=True)
            furis = sample_items_(furis, source_sample_factor=source_sample_factor)

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
            cur_texts = sample_items_(cur_texts, source_sample_factor=source_sample_factor)
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
        eps_min, eps_max = 0.0, 0.71
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

    for i, degrees in enumerate(degrees_cand):
        rot_mat = build_rotation_matrix(degrees)
        rot_medoids = medoids @ rot_mat
        unbalances[i] = right_side_prop = float(np.mean(rot_medoids[:, 0] >= 0.0))
        is_acceptable = abs(right_side_prop - 0.50) <= acceptable_unbalance
        if is_acceptable:
            break

    unbalances = np.abs(unbalances - 0.50)
    best_angle = degrees_cand[np.argmin(unbalances)]

    if best_angle > 0:
        rot_mat = build_rotation_matrix(best_angle)
        embs = embs @ rot_mat
        medoids = medoids @ rot_mat
        print(f"Rotated embeddings {best_angle} degrees to optimize keyword box placement.")

    return embs, medoids


def plot_keywords(
    ax,
    medoids: npt.NDArray[np.float64],
    cluster_keywords: dict[int, list[str]],
    args,
) -> None:
    textcoord_x = medoids[:, 0] >= 0.5 * float(np.add(*ax.get_xlim()))
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
        "connectionstyle": f"arc3,rad={args.arrow_rad}",
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
            cur_arrowprops["connectionstyle"] = f"arc3,rad=-{args.arrow_rad}"

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


def run(args) -> None:
    texts = read_corpus(
        corpus_uris=args.corpus_uris,
        sep=args.corpus_file_sep,
        keep_index=args.corpus_file_col_index,
        file_ext=args.corpus_dir_ext,
        source_sample_factor=args.source_sample_factor,
        sample_random_state=args.sample_random_state,
        max_chars_per_instance=args.max_chars_per_instance,
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
            texts,
            batch_size=args.sbert_batch_size,
            show_progress_bar=not args.disable_progress_bar,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embs = projector.fit_transform(embs)

        np.save(embs_uri, embs)

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
                banned_tokens=None,
                minimum_length=1,
            )
        )

        with open(very_common_tokens_uri, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(sorted(very_common_tokens)))

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))
    embs = scaler.fit_transform(embs)

    inches_to_pixels = 1.0 / plt.rcParams["figure.dpi"]
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

    user_banned_tokens = set(args.banned_keywords.split(",")) if args.banned_keywords else set()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "sbert_uri", help="Path to the Sentence Transformer used to embed documents."
    )
    parser.add_argument(
        "corpus_uris",
        action="extend",
        nargs="+",
        type=str,
        help="One or more paths to copora files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.pdf",
        help="Output file URI. Setting to blank disables saving the output file.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Directory to cache embeddings and very common words.",
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="If set, force recomputation of any cached value.",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="If set, disable all progress bars.",
    )

    parser_plot = parser.add_argument_group("plot arguments")
    parser_plot.add_argument(
        "--scatterplot-palette",
        default="hsv",
        help="Matplotlib pallete color for scatter plot. See 'https://matplotlib.org/stable/tutorials/colors/colormaps.html'.",
    )
    parser_plot.add_argument(
        "--kdeplot-thresh",
        default=1e-3,
        type=float,
        help=(
            "KDE plot threshold to set the lowest contour plot level drawn. "
            "See 'https://seaborn.pydata.org/generated/seaborn.kdeplot.html'."
        ),
    )
    parser_plot.add_argument(
        "--kdeplot-levels",
        default=12,
        type=int,
        help="Set the number of contour plot levels.",
    )
    parser_plot.add_argument(
        "--kdeplot-alpha",
        default=1.0,
        type=float,
        help="Set the transparency level to contour plot. Must be in [0, 1] range.",
    )
    parser_plot.add_argument(
        "--kdeplot-cmap",
        default="viridis",
        help="Matplotlib color map for contour plot. See 'https://matplotlib.org/stable/tutorials/colors/colormaps.html'.",
    )
    parser_plot.add_argument(
        "--arrow-color",
        default="black",
        help="Color for arrows connecting cluster and their keyword boxes.",
    )
    parser_plot.add_argument(
        "--arrow-alpha",
        default=0.80,
        type=float,
        help="Transparency level for cluster-keyword arrows. Must be in [0, 1] range.",
    )
    parser_plot.add_argument(
        "--arrow-linestyle",
        default="dashed",
        help="Line style for cluster-keyword arrows. See 'https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html'.",
    )
    parser_plot.add_argument(
        "--arrow-linewidth",
        default=2.0,
        type=float,
        help="Line width for cluster-keyword arrows.",
    )
    parser_plot.add_argument(
        "--arrow-rad",
        default=0.20,
        type=float,
        help="Set cluster-keyword arrow curvature. If 0.0, connections will be straight lines.",
    )
    parser_plot.add_argument(
        "--keyword-inflate-prop",
        default=0.05,
        type=float,
        help="Horizontal figure proportion for which keyword boxes are placed outside the plot axis.",
    )
    parser_plot.add_argument(
        "--fig-width",
        default=1024,
        type=float,
        help="Set plot figure width. Unit is pixels.",
    )
    parser_plot.add_argument(
        "--fig-height",
        default=768,
        type=float,
        help="Set plot figure height. Unit is pixels.",
    )
    parser_plot.add_argument(
        "--background-color",
        default="white",
        type=str,
        help="Set plot background color.",
    )
    parser_plot.add_argument(
        "--do-not-show", action="store_true", help="If set, disable output display."
    )

    parser_dbscan = parser.add_argument_group("dbscan arguments")
    parser_dbscan.add_argument(
        "--dbscan-eps",
        default=0.05,
        type=float,
        help="DBSCAN radius size. See 'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html'.",
    )
    parser_dbscan.add_argument(
        "--dbscan-min-samples",
        default=10,
        type=int,
        help=(
            "DBSCAN minimum number of connected points for core points. "
            "See 'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html'."
        ),
    )
    parser_dbscan.add_argument(
        "--n-clusters",
        default=None,
        type=int,
        help=(
            "If provided, try to match the specified number of clusters by optimizing "
            "DBSCAN's eps hyperparameter using binary search. In this case, the value "
            "of parameter 'dbscan-eps' is ignored."
        ),
    )

    parser_keywords = parser.add_argument_group("keywords arguments")
    parser_keywords.add_argument(
        "--keywords-per-cluster",
        default=3,
        type=int,
        help="Number of keywords per cluster.",
    )
    parser_keywords.add_argument(
        "--keyword-sep",
        default="\n",
        type=str,
        help="String used to separate keywords of each cluster.",
    )
    parser_keywords.add_argument(
        "--banned-keywords",
        default=None,
        type=str,
        help="List of banned keywords. To provide multiple words, separate by ',' (comma).",
    )
    parser_keywords.add_argument(
        "--keyword-minimum-length",
        default=3,
        type=int,
        help="Minimum length (in characters) of keyword candidates.",
    )
    parser_keywords.add_argument(
        "--keyword-font-size",
        default=10,
        type=int,
        help="Font size for displaying keywords.",
    )
    parser_keywords.add_argument(
        "--very-common-tokens-cutoff",
        default=0.01,
        type=float,
        help=(
            "Proportion of words considered 'very common', which are disallowed to become "
            "cluster keywords. Must be in [0, 1] range."
        ),
    )
    parser_keywords.add_argument(
        "--spacy-model-name",
        default="pt_core_news_sm",
        help="Spacy model name to apply lemmatization.",
    )

    parser_sbert = parser.add_argument_group("sbert arguments")
    parser_sbert.add_argument(
        "--sbert-device",
        default=None,
        type=str,
        help="Device to embed documents using SBERT. If not provided, will use GPU if possible.",
    )
    parser_sbert.add_argument(
        "--sbert-batch-size",
        default=128,
        type=int,
        help="SBERT batch size for document embedding.",
    )

    parser_umap = parser.add_argument_group("umap arguments")
    parser_umap.add_argument(
        "--umap-random-state", default=182783, type=int, help="Random seed for UMAP."
    )

    parser_corpus = parser.add_argument_group("corpus arguments")
    parser_corpus.add_argument(
        "--max-chars-per-instance",
        default=None,
        type=int,
        help="If set, truncate every instance up to the specified amount of characters.",
    )
    parser_corpus.add_argument(
        "--corpus-file-sep",
        default=",",
        help="Column separator when reading corpus from a single file.",
    )
    parser_corpus.add_argument(
        "--corpus-file-col-index",
        default=0,
        type=int,
        help="Column index to keep when reading corpus from a single file.",
    )
    parser_corpus.add_argument(
        "--corpus-dir-ext",
        default="txt",
        help="File extension to find files when reading corpus from a directory hierarchy.",
    )
    parser_corpus.add_argument(
        "--source-sample-factor",
        default=None,
        type=float,
        help=(
            "If provided, set the sample factor for each corpus source. "
            "Each source is sampled independently. For instance, if you provide N "
            "data sources with a sample factor of M, you'll end up with at most N * M samples. "
            "Can be either a float in range [0, 1) (sample by proportion of total instances), "
            "or an integer >= 1 (sets sample maximum size)."
        ),
    )
    parser_corpus.add_argument(
        "--sample-random-state",
        default=8101192,
        type=int,
        help="Random state for source instance sampling.",
    )

    parser_seaborn = parser.add_argument_group("seaborn arguments")
    parser_seaborn.add_argument(
        "--seaborn-context",
        default="paper",
        help=(
            "Set seaborn plot context. "
            "See 'https://seaborn.pydata.org/generated/seaborn.set_context.html'."
        ),
    )
    parser_seaborn.add_argument(
        "--seaborn-style",
        default="whitegrid",
        help=(
            "Set seaborn plot style. "
            "See 'https://seaborn.pydata.org/generated/seaborn.set_style.html'."
        ),
    )
    parser_seaborn.add_argument(
        "--seaborn-font-scale",
        default=1.0,
        type=float,
        help="Set seaborn font scaling factor.",
    )

    args = parser.parse_args()

    args.cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        spacy.load(args.spacy_model_name)

    except OSError:
        import spacy.cli  # pylint: disable="ungrouped-imports"

        spacy.cli.download(args.spacy_model_name)

    sns.set_theme(
        context=args.seaborn_context,
        style=args.seaborn_style,
        palette="colorblind",
        font_scale=args.seaborn_font_scale,
    )

    sns.set(rc={"axes.facecolor": args.background_color})

    run(args)
