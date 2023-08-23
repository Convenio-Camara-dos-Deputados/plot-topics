import argparse
import os

import seaborn as sns
import spacy

import plot_topics


def expand_path(path: str) -> str:
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    path = os.path.abspath(path)
    return path


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
        help=(
            "Matplotlib color map for contour plot. "
            "See 'https://matplotlib.org/stable/tutorials/colors/colormaps.html'."
        ),
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
        help=(
            "Line style for cluster-keyword arrows. See "
            "'https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html'."
        ),
    )
    parser_plot.add_argument(
        "--arrow-linewidth",
        default=2.0,
        type=float,
        help="Line width for cluster-keyword arrows.",
    )
    parser_plot.add_argument(
        "--arrow-rad",
        default=0.30,
        type=float,
        help=(
            "Set cluster-keyword arrow maximum curvature. "
            "If 0.0, connections will be straight lines."
        ),
    )
    parser_plot.add_argument(
        "--keyword-inflate-prop",
        default=0.05,
        type=float,
        help=(
            "Horizontal figure proportion for which keyword boxes are placed outside the plot axis."
        ),
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
        help=(
            "DBSCAN radius size. See "
            "'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html'."
        ),
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
        help=(
            "List of banned keywords. To provide multiple words, separate each word by ',' (comma)."
        ),
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
    parser_keywords.add_argument(
        "--enable-plural-stemmer",
        action="store_true",
        help=(
            "Enable RSLP-S stemmer to remove plural form from keywords. Note that this stemmer "
            "does not work well with proper nouns. Therefore, you may want to use with alongside "
            "the --do-not-stem-keywords argument."
        ),
    )
    parser_keywords.add_argument(
        "--do-not-stem-keywords",
        default=None,
        type=str,
        help=(
            "List of keywords not to apply RSLP-S stemmer. "
            "This argument only takes effect when --enable-plural-stemmer is provided. "
            "To provide multiple words, separate each word by ',' (comma)."
        ),
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
        "--source-sample-size",
        default=None,
        type=float,
        help=(
            "If provided, set the sample size for each corpus source. "
            "Each source is sampled independently. For instance, if you provide N "
            "data sources with a sample size of M, you'll end up with at most N * M samples. "
            "Can be either a float in range [0, 1) (sample proportionally source size), "
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

    args.cache_dir = expand_path(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)

    args.sbert_uri = expand_path(args.sbert_uri)
    args.corpus_uris = [expand_path(item) for item in args.corpus_uris]

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

    plot_topics.plot(args)
