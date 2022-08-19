import os
import string

import numpy as np
import pandas as pd
import string
from scipy.spatial import distance
from scipy import stats
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wordfreq import word_frequency

from utils import main_timer
from tfsenc_read_datum import load_datum, clean_datum


def run_pca(df):
    pca = PCA(n_components=50, svd_solver="auto", whiten=True)

    df_emb = df["embeddings"]
    embs = np.vstack(df_emb.values)

    pca_output = pca.fit_transform(embs)
    df["embeddings"] = pca_output.tolist()

    return df


def get_dist(arr1, arr2):
    return distance.cosine(arr1, arr2)


def get_freq(word):
    return word_frequency(word, "en")


def plot_dist(df, col1, col2):
    fig, ax = plt.subplots(figsize=(15, 6))
    # ax.set(yscale="log")
    ax = sns.scatterplot(data=df, x=col1, y=col2)
    linefit = stats.linregress(df[col1], df[col2])
    ax.plot(
        df[col1],
        df[col1] * linefit.slope + linefit.intercept,
        c="r",
    )
    corr = stats.pearsonr(df[col1], df[col2])
    ax.text(
        0.9,
        ax.get_ylim()[1],
        f"r={corr[0]:3f} p={corr[1]:.3f}",
        color="r",
    )
    plot_name = col1 + "_vs_" + col2
    plt.savefig("results/figures/676_" + plot_name)
    plt.close()


def plot_hist(df, col1):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax = sns.histplot(data=df, x=col1)
    plot_name = col1 + "hist"
    plt.savefig("results/figures/625_" + plot_name)
    plt.close()


@main_timer
def main():

    file_name = "data/tfs/625/pickles/625_full_gpt2-xl_cnxt_1024_layer_48_embeddings.pkl"
    df = load_datum(file_name)
    print(f"After loading: Datum loads with {len(df)} words")
    df = clean_datum("gpt2-xl", df)
    df = run_pca(df)

    datum = df
    datum.loc[:, "embeddings_n"] = datum.embeddings.shift(-1)
    datum.loc[:, "embeddings_n-2"] = datum.embeddings.shift(1)
    datum = datum[
        datum.conversation_id.shift(-1) == datum.conversation_id.shift(1)
    ]

    datum.loc[:, "cos-dist_nn-1"] = datum.apply(
        lambda x: get_dist(x["embeddings_n"], x["embeddings"]), axis=1
    )
    datum.loc[:, "cos-dist_nn-2"] = datum.apply(
        lambda x: get_dist(x["embeddings_n-2"], x["embeddings_n"]), axis=1
    )

    datum.loc[:, "word_freq_en"] = datum.apply(
        lambda x: get_freq(x["word"]), axis=1
    )

    plot_hist(datum, "cos-dist_nn-1")
    plot_hist(datum, "cos-dist_nn-2")
    plot_dist(datum, "cos-dist_nn-1", "top1_pred_prob")
    plot_dist(datum, "cos-dist_nn-2", "top1_pred_prob")
    plot_dist(datum, "cos-dist_nn-1", "word_freq_en")
    plot_dist(datum, "cos-dist_nn-2", "word_freq_en")
    plot_dist(datum, "cos-dist_nn-1", "word_freq_phase")
    plot_dist(datum, "cos-dist_nn-2", "word_freq_phase")

    return


if __name__ == "__main__":
    main()
