import glob
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# from scipy.stats import entropy
# from scipy.special import softmax

parser = argparse.ArgumentParser()
parser.add_argument("--formats", nargs="+", required=True)
parser.add_argument("--labels", nargs="+", required=True)
parser.add_argument("--values", nargs="+", type=float, required=True)
parser.add_argument("--sig-lags", nargs="+", type=float, required=True)
parser.add_argument("--keys", nargs="+", required=True)
parser.add_argument("--sid", type=int, default=625)
parser.add_argument("--emb", type=str, default="glove")
parser.add_argument("--sig-elec-file", nargs="+", default="")
parser.add_argument("--outfile", default="data/tfs-sig-file-")
parser.add_argument("--sig-percents", nargs="+", type=float, required=True)

args = parser.parse_args()

assert (
    len(args.sig_elec_file) == 2 or len(args.sig_elec_file) == 0
), "Need exactly 0 or 2 significant files"
# assert len(args.formats) == 1
assert str(args.sid) in args.formats[0]
if len(args.sig_elec_file) == 2:
    assert (
        str(args.sid) in args.sig_elec_file[0]
        and "comp" in args.sig_elec_file[0]
    ), "Make sure to use the right sig elec files"
    assert (
        str(args.sid) in args.sig_elec_file[1]
        and "prod" in args.sig_elec_file[1]
    ), "Make sure to use the right sig elec files"


####################################################################################
print("Aggregating data")
####################################################################################

data = []
for fmt, label in zip(args.formats, args.labels):
    for key in args.keys:
        fname = fmt % key
        files = glob.glob(fname)
        assert len(files) > 0, f"No results found under {fname}"
        for resultfn in files:
            elec = os.path.basename(resultfn).replace(".csv", "")[:-5]
            # Skip electrodes if they're not part of the sig list
            df = pd.read_csv(resultfn, header=None)
            df.insert(0, "mode", key)
            df.insert(0, "electrode", elec)
            df.insert(0, "label", label)
            data.append(df)
if not len(data):
    print("No data found")
    exit(1)

df = pd.concat(data)
df.set_index(["label", "electrode", "mode"], inplace=True)
assert len(args.values) == len(df.columns), "need correct lag values"


####################################################################################
print("Writing to Sig File")
####################################################################################

chosen_lag_idx = [
    idx for idx, element in enumerate(args.values) if (element in args.sig_lags)
]
df_short = df.loc[:, chosen_lag_idx]  # chose from sig lags


metric = "range"
metric = "maxmin"
metric = "max"
metric = "diff"

if metric == "range":
    df["sort_metric"] = df_short.max(axis=1) - df.min(axis=1)  # row range max
elif metric == "max":
    df["sort_metric"] = df_short.max(axis=1)  # row max
elif metric == "maxmin":
    df["max"] = df_short.max(axis=1)
    df["min"] = df.min(axis=1)
elif metric == "diff":
    x_vals = [args.values[lag] / 1000 for lag in chosen_lag_idx]
    df_short["area"] = np.trapz(df_short, x=x_vals, axis=1)  # integration
    df_short.loc[df_short.area < 0, "area"] = 0  # turn negative area to 0

    # calculate area difference
    df_save = df_short.groupby(level=["electrode", "mode"]).diff().dropna()
    df_save = df_save.reset_index(level="label", drop=True)

    # calculate area max
    df_save2 = df_short.groupby(level=["electrode", "mode"]).agg(
        {"area": ["max"]}
    )
    df_save2.columns = df_save2.columns.droplevel(0)

    glove = False
    if glove:
        # calculate correlation max and choose only >=0.08
        pass
        # df_save3 = df.groupby(level=["electrode", "mode"]).max()
        # df_save3["max_cor"] = df_save3.max(axis=1)
        # df_save3 = df_save3[df_save3["max_cor"] >= 0.08]
    else:
        pass
        # choose only >= 0.08 based on glove
        # df_save3 = pd.read_csv(
        #     f"results/brainplot/area_-500_-100_glove/{args.sid}_area_glove.csv"
        # )
        # df_save3.set_index(["electrode", "mode"], inplace=True)
        # df_save3 = df_save3.rename(columns={"area_diff": "max_cor"})

    # normalize area difference
    df_save = pd.concat([df_save, df_save2], axis=1)
    # df_save = pd.concat([df_save, df_save3.loc[:, "max_cor"]], axis=1).dropna()
    df_save["area_diff"] = df_save["area"] * -1 / df_save["max"]

    df_save.loc[:, "area_diff"].to_csv(
        f"results/brainplot/{args.sid}_area_{args.emb}.csv"
    )
    breakpoint()


def gaussmix(data, mode, metric):
    print(args.sid, mode, metric)
    data = data[data.index.isin([mode], level="mode")]
    if metric == "maxmin":
        sample = data.loc[:, ("max", "min")].to_numpy(dtype=object)
    else:
        sample = data.sort_metric.to_numpy(dtype=object).reshape(-1, 1)

    gausclass = GaussianMixture(n_components=2, random_state=0).fit_predict(
        sample
    )
    data = data.assign(gausclass=gausclass)

    if metric == "maxmin":
        sns_plot = sns.scatterplot(data=data, x="max", y="min", hue="gausclass")
    else:
        sns_plot = sns.histplot(
            data=data, x="sort_metric", bins=50, kde=True, hue="gausclass"
        )
    sns_plot.figure.savefig(str(args.sid) + "_" + metric + "_" + mode + ".png")

    # pdf = PdfPages('sig_test.pdf')
    # fig, ax = plt.subplots(figsize=(15,6))
    # breakpoint()
    # ax.hist(sample[gausclass==1].tolist(), color='blue')
    # ax.hist(sample[gausclass==0].tolist(), color='red')
    # pdf.savefig(fig)
    # plt.close()

    return None


# gaussmix(df, mode, metric)


def save_sig_file(data, mode, sig_file=""):
    no_depth = True
    if no_depth:
        surface_rows = [
            (label, elec, mode)
            for (label, elec, mode) in df.index
            if "D" not in elec
        ]
        data = data.loc[surface_rows, :]
    filetype = "-top-"
    if sig_file != "":
        sig_elecs = pd.read_csv("data/" + sig_file)[
            "electrode"
        ].tolist()  # load significant elecs
        data = data[
            data.index.isin(sig_elecs, level="electrode")
        ]  # only keep significant elecs
        filetype = "-sig-"
    data = data[data.index.isin([mode], level="mode")]  # only keep prod or comp
    df_partial = data.sort_values(
        by=["sort_metric"], ascending=False
    )  # sort by metric
    filetype = "-top-0.08-"
    df_partial = df_partial[df_partial.sort_metric >= 0.08]
    file_name = args.outfile + str(args.sid) + filetype + mode + ".csv"
    sig_elecs = df_partial.index.get_level_values("electrode")
    sig_elec_df = {"subject": args.sid, "electrode": sig_elecs}
    sig_elec_df = pd.DataFrame(sig_elec_df)
    sig_elec_df.to_csv(file_name, index=False)

    # for sig_percent in args.sig_percents:
    #     sig_num = int(sig_percent * len(df_partial.index))
    #     file_name = (
    #         args.outfile
    #         + str(args.sid)
    #         + filetype
    #         + str(sig_percent)
    #         + "-"
    #         + mode
    #         + ".csv"
    #     )
    #     sig_elecs = df_partial.index.get_level_values("electrode")[0:sig_num]
    #     sig_elec_df = {"subject": args.sid, "electrode": sig_elecs}
    #     sig_elec_df = pd.DataFrame(sig_elec_df)
    #     sig_elec_df.to_csv(file_name, index=False)


if len(args.sig_elec_file) == 2:
    save_sig_file(df, "prod", args.sig_elec_file[0])
    save_sig_file(df, "comp", args.sig_elec_file[1])
else:
    save_sig_file(df, "prod")
    save_sig_file(df, "comp")
