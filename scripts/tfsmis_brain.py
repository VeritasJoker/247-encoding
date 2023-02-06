import glob
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pickle


sub = 798

keys = ["prod", "comp"]

emb = "glove"

corr = "ave"

saving_elec_file = False  # DO NOT TURN THIS ON OR YOU DIE

# formats = [
#     'results/tfs/kw-tfs-full-' + sub + '-glove50-triple/kw-200ms-all-' + sub + '/',
#     'results/tfs/kw-tfs-full-' + sub + '-gpt2-xl-triple/kw-200ms-all-' + sub + '/',
#     'results/tfs/kw-tfs-full-' + sub + '-blenderbot-small-triple/kw-200ms-all-' + sub + '/'
#     ]
formats = [
    f"results/tfs/stock/kw-tfs-full-{sub}-erp-lag2k-25-all-test/*/",
]

formats = [
    f"results/tfs/stock-1024-24/kw-tfs-full-{sub}-gpt2-xl-lag10k-25-all-shift-emb/*/",
    f"results/tfs/20230131-whisper-encoder-onset/kw-tfs-full-en-onset-{sub}-whisper-tiny.en-l4-wn2-6/*/",
]

comp_sig_file = f"data/tfs-sig-file-{sub}-sig-1.0-comp.csv"
prod_sig_file = f"data/tfs-sig-file-{sub}-sig-1.0-prod.csv"

coordinatefilename = f"data/brainplot/{sub}_{corr}.txt"
elecfilename = f"data/brainplot/{sub}_elecs.csv"

#################################################################################################
###################################### Compare two sources ######################################
#################################################################################################

data = pd.read_csv(coordinatefilename, sep=" ", header=None)
data = data.set_index(0)
data = data.loc[:, 1:4]
print(f"\nFor subject {sub}:\ntxt has {len(data.index)} electrodes")

if saving_elec_file:
    print("YOU DIED")
    breakpoint()
    files = glob.glob(formats[0] + "*_prod.csv")
    files = [os.path.basename(file) for file in files]
    print(f"encoding has {len(files)} electrodes")

    files = pd.DataFrame(data=files)
    files["elec"] = ""
    files["elec2"] = ""
    for row, values in files.iterrows():
        elec = os.path.basename(values[0]).replace(".csv", "")[:-5]
        files.loc[row, "elec"] = elec
        # elec = elec[:-3] # just for 625 and 676
        elec = elec.replace("EEG", "").replace("GR_", "G").replace("_", "")
        files.loc[row, "elec2"] = elec
    files = files.set_index(0)
    files = files.sort_values(by="elec2")
    # files.to_csv(elecfilename, index=False)

    breakpoint()
    # go to the file and fix everything

    elecs = pd.read_csv(elecfilename)
    set1 = set(data.index)
    set2 = set(files.elec2)
    set3 = set(elecs.dropna(subset="elec2").elec2)
    set4 = set(elecs.dropna(subset="elec").elec)

    breakpoint()
    print(f"txt and encoding share {len(set1.intersection(set2))} electrodes\n")
    print(f"encoding does not have these electrodes: {sorted(set1-set2)}\n")
    print(f"txt does not have these electrodes: {sorted(set2-set1)}\n")


#############################################################################################
###################################### Getting Results ######################################
#############################################################################################

elecs = pd.read_csv(elecfilename)
elecs = elecs.dropna()
elecs = elecs.rename(columns={"elec2": 0})
elecs.set_index(0, inplace=True)

df = pd.merge(data, elecs, left_index=True, right_index=True)
print(f"Now subject has {len(df)} electrodes")

# df["comp"] = 100
# df["prod"] = 100
df["comp_sig"] = 0
df["prod_sig"] = 0

comp_sig_elecs = pd.read_csv(comp_sig_file)["electrode"].tolist()
prod_sig_elecs = pd.read_csv(prod_sig_file)["electrode"].tolist()


################# GET area for a curve #################
# area = pd.read_csv(f"results/brainplot/{sub}_area_{emb}.csv")
# area = area.drop(["label"], errors="ignore")
# area["elec_name"] = area.electrode.str[len(str(sub)) + 1 :]
# area.set_index("elec_name", inplace=True)

# area_prod = area.loc[area["mode"] == "prod", "area_diff"]
# area_comp = area.loc[area["mode"] == "comp", "area_diff"]

# for row, values in df.iterrows():
#     if values["elec"] in comp_sig_elecs:
#         df.loc[row, "comp_sig"] = 1
#     if values["elec"] in prod_sig_elecs:
#         df.loc[row, "prod_sig"] = 1
#     try:
#         df.loc[row, "comp"] = area_comp[values["elec"]]
#     except:
#         print(row, values["elec"])
#     try:
#         df.loc[row, "prod"] = area_prod[values["elec"]]
#     except:
#         print(row, values["elec"])

# df["comp"] = df["comp"].fillna(10)
# df["prod"] = df["prod"].fillna(10)


# def save_area_results(sub, df, outname, mode, sig=False):
#     df = df.loc[df[mode] < 100, :]
#     sig_string = ""
#     if sig:
#         sig_string = "_sig"
#         df = df.loc[df[mode + sig_string] == 1, :]  # choose only sig electrodes
#     outname = f"{outname}{sub}_{emb}_{mode}{sig_string}.txt"

#     with open(outname, "w") as outfile:
#         df = df.loc[:, [1, 2, 3, 4, mode]]
#         df.to_string(outfile)


# outname = "results/brainplot/"
# if corr == "ind":
#     outname = outname + "ind/"

# for key in keys:
#     save_area_results(sub, df, outname, key)
#     save_area_results(sub, df, outname, key, True)


################# GET ERP correlation between prod/comp #################
# def get_erp_corr(compfile, prodfile, path):

#     filename = os.path.join(path, compfile)
#     comp_data = pd.read_csv(filename, header=None)
#     filename = os.path.join(path, prodfile)
#     prod_data = pd.read_csv(filename, header=None)
#     corr_erp, _ = pearsonr(comp_data.loc[0, :], prod_data.loc[0, :])

#     return corr_erp


# df["erp"] = -1

# for format in formats:
#     for row, values in df.iterrows():
#         if row in prod_sig_elecs or row in comp_sig_elecs:
#             prod_name = values[0]
#             comp_name = values[0].replace("prod", "comp")
#             df.loc[row, "erp"] = get_erp_corr(comp_name, prod_name, format)

# output_filename = "results/cor_tfs/" + sub + "_" + corr + "_erp_sig" + ".txt"
# with open(output_filename, "w") as outfile:
#     df = df.loc[:, [1, 2, 3, 4, "erp"]]
#     df.to_string(outfile)
# breakpoint()
#####################################################################################


################################ GET max correlation #################################
embs = ["gptn", "whisper_en"]

emb_key = [emb + "_" + key for emb in embs for key in keys]
for col in emb_key:
    df[col] = -1


def get_max(filename, path):
    filename = os.path.join(path, filename)
    if len(glob.glob(filename)) == 1:
        filename = glob.glob(filename)[0]
    elif len(glob.glob(filename)) == 0:
        return -1
    else:
        AssertionError("huh this shouldn't happen")
    elec_data = pd.read_csv(filename, header=None)
    return max(elec_data.loc[0])


for format in formats:
    if "glove50" in format:
        col_name = "glove"
    elif "gpt2-xl" in format:
        col_name = "gptn"
    elif "blenderbot-small" in format:
        col_name = "bbot"
    elif "whisper-encoder" in format:
        col_name = "whisper_en"
    elif "whisper-decoder" in format:
        col_name = "whisper_de"
    elif "whisper-for-grant" in format:
        col_name = "whisper_full"
    print(f"getting results for {col_name} embedding")
    for row, values in df.iterrows():
        col_name1 = col_name + "_prod"
        col_name2 = col_name + "_comp"
        prod_name = f"{sub}_{values['elec']}_prod.csv"
        comp_name = f"{sub}_{values['elec']}_comp.csv"
        if values["elec"] in prod_sig_elecs:
            df.loc[row, "prod_sig"] = 1
        df.loc[row, col_name1] = get_max(prod_name, format)
        if values["elec"] in comp_sig_elecs:
            df.loc[row, "comp_sig"] = 1
        df.loc[row, col_name2] = get_max(comp_name, format)

for col in emb_key:
    output_filename = f"results/cor_tfs-20220202/{sub}_{corr}_{col}.txt"
    output_filename2 = f"results/cor_tfs-20220202/tfs_{corr}_{col}.txt"
    elecs_sig = False
    if not elecs_sig:
        df_output = df.loc[:, [1, 2, 3, 4, col]]
    else:
        if "_comp" in col:
            df_output = df.loc[
                df.comp_sig == 1, [1, 2, 3, 4, col]
            ]  # choose only sig electrodes
        elif "_prod" in col:
            df_output = df.loc[
                df.prod_sig == 1, [1, 2, 3, 4, col]
            ]  # choose only sig electrodes
    with open(output_filename, "w") as outfile:
        df_output.to_string(outfile)
    with open(output_filename2, "a") as outfile:
        df_output.to_string(outfile)
