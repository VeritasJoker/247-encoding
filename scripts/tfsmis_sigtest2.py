import glob
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pickle


subs = [625, 676, 7170, 798]

keys = ["prod", "comp"]

corr = "ave"


#################################################################################################
###################################### Compare two sources ######################################
#################################################################################################

for sub in subs:
    elecfilename = f"data/brainplot/{sub}_elecs.csv"
    for key in keys:
        filename = f"results/brainplot/area_-500_-100_n/{sub}_{key}_selected.txt"

        data = pd.read_fwf(filename, header=None)
        data = data.set_index(0)
        data = data.loc[:, 5]

        elecs = pd.read_csv(elecfilename)
        elecs = elecs.dropna()
        elecs = elecs.rename(columns={"elec2": 0})
        elecs.set_index(0, inplace=True)

        df = pd.merge(data, elecs, left_index=True, right_index=True)
        df = df.rename(columns={5: "area", "elec": "electrode"})
        df["subject"] = str(sub)

        df_pos = df[df.area > 0].sort_values(by=["area"], ascending=False)
        df_neg = df[df.area < 0].sort_values(by=["area"], ascending=True)

        df_pos.loc[:, ["subject", "electrode", "area"]].to_csv(
            f"data/tfs-sig-{sub}-n-area+_{key}.csv", index=False
        )
        df_neg.loc[:, ["subject", "electrode", "area"]].to_csv(
            f"data/tfs-sig-{sub}-n-area-_{key}.csv", index=False
        )
        df.loc[:, ["subject", "electrode", "area"]].to_csv(
            f"data/tfs-sig-{sub}-n-area_{key}.csv", index=False
        )
