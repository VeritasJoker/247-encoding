from distutils.command.config import dump_file
import glob
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy import stats


def read_sig_file(filename, old_results=False):

    sig_file = pd.read_csv("data/" + filename)
    sig_file["sid_electrode"] = (
        sig_file["subject"].astype(str) + "_" + sig_file["electrode"]
    )
    elecs = sig_file["sid_electrode"].tolist()

    if old_results:  # might need to use this for old 625-676 results
        elecs = sig_file["electrode"].tolist()  # no sid name in front

    return set(elecs)


def read_file(file_name, sigelecs, sigelecs_key, load_sid, label, mode, type):

    elec = os.path.basename(file_name).replace(".csv", "")[:-5]
    if (
        len(sigelecs)
        and elec not in sigelecs[sigelecs_key]
        and "whole_brain" not in sigelecs_key
    ):
        return None
    # Skip electrodes if they're not part of the sig list
    # if 'LGA' not in elec and 'LGB' not in elec: # for 717, only grid
    #     continue
    df = pd.read_csv(file_name, header=None)
    df.insert(0, "sid", load_sid)
    df.insert(0, "mode", mode)
    df.insert(0, "electrode", elec)
    df.insert(0, "label", label)
    df.insert(0, "type", type)

    return df


def read_folder(
    data,
    fname,
    sigelecs,
    sigelecs_key,
    load_sid="load_sid",
    label="label",
    mode="mode",
    type="all",
    parallel=False,
):

    files = glob.glob(fname)
    assert (
        len(files) > 0
    ), f"No results found under {fname}"  # check files exist under format

    if parallel:
        p = Pool(10)
        for result in p.map(
            partial(
                read_file,
                sigelecs=sigelecs,
                sigelecs_key=sigelecs_key,
                load_sid=load_sid,
                label=label,
                mode=mode,
                type=type,
            ),
            files,
        ):
            data.append(result)

    else:
        for resultfn in files:
            data.append(
                read_file(
                    resultfn,
                    sigelecs,
                    sigelecs_key,
                    load_sid,
                    label,
                    mode,
                    type,
                )
            )

    return data
