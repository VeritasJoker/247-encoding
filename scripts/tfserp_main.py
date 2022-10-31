import csv
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tfsenc_load_signal import load_electrode_data
from tfsenc_main import process_subjects, return_stitch_index
from tfsenc_parser import parse_arguments
from tfsenc_read_datum import read_datum
from tfsenc_utils import write_encoding_results
from tfsenc_config import setup_environ
from utils import main_timer, write_config


def erp(args, datum, elec_signal, name):
    datum = datum[datum.adjusted_onset.notna()]

    datum_comp = datum[datum.speaker != "Speaker1"]  # comprehension data
    datum_prod = datum[datum.speaker == "Speaker1"]  # production data
    print(
        f"{args.sid} {name} Prod: {len(datum_comp.index)} Comp: {len(datum_prod.index)}"
    )

    erp_comp = calc_average(
        args.lags, datum_comp, elec_signal
    )  # calculate average erp
    erp_prod = calc_average(
        args.lags, datum_prod, elec_signal
    )  # calculate average erp

    print(f"writing output for electrode {name}")
    write_encoding_results(args, erp_prod, name, "prod")
    write_encoding_results(args, erp_comp, name, "comp")

    return


def calc_average(lags, datum, brain_signal):
    """[summary]
    Args:
        args ([type]): [description]
        datum ([type]): [description]
        brain_signal ([type]): [description]

    Returns:
        [type]: [description]
    """
    onsets = datum.adjusted_onset.values
    erp = np.zeros((len(onsets), len(lags)))

    for lag_idx, lag in enumerate(lags):  # loop through each lag
        lag_amount = int(lag / 1000 * 512)
        index_onsets = (
            np.round_(onsets, 0, onsets) + lag_amount
        )  # take correct idx for all words
        index_onsets = index_onsets.astype(
            int
        )  # uncomment this if not running jit
        erp[:, lag_idx] = brain_signal[index_onsets].reshape(
            -1
        )  # take the signal for that lag

    erp = [
        np.mean(erp, axis=(0), dtype=np.float64).tolist()
    ]  # average by words

    return erp


def load_and_erp(electrode, args, datum, stitch_index):

    (sid, elec_id), elec_name = electrode  # get electrode info

    # load electrode signal (with z_score)
    elec_signal, missing_convos = load_electrode_data(
        args, sid, elec_id, stitch_index, True
    )
    elec_signal = elec_signal.reshape(-1, 1)

    # trim datum based on signal
    if len(missing_convos) > 0:  # signal missing convos
        elec_datum = datum.loc[
            ~datum["conversation_name"].isin(missing_convos)
        ]  # filter missing convos
    else:
        elec_datum = datum

    # special cases for missing signal
    if len(elec_datum) == 0:  # no signal
        print(f"{args.sid} {elec_name} No Signal")
        return None

    # do and save erp
    erp(args, elec_datum, elec_signal, elec_name)

    return


def load_and_erp_parallel(
    args, electrode_info, datum, stitch_index, parallel=True
):
    if parallel:
        print("Running all electrodes in parallel")
        with Pool(8) as p:
            p.map(
                partial(
                    load_and_erp,
                    args=args,
                    datum=datum,
                    stitch_index=stitch_index,
                ),
                electrode_info.items(),
            )
    else:
        for electrode in electrode_info.items():
            load_and_erp(electrode, args, datum, stitch_index)


@main_timer
def main():

    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Saving configuration to output directory
    write_config(vars(args))

    # Locate and read datum
    stitch_index = return_stitch_index(args)  # load stitch index
    datum = read_datum(args, stitch_index)
    datum = datum.drop("embeddings", 1)  # trim datum to smaller size

    assert args.sig_elec_file == None, "Do not input significant electrode list"
    electrode_info = process_subjects(args)
    load_and_erp_parallel(args, electrode_info, datum, stitch_index, False)

    return


if __name__ == "__main__":
    main()
