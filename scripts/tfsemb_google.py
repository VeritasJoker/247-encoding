import numpy as np
import pandas as pd
import os
from tfsenc_read_datum import load_datum


def main():

    sid = 798
    # base_filepath = (
    #     f"data/tfs/{sid}/pickles/embeddings/whisper-tiny.en/full-en-onset/base_df.pkl"
    # )
    # emb_filepath = f"data/tfs/{sid}/pickles/embeddings/whisper-tiny.en/full-en-onset/cnxt_0001/layer_04.pkl"

    # base_df = load_datum(base_filepath)
    # emb_df = load_datum(emb_filepath)
    # df = pd.concat([base_df, emb_df], axis=1)
    # breakpoint()

    # embs = emb_df.embeddings.tolist()
    # embs = np.array(embs)
    # np.save("emb.npy", embs, allow_pickle=False)

    df = np.load("layer_04.npy", allow_pickle=False)

    breakpoint()
    return


if __name__ == "__main__":
    main()
