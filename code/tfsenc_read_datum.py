from multiprocessing.spawn import is_forking
import os
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from utils import load_pickle

# import gensim.downloader as api
# import re

NONWORDS = {"hm", "huh", "mhm", "mm", "oh", "uh", "uhuh", "um"}


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding"""
    df["is_nan"] = df["embeddings"].apply(lambda x: np.isnan(x).all())
    df = df[~df["is_nan"]]

    return df


def adjust_onset_offset(args, df, stitch_index):
    """[summary]

    Args:
        args ([type]): [description]
        df ([type]): [description]
        stitch_index ([list]): stitch index

    Returns:
        [type]: [description]
    """
    print("adjusting datum onset and offset")
    stitch_index = stitch_index[:-1]

    df["adjusted_onset"], df["onset"] = df["onset"], np.nan
    df["adjusted_offset"], df["offset"] = df["offset"], np.nan

    for _, conv in enumerate(df.conversation_id.unique()):
        shift = stitch_index[conv - 1]
        df.loc[df.conversation_id == conv, "onset"] = (
            df.loc[df.conversation_id == conv, "adjusted_onset"] - shift
        )
        df.loc[df.conversation_id == conv, "offset"] = (
            df.loc[df.conversation_id == conv, "adjusted_offset"] - shift
        )
    return df


def make_input_from_tokens(token_list):
    """[summary]

    Args:
        args ([type]): [description]
        token_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    windows = [tuple(token_list[x : x + 2]) for x in range(len(token_list) - 2 + 1)]

    return windows


def add_convo_onset_offset(args, df, stitch_index):
    """Add conversation onset and offset to datum

    Args:
        args (namespace): commandline arguments
        df (DataFrame): datum being processed
        stitch_index ([list]): stitch_index

    Returns:
        Dataframe: df with conversation onset and offset
    """
    windows = make_input_from_tokens(stitch_index)

    df["convo_onset"], df["convo_offset"] = np.nan, np.nan

    for _, conv in enumerate(df.conversation_id.unique()):
        edges = windows[conv - 1]

        df.loc[df.conversation_id == conv, "convo_onset"] = edges[0]
        df.loc[df.conversation_id == conv, "convo_offset"] = edges[1]

    return df


def add_signal_length(df, stitch):
    """Add conversation signal length to datum

    Args:
        df (DataFrame): datum being processed
        stitch (List): stitch index

    Returns:
        DataFrame: df with conversation signal length
    """
    signal_lengths = np.diff(stitch).tolist()

    df["conv_signal_length"] = np.nan

    for idx, conv in enumerate(df.conversation_id.unique()):
        df.loc[df.conversation_id == conv, "conv_signal_length"] = signal_lengths[idx]

    return df


def normalize_embeddings(args, df):
    """Normalize the embeddings
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html

    Args:
        args ([type]): [description]
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    k = np.array(df.embeddings.tolist())

    try:
        k = normalize(k, norm=args.normalize, axis=1)
    except ValueError:
        df["embeddings"] = k.tolist()

    return df


def clean_datum(emb_type, df):

    df.loc[:, "word"] = df.word.str.lower().str.strip(string.punctuation)

    if emb_type == "glove50":
        df = df.dropna(subset=["embeddings"])
    else:
        df = drop_nan_embeddings(df)
        df = remove_punctuation(df)

    nans = df.embeddings.apply(lambda x: np.isnan(x).any())
    notnon = df.is_nonword == 0
    if "glove" not in emb_type:
        same = df.token2word.str.lower().str.strip() == df.word
        df = df[same & ~nans & notnon]
    else:
        df = df[~nans & notnon]
    df.reset_index(drop=True, inplace=True)
    assert not df.adjusted_onset.isna().any()

    return df


def zeroshot_datum(df):

    dfz = (
        df[["word", "adjusted_onset"]]
        .groupby("word")
        .apply(lambda x: x.sample(1, random_state=42))
    )
    dfz.reset_index(level=1, inplace=True)
    dfz.sort_values("adjusted_onset", inplace=True)

    df = df.iloc[dfz.level_1.values]
    print(f"Zeroshot created datum with {len(df.index)} words")

    return df


def onehot_datum(df):

    df2 = zeroshot_datum(df)
    df2 = df2.loc[:, ("word", "embeddings")]
    df2.reset_index(drop=True, inplace=True)

    values = np.arange(0, len(df2.index))
    n_values = len(df2.index)
    onehot_mat = np.eye(n_values)[values]
    df2["embeddings"] = pd.Series(list(onehot_mat))
    df = df.drop("embeddings", axis=1, errors="ignore")

    df = df.merge(df2, how="left", on="word")
    df.sort_values(["conversation_id", "index"], inplace=True)

    print(f"Onehot created datum with {len(df.index)} words")

    return df


def load_datum(file_name):
    """Read raw datum

    Args:
        filename: raw datum full file path

    Returns:
        DataFrame: datum
    """
    datum = load_pickle(file_name)
    df = pd.DataFrame.from_dict(datum)
    return df


def process_datum(args, df, stitch):
    """Process the datum based on input arguments

    Args:
        args (namespace): commandline arguments
        df : raw datum as a DataFrame
        stitch: stitch index

    Raises:
        Exception: args.word_value should be one of ['top', 'bottom', 'all']

    Returns:
        DataFrame: processed datum
    """

    df = add_signal_length(df, stitch)

    df = df.loc[~df["conversation_id"].isin(args.bad_convos)]  # filter bad convos
    assert len(stitch) - len(args.bad_convos) == df.conversation_id.nunique() + 1

    if args.project_id == "tfs" and not all(
        [item in df.columns for item in ["adjusted_onset", "adjusted_offset"]]
    ):
        df = adjust_onset_offset(
            args, df, stitch
        )  # not sure if needed (maybe as a failsafe?)
    # TODO this was needed for podcast but destroys tfs
    # else:
    #     df['adjusted_onset'], df['onset'] = df['onset'], np.nan
    #     df['adjusted_offset'], df['offset'] = df['offset'], np.nan

    df = df[df.adjusted_onset.notna()]
    df = add_convo_onset_offset(args, df, stitch)

    if args.emb_type == "glove50":
        df = df.dropna(subset=["embeddings"])
    else:
        df = drop_nan_embeddings(df)
        df = remove_punctuation(df)
    # df = df[~df['glove50_embeddings'].isna()]

    # if encoding is on glove embeddings copy them into 'embeddings' column
    if args.emb_type == "glove50":
        try:
            df["embeddings"] = df["glove50_embeddings"]
        except KeyError:
            pass

    if not args.normalize:
        df = normalize_embeddings(args, df)

    return df


def filter_datum(args, df):
    """Filter the datum based on embedding types, non_words, min_word_freq arguments

    Args:
        args (namespace): commandline arguments
        df: processed datum

    Returns:
        DataFrame: filtered datum
    """
    if "glove50" in args.emb_type.lower():  # filter based on embedding type argument
        common = df.in_glove
    elif "gpt2" in args.emb_type.lower():
        common = df.in_gpt2
    elif "blenderbot-small" in args.emb_type.lower():
        common = df.in_blenderbot_small_90M
    elif "blenderbot" in args.emb_type.lower():
        common = df.in_blenderbot_3B

    for model in args.align_with:  # filter based on align with arguments
        if "glove" in model:
            common = common & df.in_glove
        elif "gpt2" in model:
            common = common & df.in_gpt2
        elif "blenderbot-small" in model:
            common = common & df.in_blenderbot_small_90M
        elif "blenderbot" in model:
            common = common & df.in_blenderbot_3B

    if args.exclude_nonwords:  # filter based on exclude_nonwords argument
        nonword_mask = df.word.str.lower().apply(lambda x: x in NONWORDS)
        common &= ~nonword_mask

    if args.min_word_freq > 0:  # filter based on min_word_freq argument
        freq_mask = df.word_freq_overall >= args.min_word_freq
        common &= freq_mask

    df = df[common]
    return df


def mod_datum_by_preds(args, datum, emb_type):
    """Filter the datum based on the predictions of a potentially different model

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum
        emb_type: embedding type needed to filter the datum

    Returns:
        DataFrame: further filtered datum
    """
    if emb_type in args.load_emb_file:  # current datum has the correct emb_type
        pass
    else:  # current datum does not have the correct emb_type, need to load a second datum

        # load second datum
        if emb_type == "gpt2-xl":
            second_datum_name = (
                str(args.sid) + "_full_gpt2-xl_cnxt_1024_layer_48_embeddings.pkl"
            )
        elif emb_type == "blenderbot-small":
            second_datum_name = (
                str(args.sid) + "_full_blenderbot-small_layer_16_embeddings.pkl"
            )
        second_datum_path = os.path.join(
            args.PICKLE_DIR, second_datum_name
        )  # second datum full path
        second_datum = load_datum(second_datum_path)[
            ["adjusted_onset", "top1_pred", "top1_pred_prob"]
        ]  # load second datum
        # merge second datum prediction columns to datum
        datum = datum.drop(
            ["top1_pred", "top_1_pred_prob"], axis=1, errors="ignore"
        )  # delete the current top predictions if any
        datum = datum[datum.adjusted_onset.notna()]
        second_datum = second_datum[second_datum.adjusted_onset.notna()]
        datum = datum.merge(second_datum, how="inner", on="adjusted_onset")
    print(f"Using {emb_type} predictions")

    # modify datum based on correct or incorrect predictions
    if "incorrect" in args.datum_mod:  # incorrectly predicted (top 1)
        datum = datum[datum.word.str.lower() != datum.top1_pred.str.lower().str.strip()]
        print(f"Selected {len(datum.index)} incorrect words")
    elif "correct" in args.datum_mod:  # correctly predicted (top 1)
        datum = datum[datum.word.str.lower() == datum.top1_pred.str.lower().str.strip()]
        print(f"Selected {len(datum.index)} correct words")
    elif "top0.5" in args.datum_mod:  # top 30% pred prob
        top = datum.top1_pred_prob.quantile(0.5)
        datum = datum[datum.top1_pred_prob >= top]
        print(f"Selected {len(datum.index)} top pred prob words")
    elif "bot0.5" in args.datum_mod:  # bot 30% pred prob
        bot = datum.top1_pred_prob.quantile(0.5)
        datum = datum[datum.top1_pred_prob <= bot]
        print(f"Selected {len(datum.index)} bot pred prob words")

    # elif args.datum_mod == emb_type + "-pred": # for incorrectly predicted words, replace with top 1 pred (only used for podcast glove)
    #     glove = api.load('glove-wiki-gigaword-50')
    #     datum['embeddings'] = datum.top1_pred.str.strip().apply(lambda x: get_vector(x.lower(), glove))
    #     datum = datum[datum.embeddings.notna()]
    #     print(f'Changed words into {emb_type} top predictions')
    else:  # exception
        raise Exception("Invalid Datum Modification")

    return datum


def shift_emb(args, datum):
    """Shift the embeddings based on datum_mod argument

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum

    Returns:
        DataFrame: datum with shifted embeddings
    """
    partial = args.datum_mod[args.datum_mod.find("shift-emb") + 9]
    if len(partial) == 0:
        partial = "1"
    if partial.find("-") > 0:
        partial = partial[: partial.find("-")]
    else:
        pass
    assert partial.isdigit()
    shift_num = int(partial)

    before_shift_num = len(datum.index)
    for i in np.arange(shift_num):
        datum.loc[:, "embeddings"] = datum.embeddings.shift(-1)
        datum = datum.loc[datum.conversation_id.shift(-1) == datum.conversation_id, :]
        if "blenderbot-small" in args.emb_type.lower():
            datum = datum[datum.speaker.shift(-1) == datum.speaker]
    print(
        f"Shifting {shift_num} times resulted in {before_shift_num - len(datum.index)} less words"
    )
    return datum


def trim_datum(args, datum):
    """Trim the datum based on the largest lag size

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum

    Returns:
        DataFrame: datum with trimmed words
    """
    half_window = round((args.window_size / 1000) * 512 / 2)
    # lag = int(60000 / 1000 * 512) # trim edges with set length
    lag = int(args.lags[-1] / 1000 * 512)  # trim edges based on lag
    original_len = len(datum.index)
    datum = datum.loc[
        ((datum["adjusted_onset"] - lag) >= (datum["convo_onset"] + half_window + 1))
        & ((datum["adjusted_onset"] + lag) <= (datum["convo_offset"] - half_window - 1))
    ]
    new_datum_len = len(datum.index)
    print(
        f"Trimming resulted in {new_datum_len} ({round(new_datum_len/original_len*100,5)}%) words"
    )
    return datum


def mod_datum(args, datum):
    """Filter the datum based on datum_mod argument

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum

    Returns:
        DataFrame: further filtered datum
    """
    if args.conversation_id:  # picking single conversation
        datum = datum[datum.conversation_id == args.conversation_id]
        datum.convo_offset = datum["convo_offset"] - datum["convo_onset"]
        datum.convo_onset = 0

    if "no-trim" in args.datum_mod:  # no need for edge trimming
        pass
    else:
        datum = trim_datum(args, datum)  # trim edges

    if "shift-emb" in args.datum_mod:  # shift embeddings to include word
        datum = shift_emb(args, datum)
    else:
        pass

    if "all" in args.datum_mod:
        pass

    elif "zeroshot" in args.datum_mod:
        datum = clean_datum(args.emb_type, datum)
        datum = zeroshot_datum(datum)

    elif "onehot" in args.datum_mod:
        datum = clean_datum(args.emb_type, datum)
        datum = onehot_datum(datum)

    else:  # modify datum based on predictions
        pred_type = args.emb_type
        if "gpt2-xl" in args.datum_mod:
            pred_type = "gpt2-xl"
        elif "bbot" in args.datum_mod:
            pred_type = "blenderbot-small"
        assert "glove" not in pred_type, "Glove embeddings does not have predictions"
        datum = mod_datum_by_preds(args, datum, pred_type)

    # else:
    #     raise Exception('Invalid Datum Modification')
    assert len(datum.index) > 0, "Empty Datum"
    return datum


def read_datum(args, stitch):
    """Load, process, and filter datum

    Args:
        args (namespace): commandline arguments
        stitch (list): stitch_index

    Returns:
        DataFrame: processed and filtered datum
    """
    file_name = os.path.join(args.PICKLE_DIR, args.load_emb_file)
    df = load_datum(file_name)
    print(f"After loading: Datum loads with {len(df)} words")

    df = process_datum(args, df, stitch)
    print(f"After processing: Datum now has {len(df)} words")
    df = filter_datum(args, df)
    print(f"After filtering: Datum now has {len(df)} words")

    df = mod_datum(args, df)  # further filter datum based on datum_mod argument

    return df
