import argparse

import numpy as np
from sklearn.decomposition import PCA
from tfsenc_read_datum import read_datum
from tfsenc_utils import setup_environ


def run_pca(args, df):
    pca = PCA(n_components=args.pca_to, svd_solver='auto', whiten=True)

    df_emb = df['embeddings']
    embs = np.vstack(df_emb.values)

    pca_output = pca.fit_transform(embs)
    df['embeddings'] = pca_output.tolist()

    return df


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', nargs='?', type=int, default=None)
    parser.add_argument('--emb-type', type=str, default=None)
    parser.add_argument('--pca-to', type=int, default=1)
    parser.add_argument('--context-length', type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    args = setup_environ(args)

    # file_name, file_ext = os.path.splitext(args.emb_file)
    # pca_output_file_name = file_name + '-pca' + str(args.reduce_to) + file_ext

    df = read_datum(args)
    df = run_pca(args, df)


if __name__ == "__main__":
    main()
