import pandas as pd


def main():

    elec_list = [
        "G101",
        "G77",
        "G17",
        "G78",
        "G88",
        "G95",
        "G16",
        "G20",
        "G18",
        "G22",
        "G90",
        "G12",
        "G19",
        "G86",
        "G63",
        "AF6",
        "G93",
        "G113",
        "G56",
        "G92",
        "G97",
        "G64",
        "G79",
        "AIT3",
        "G83",
        "G81",
        "G87",
        "O6",
        "G84",
        "DPI1",
        "G11",
        "AIT2",
        "DPI3",
        "G96",
        "G6",
        "G72",
        "O7",
        "DPI2",
        "SF4",
        "G76",
        "O3",
        "O5",
        "IF1",
        "G82",
        "P6",
        "G27",
        "O4",
        "G66",
        "DAMT7",
        "DPI4",
        "G125",
        "G70",
        "G117",
        "G55",
        "G34",
        "G107",
        "G33",
        "G127",
        "G47",
    ]

    sid = "798"
    mode = "comp"
    emb = "gpt"

    df = pd.DataFrame({"subject": sid, "electrode": elec_list})
    filename = f"data/tfs-sig-file-{sid}-{emb}-{mode}.csv"
    df.to_csv(filename, index=False)

    return


if __name__ == "__main__":
    main()
