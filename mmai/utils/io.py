import pandas as pd

# mmai/utils/io.py
def write_ped_from_df(df, path):
    with open(path, "w") as f:
        for pid in df.index:
            line = [pid]*2 + ["0", "0", "1", "-9"]
            snp_values = df.loc[pid].apply(lambda x: "0 0" if pd.isna(x) else
                                                       "0 0" if x == 0 else
                                                       "0 1" if x == 1 else
                                                       "1 1" if x == 2 else
                                                       "0 0")
            f.write(" ".join(line + snp_values.tolist()) + "\n")

def write_vcf_from_df(df, bim_df, path):
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=SynthWGS\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        samples = df.index.tolist()
        header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
        f.write("\t".join(header) + "\n")
        for snp in df.columns:
            row = bim_df[bim_df.snp == snp]
            if row.empty:
                continue
            row = row.iloc[0]
            chrom, pos, ref, alt = row.chrom, row.pos, row.a1, row.a2
            calls = []
            for val in df[snp]:
                if pd.isna(val): calls.append("./.")
                elif val == 0: calls.append("0/0")
                elif val == 1: calls.append("0/1")
                elif val == 2: calls.append("1/1")
                else: calls.append("./.")
            f.write("\t".join(map(str, [chrom, pos, snp, ref, alt, ".", ".", ".", "GT"] + calls)) + "\n")
