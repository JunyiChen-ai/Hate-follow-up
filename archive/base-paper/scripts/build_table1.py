"""Build Table 1 LaTeX with correct bold/underline assignments per column."""

DATA = {
    # name: {(ds, metric): value}
    "BERT":       {("HM","ACC"):69.1, ("HM","F1"):63.7, ("HM","P"):70.1, ("HM","R"):64.0,
                   ("EN","ACC"):65.5, ("EN","F1"):49.1, ("EN","P"):55.2, ("EN","R"):52.2,
                   ("ZH","ACC"):72.5, ("ZH","F1"):67.7, ("ZH","P"):68.4, ("ZH","R"):62.8,
                   ("IH","ACC"):75.2, ("IH","F1"):75.2, ("IH","P"):75.3, ("IH","R"):75.2},
    "ViViT":      {("HM","ACC"):68.2, ("HM","F1"):66.7, ("HM","P"):66.8, ("HM","R"):66.6,
                   ("EN","ACC"):61.4, ("EN","F1"):61.4, ("EN","P"):62.2, ("EN","R"):61.1,
                   ("ZH","ACC"):71.0, ("ZH","F1"):66.1, ("ZH","P"):66.6, ("ZH","R"):65.8,
                   ("IH","ACC"):76.5, ("IH","F1"):76.5, ("IH","P"):76.7, ("IH","R"):76.5},
    "MFCC":       {("HM","ACC"):65.4, ("HM","F1"):60.3, ("HM","P"):64.1, ("HM","R"):60.7,
                   ("EN","ACC"):46.5, ("EN","F1"):47.2, ("EN","P"):58.8, ("EN","R"):52.2,
                   ("ZH","ACC"):63.1, ("ZH","F1"):52.5, ("ZH","P"):54.1, ("ZH","R"):53.0,
                   ("IH","ACC"):70.0, ("IH","F1"):70.0, ("IH","P"):70.1, ("IH","R"):70.0},
    "Pro-Cap":    {("HM","ACC"):64.5, ("HM","F1"):63.3, ("HM","P"):63.4, ("HM","R"):63.2,
                   ("EN","ACC"):70.1, ("EN","F1"):66.3, ("EN","P"):66.3, ("EN","R"):66.3,
                   ("ZH","ACC"):72.5, ("ZH","F1"):66.8, ("ZH","P"):66.1, ("ZH","R"):68.3,
                   ("IH","ACC"):82.3, ("IH","F1"):82.3, ("IH","P"):82.4, ("IH","R"):82.3},
    "HateMM":     {("HM","ACC"):76.0, ("HM","F1"):72.8, ("HM","P"):77.9, ("HM","R"):72.0,
                   ("EN","ACC"):71.5, ("EN","F1"):63.2, ("EN","P"):68.3, ("EN","R"):62.6,
                   ("ZH","ACC"):71.0, ("ZH","F1"):61.8, ("ZH","P"):66.5, ("ZH","R"):61.4,
                   ("IH","ACC"):76.6, ("IH","F1"):76.5, ("IH","P"):76.8, ("IH","R"):76.6},
    "MHCL":       {("HM","ACC"):77.4, ("HM","F1"):76.5, ("HM","P"):77.5, ("HM","R"):76.6,
                   ("EN","ACC"):65.4, ("EN","F1"):65.4, ("EN","P"):67.2, ("EN","R"):64.9,
                   ("ZH","ACC"):76.5, ("ZH","F1"):73.1, ("ZH","P"):73.2, ("ZH","R"):73.0,
                   ("IH","ACC"):85.2, ("IH","F1"):85.2, ("IH","P"):85.4, ("IH","R"):85.2},
    "MoRE":       {("HM","ACC"):83.4, ("HM","F1"):82.4, ("HM","P"):81.8, ("HM","R"):83.3,
                   ("EN","ACC"):77.5, ("EN","F1"):75.2, ("EN","P"):75.7, ("EN","R"):74.8,
                   ("ZH","ACC"):78.5, ("ZH","F1"):74.8, ("ZH","P"):75.7, ("ZH","R"):74.1,
                   ("IH","ACC"):84.8, ("IH","F1"):84.7, ("IH","P"):85.4, ("IH","R"):84.8},
    "ImpliHateVid":{("HM","ACC"):82.8, ("HM","F1"):82.0, ("HM","P"):82.1, ("HM","R"):82.0,
                   ("EN","ACC"):76.1, ("EN","F1"):71.7, ("EN","P"):71.6, ("EN","R"):71.8,
                   ("ZH","ACC"):78.3, ("ZH","F1"):72.8, ("ZH","P"):74.4, ("ZH","R"):71.8,
                   ("IH","ACC"):87.5, ("IH","F1"):87.5, ("IH","P"):87.6, ("IH","R"):87.5},
    "CMFusion":   {("HM","ACC"):75.6, ("HM","F1"):74.1, ("HM","P"):75.3, ("HM","R"):73.7,
                   ("EN","ACC"):72.2, ("EN","F1"):68.8, ("EN","P"):69.5, ("EN","R"):68.2,
                   ("ZH","ACC"):73.8, ("ZH","F1"):70.1, ("ZH","P"):71.0, ("ZH","R"):69.4,
                   ("IH","ACC"):68.5, ("IH","F1"):68.4, ("IH","P"):68.9, ("IH","R"):68.6},
    "MiniCPM-V":  {("HM","ACC"):72.4, ("HM","F1"):72.3, ("HM","P"):77.8, ("HM","R"):76.4,
                   ("EN","ACC"):69.1, ("EN","F1"):67.4, ("EN","P"):69.3, ("EN","R"):67.4,
                   ("ZH","ACC"):73.5, ("ZH","F1"):73.2, ("ZH","P"):72.0, ("ZH","R"):73.0,
                   ("IH","ACC"):77.5, ("IH","F1"):77.5, ("IH","P"):77.6, ("IH","R"):77.5},
    "LLaVA-OV":   {("HM","ACC"):75.6, ("HM","F1"):75.6, ("HM","P"):77.9, ("HM","R"):78.3,
                   ("EN","ACC"):73.7, ("EN","F1"):70.5, ("EN","P"):70.5, ("EN","R"):66.8,
                   ("ZH","ACC"):75.2, ("ZH","F1"):70.2, ("ZH","P"):71.4, ("ZH","R"):70.3,
                   ("IH","ACC"):79.0, ("IH","F1"):79.2, ("IH","P"):79.0, ("IH","R"):79.0},
    "Qwen2-VL":   {("HM","ACC"):73.7, ("HM","F1"):73.7, ("HM","P"):78.1, ("HM","R"):77.3,
                   ("EN","ACC"):70.5, ("EN","F1"):66.8, ("EN","P"):66.8, ("EN","R"):66.7,
                   ("ZH","ACC"):73.3, ("ZH","F1"):74.0, ("ZH","P"):73.9, ("ZH","R"):72.9,
                   ("IH","ACC"):73.4, ("IH","F1"):74.9, ("IH","P"):73.5, ("IH","R"):73.4},
    "Naive-Qwen3-2B": {("HM","ACC"):67.4, ("HM","F1"):58.7, ("HM","P"):71.3, ("HM","R"):60.7,
                   ("EN","ACC"):71.4, ("EN","F1"):52.9, ("EN","P"):67.8, ("EN","R"):55.4,
                   ("ZH","ACC"):74.5, ("ZH","F1"):58.2, ("ZH","P"):77.9, ("ZH","R"):59.0,
                   ("IH","ACC"):55.6, ("IH","F1"):45.7, ("IH","P"):69.9, ("IH","R"):55.5},
    "Naive-InternVL3-8B": {("HM","ACC"):79.5, ("HM","F1"):77.0, ("HM","P"):81.6, ("HM","R"):76.0,
                   ("EN","ACC"):72.0, ("EN","F1"):53.3, ("EN","P"):71.1, ("EN","R"):55.8,
                   ("ZH","ACC"):71.8, ("ZH","F1"):49.5, ("ZH","P"):75.8, ("ZH","R"):54.0,
                   ("IH","ACC"):58.1, ("IH","F1"):49.6, ("IH","P"):74.5, ("IH","R"):58.0},
    "MARS":       {("HM","ACC"):69.8, ("HM","F1"):69.6, ("HM","P"):75.9, ("HM","R"):74.0,
                   ("EN","ACC"):65.8, ("EN","F1"):62.8, ("EN","P"):62.7, ("EN","R"):64.5,
                   ("ZH","ACC"):73.8, ("ZH","F1"):46.8, ("ZH","P"):46.7, ("ZH","R"):47.0,
                   ("IH","ACC"):83.3, ("IH","F1"):83.3, ("IH","P"):83.4, ("IH","R"):83.3},
    "Mod-HATE":   {("HM","ACC"):77.7, ("HM","F1"):76.3, ("HM","P"):77.0, ("HM","R"):76.0,
                   ("EN","ACC"):70.2, ("EN","F1"):59.4, ("EN","P"):63.0, ("EN","R"):59.1,
                   ("ZH","ACC"):38.3, ("ZH","F1"):38.2, ("ZH","P"):46.2, ("ZH","R"):46.3,
                   ("IH","ACC"):71.0, ("IH","F1"):69.8, ("IH","P"):74.8, ("IH","R"):70.9},
    "LoReHM":     {("HM","ACC"):79.1, ("HM","F1"):78.9, ("HM","P"):79.4, ("HM","R"):80.6,
                   ("EN","ACC"):76.4, ("EN","F1"):67.3, ("EN","P"):74.0, ("EN","R"):65.8,
                   ("ZH","ACC"):74.5, ("ZH","F1"):62.2, ("ZH","P"):71.9, ("ZH","R"):61.6,
                   ("IH","ACC"):87.0, ("IH","F1"):86.9, ("IH","P"):87.6, ("IH","R"):87.0},
    "ALARM":      {("HM","ACC"):79.5, ("HM","F1"):79.4, ("HM","P"):79.8, ("HM","R"):81.0,
                   ("EN","ACC"):69.6, ("EN","F1"):60.2, ("EN","P"):62.4, ("EN","R"):59.8,
                   ("ZH","ACC"):71.1, ("ZH","F1"):47.5, ("ZH","P"):73.0, ("ZH","R"):52.9,
                   ("IH","ACC"):71.8, ("IH","F1"):69.7, ("IH","P"):79.2, ("IH","R"):71.6},
    "TRIAGE":     {("HM","ACC"):84.2, ("HM","F1"):83.2, ("HM","P"):84.0, ("HM","R"):82.8,
                   ("EN","ACC"):78.3, ("EN","F1"):71.1, ("EN","P"):75.9, ("EN","R"):69.5,
                   ("ZH","ACC"):82.6, ("ZH","F1"):80.6, ("ZH","P"):79.5, ("ZH","R"):83.1,
                   ("IH","ACC"):83.3, ("IH","F1"):83.2, ("IH","P"):83.7, ("IH","R"):83.3},
}

ORDER = ["BERT","ViViT","MFCC","Pro-Cap","HateMM","MHCL","MoRE","ImpliHateVid","CMFusion",
         "MiniCPM-V","LLaVA-OV","Qwen2-VL","Naive-Qwen3-2B","Naive-InternVL3-8B","MARS","Mod-HATE","LoReHM","ALARM","TRIAGE"]

DS_LIST = ["HM","EN","ZH","IH"]
METRIC_LIST = ["ACC","F1","P","R"]

# Compute best/second per column
rank = {}
for ds in DS_LIST:
    for m in METRIC_LIST:
        col = [(name, DATA[name][(ds,m)]) for name in ORDER]
        col_sorted = sorted(col, key=lambda x: -x[1])
        best_val = col_sorted[0][1]
        second_val = None
        for name, v in col_sorted[1:]:
            if v < best_val:
                second_val = v
                break
        rank[(ds,m)] = (best_val, second_val)

def fmt(name, ds, m):
    v = DATA[name][(ds,m)]
    best, second = rank[(ds,m)]
    if v == best:
        return f"\\textbf{{{v:.1f}}}"
    if second is not None and v == second:
        return f"\\underline{{{v:.1f}}}"
    return f"{v:.1f}"

def row(name):
    disp = {"TRIAGE": r"\textbf{\ours{}}",
            "Naive-Qwen3-2B": r"Naive Qwen3-VL-2B",
            "Naive-InternVL3-8B": r"Naive InternVL3-8B",
            "Mod-HATE": r"Mod-HATE (8-shot)",
            "HateMM": r"HateMM-det",
            "ImpliHateVid": r"ImpliHateVid-det"}.get(name, name)
    cells = []
    for ds in DS_LIST:
        for m in METRIC_LIST:
            cells.append(fmt(name, ds, m))
    return disp + " & " + " & ".join(cells) + r" \\"

# Print table
print(r"\begin{table*}[t]")
print(r"\centering")
print(r"\scriptsize")
print(r"\setlength{\tabcolsep}{3pt}")
print(r"\caption{Comparison with state-of-the-art methods on four hateful video detection datasets. Best results in \textbf{bold}, second best \underline{underlined}. All numbers in \%.}")
print(r"\label{tab:main}")
print(r"\begin{tabular}{@{}l cccc cccc cccc cccc@{}}")
print(r"\toprule")
print(r"& \multicolumn{4}{c}{\textbf{HateMM}} & \multicolumn{4}{c}{\textbf{MHClip-EN}} & \multicolumn{4}{c}{\textbf{MHClip-ZH}} & \multicolumn{4}{c}{\textbf{ImpliHateVid}} \\")
print(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17}")
print(r"\textbf{Method} & ACC & M-F1 & M-P & M-R & ACC & M-F1 & M-P & M-R & ACC & M-F1 & M-P & M-R & ACC & M-F1 & M-P & M-R \\")
print(r"\midrule")
print(r"\rowcolor{gray!8}\multicolumn{17}{l}{\emph{Unimodal Methods}} \\")
for n in ["BERT","ViViT","MFCC"]: print(row(n))
print(r"\midrule")
print(r"\rowcolor{gray!8}\multicolumn{17}{l}{\emph{Multimodal Fusion Methods (supervised)}} \\")
for n in ["Pro-Cap","HateMM","MHCL","MoRE","ImpliHateVid","CMFusion"]: print(row(n))
print(r"\midrule")
print(r"\rowcolor{gray!8}\multicolumn{17}{l}{\emph{MLLM-based Methods (label-free / few-shot)}} \\")
for n in ["MiniCPM-V","LLaVA-OV","Qwen2-VL","Naive-Qwen3-2B","Naive-InternVL3-8B","MARS","Mod-HATE","LoReHM","ALARM"]: print(row(n))
print(r"\midrule")
print(r"\rowcolor{blue!6}" + row("TRIAGE"))
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")
