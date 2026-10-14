#!/usr/bin/env python
"""Candidate 44 pre-check (rule 10: reads test GT; analysis only, the method is unchanged).

P1 share of multi-speaker videos among within-eligible videos; P2 speaker-identity ceiling against r6_bma on the
multi-speaker subset; P3 which speaker the speech-driven false-high windows belong to; P4 label-free pooling of the
window reads by speaker. Constants and gates are declared in README section 2. CPU only (conda HateVideo).
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
READS = ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl"
DECODED = ROOT / "runs/20260926_twolevel/r6_bma/predictions.jsonl"
TURNS = ROOT / "data/speaker_turns/{ds}/turns.jsonl"
OUT = ROOT / "runs/20261014_m1_speaker_bound/precheck"
DATASETS = ("HateMM", "HateClipSeg")
FPS = 4
MIN_OVERLAP = 1.0   # seconds of a speaker's turns inside a window for the window to belong to that speaker
MIN_SHARE = 0.10    # share of the video's speech time for a speaker to count
MIN_WINDOWS = 2     # windows a speaker must own to count
POS_FRAC = 0.5      # hateful-frame fraction of a positive window; a negative window has none
GATE_P1 = 0.30
GATE_P2 = 0.05
N_PERM = 10        # permutations of the speaker labels for the P2 control


def records(path, ds):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["dataset"] == ds and not r.get("error"):
            out[r["video_id"]] = r
    return out


def gold(ds):
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): np.asarray(y, int) for v, y, s in zip(g["video_ids"], g["y4"], g["split"]) if str(s) == "test"}


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def frames_of(start, end, n):
    a = int(round(start * FPS))
    b = int(round(end * FPS))
    return max(0, min(a, n)), max(0, min(b, n))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def analyse(ds):
    reads, dec, gt = records(READS, ds), records(DECODED, ds), gold(ds)
    turns = {}
    for line in open(str(TURNS).format(ds=ds)):
        r = json.loads(line)
        turns[r["video_id"]] = r
    diar = {"videos": len(turns), "ok": sum(r["status"] == "ok" for r in turns.values()),
            "failed": sum(r["status"] != "ok" for r in turns.values()),
            "n_speakers_hist": {}, "speech_share_median": None}
    shares = []
    for r in turns.values():
        if r["status"] == "ok":
            k = str(r["n_speakers"])
            diar["n_speakers_hist"][k] = diar["n_speakers_hist"].get(k, 0) + 1
            if r.get("audio_seconds"):
                shares.append(r["speech_seconds"] / r["audio_seconds"])
    diar["speech_share_median"] = float(np.median(shares)) if shares else None

    rows = []
    for vid, y in gt.items():
        if vid not in reads or vid not in dec or y.min() == y.max():
            continue
        n = len(y)
        wins = reads[vid]["extra"]["windows"]
        curve = np.asarray(dec[vid]["score_curve"], float)
        if len(curve) != n:
            curve = curve[:n] if len(curve) >= n else np.pad(curve, (0, n - len(curve)), mode="edge")
        tr = turns.get(vid, {"status": "missing", "turns": []})
        spk_time = {}
        for t in tr["turns"]:
            spk_time[t["spk"]] = spk_time.get(t["spk"], 0.0) + (t["e"] - t["s"])
        speech_total = sum(spk_time.values())
        dom = []
        for w in wins:
            ov = {}
            for t in tr["turns"]:
                o = min(t["e"], w["end"]) - max(t["s"], w["start"])
                if o > 0:
                    ov[t["spk"]] = ov.get(t["spk"], 0.0) + o
            if ov:
                k = max(ov, key=ov.get)
                dom.append(k if ov[k] >= MIN_OVERLAP else None)
            else:
                dom.append(None)
        dom_count = {}
        for k in dom:
            if k is not None:
                dom_count[k] = dom_count.get(k, 0) + 1
        valid = sorted(k for k in spk_time
                       if speech_total > 0 and spk_time[k] / speech_total >= MIN_SHARE and dom_count.get(k, 0) >= MIN_WINDOWS)
        multi = len(valid) >= 2
        fr = [frames_of(w["start"], w["end"], n) for w in wins]
        hf = np.array([float(y[a:b].mean()) if b > a else 0.0 for a, b in fr])
        r6w = np.array([float(curve[a:b].mean()) if b > a else np.nan for a, b in fr])
        z = np.array([w["z"] for w in wins], float)

        def expand(ws):
            s = np.full(n, np.nan)
            for (a, b), v in zip(fr, ws):
                s[a:b] = v
            s[np.isnan(s)] = np.nanmean(s)
            return s

        vid_frac = float(y.mean())
        spks = sorted(set(d for d in dom if d is not None))
        h_k, hate_frames = {}, {}
        for k in spks:
            idx = [i for i, d in enumerate(dom) if d == k]
            tot = sum(fr[i][1] - fr[i][0] for i in idx)
            hate = int(sum(int(y[fr[i][0]:fr[i][1]].sum()) for i in idx))
            h_k[k] = hate / tot if tot else vid_frac
            hate_frames[k] = hate
        oracle_w = np.array([h_k[d] if d is not None else vid_frac for d in dom])
        # control: the same oracle with the speaker labels permuted among the windows that have one (seeded)
        perm_within = []
        rng = np.random.default_rng(0)
        has = [i for i, d in enumerate(dom) if d is not None]
        for _ in range(N_PERM if len(has) > 1 else 0):
            pd = list(dom)
            lab = [dom[i] for i in has]
            rng.shuffle(lab)
            for i, l in zip(has, lab):
                pd[i] = l
            ph = {}
            for k in spks:
                idx = [i for i, d in enumerate(pd) if d == k]
                tot = sum(fr[i][1] - fr[i][0] for i in idx)
                ph[k] = (sum(int(y[fr[i][0]:fr[i][1]].sum()) for i in idx) / tot) if tot else vid_frac
            perm_within.append(auc(y, expand(np.array([ph[d] if d is not None else vid_frac for d in pd]))))
        within_perm = mean(perm_within)
        pooled = z.copy()
        for k in spks:
            idx = [i for i, d in enumerate(dom) if d == k]
            pooled[idx] = z[idx].mean()
        pos = np.where(hf >= POS_FRAC)[0]
        neg = np.where(hf == 0)[0]
        main_spk = max(hate_frames, key=hate_frames.get) if hate_frames and max(hate_frames.values()) > 0 else None
        fh = [int(i) for i in neg if len(pos) and np.isfinite(r6w[i]) and r6w[i] > np.median(r6w[pos])]
        neg_spk = [int(i) for i in neg if dom[i] is not None]
        fh_spk = [i for i in fh if dom[i] is not None]
        rows.append({
            "video": vid, "n_windows": len(wins), "hate_share": vid_frac, "diar_status": tr["status"],
            "n_speakers": tr.get("n_speakers"), "valid_speakers": valid, "multi": multi,
            "speech_share": (speech_total / reads[vid]["duration"]) if reads[vid].get("duration") else None,
            "no_speaker_windows": int(sum(d is None for d in dom)),
            "within_r6_bma": auc(y, curve), "within_oracle": auc(y, expand(oracle_w)), "within_oracle_perm": within_perm,
            "within_z": auc(y, expand(z)), "within_pooled": auc(y, expand(pooled)),
            "n_pos": int(len(pos)), "n_neg": int(len(neg)), "main_speaker": main_spk,
            "false_high": len(fh), "false_high_with_speaker": len(fh_spk),
            "false_high_other_speaker": int(sum(dom[i] != main_spk for i in fh_spk)) if main_spk else None,
            "neg_with_speaker": len(neg_spk),
            "neg_other_speaker": int(sum(dom[i] != main_spk for i in neg_spk)) if main_spk else None,
        })

    elig = rows
    multi = [r for r in rows if r["multi"]]
    p1 = len(multi) / len(elig) if elig else None
    p2 = {"n": len(multi),
          "within_oracle": mean(r["within_oracle"] for r in multi),
          "within_r6_bma": mean(r["within_r6_bma"] for r in multi),
          "within_oracle_perm": mean(r["within_oracle_perm"] for r in multi)}
    p2["diff"] = (p2["within_oracle"] - p2["within_r6_bma"]) if multi else None
    p2_all = {"n": len(elig), "within_oracle": mean(r["within_oracle"] for r in elig),
              "within_r6_bma": mean(r["within_r6_bma"] for r in elig)}
    fh_spk = sum(r["false_high_with_speaker"] for r in rows if r["main_speaker"])
    fh_other = sum(r["false_high_other_speaker"] or 0 for r in rows if r["main_speaker"])
    neg_spk = sum(r["neg_with_speaker"] for r in rows if r["main_speaker"])
    neg_other = sum(r["neg_other_speaker"] or 0 for r in rows if r["main_speaker"])
    p3 = {"videos_with_main_speaker": sum(1 for r in rows if r["main_speaker"]),
          "false_high_windows_with_speaker": fh_spk,
          "false_high_other_speaker_share": fh_other / fh_spk if fh_spk else None,
          "negative_windows_with_speaker": neg_spk,
          "negative_other_speaker_share": neg_other / neg_spk if neg_spk else None}
    p3_multi = {}
    mm = [r for r in multi if r["main_speaker"]]
    a = sum(r["false_high_with_speaker"] for r in mm)
    b = sum(r["false_high_other_speaker"] or 0 for r in mm)
    c = sum(r["neg_with_speaker"] for r in mm)
    d = sum(r["neg_other_speaker"] or 0 for r in mm)
    p3_multi = {"videos": len(mm), "false_high_other_speaker_share": b / a if a else None,
                "negative_other_speaker_share": d / c if c else None}
    p4 = {"n": len(multi), "within_z": mean(r["within_z"] for r in multi),
          "within_pooled": mean(r["within_pooled"] for r in multi)}
    p4["diff"] = (p4["within_pooled"] - p4["within_z"]) if multi else None
    gates = {"P1": (p1 is not None and p1 >= GATE_P1), "P2": (p2["diff"] is not None and p2["diff"] >= GATE_P2)}
    return {"dataset": ds, "diarization": diar, "n_eligible": len(elig), "n_multi": len(multi), "P1": p1, "P2": p2,
            "P2_all_eligible": p2_all, "P3": p3, "P3_multi": p3_multi, "P4": p4, "gates": gates}, rows


def main():
    global TURNS, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-root", default=None, help="override data/speaker_turns (dry runs only)")
    ap.add_argument("--out", default=None, help="override the output directory (dry runs only)")
    a = ap.parse_args()
    if a.turns_root:
        TURNS = Path(a.turns_root) / "{ds}/turns.jsonl"
    if a.out:
        OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    summary, per_video, lines = {}, {}, []
    lines.append(f"candidate 44 pre-check, host {socket.gethostname()}; constants: overlap>={MIN_OVERLAP}s share>={MIN_SHARE} "
                 f"windows>={MIN_WINDOWS} pos>={POS_FRAC}; gates P1>={GATE_P1} P2>={GATE_P2}")
    for ds in DATASETS:
        s, rows = analyse(ds)
        summary[ds] = s
        per_video[ds] = rows
        d = s["diarization"]
        lines.append(f"\n{ds}: diarization ok {d['ok']}/{d['videos']} failed {d['failed']} speakers {d['n_speakers_hist']} "
                     f"speech share median {d['speech_share_median']}")
        lines.append(f"  P1 multi-speaker share {s['P1']:.3f} ({s['n_multi']}/{s['n_eligible']}) gate {s['gates']['P1']}")
        p2 = s["P2"]
        lines.append(f"  P2 multi subset n={p2['n']}: oracle within {p2['within_oracle']} r6_bma {p2['within_r6_bma']} "
                     f"diff {p2['diff']} gate {s['gates']['P2']}; permuted-label oracle {p2['within_oracle_perm']}; all eligible: oracle {s['P2_all_eligible']['within_oracle']} "
                     f"r6_bma {s['P2_all_eligible']['within_r6_bma']}")
        p3 = s["P3"]
        lines.append(f"  P3 false-high windows with speaker {p3['false_high_windows_with_speaker']}: other-speaker share "
                     f"{p3['false_high_other_speaker_share']} (all negatives {p3['negative_other_speaker_share']}); "
                     f"multi subset: {s['P3_multi']}")
        p4 = s["P4"]
        lines.append(f"  P4 multi subset: within z {p4['within_z']} pooled-by-speaker {p4['within_pooled']} diff {p4['diff']}")
    both = all(summary[ds]["gates"]["P1"] and summary[ds]["gates"]["P2"] for ds in DATASETS)
    summary["decision"] = "implement candidate 44 (P1 and P2 pass on both corpora)" if both else \
        "do not implement (a declared gate fails on at least one corpus)"
    lines.append("\ndecision: " + summary["decision"])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "per_video.json").write_text(json.dumps(per_video, indent=1))
    (OUT / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
