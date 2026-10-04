#!/usr/bin/env python3
"""Describe actual lexical alternatives without labels or score selection."""
import argparse
import json
from pathlib import Path
import re

from extract import CACHE, ROOT, selected_rows


def lexical(text):
    return re.findall(r"\w+", text.casefold())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    rows = []
    for row in selected_rows(args.smoke):
        path = CACHE / row['dataset'] / (row['video_id'] + '.json')
        source = json.loads(path.read_text())
        available = [w for w in source['windows'] if w['available']]
        rows.append(dict(
            dataset=row['dataset'], video_id=row['video_id'], source=str(path.relative_to(ROOT)),
            windows=len(source['windows']), recognized=len(available),
            multiple_exact_text=sum(len({b['text'] for b in w['beams']}) > 1 for w in available),
            multiple_lexical_text=sum(len({tuple(lexical(b['text'])) for b in w['beams']}) > 1 for w in available),
            onebest_differs_from_native_lexical=sum(lexical(w['confusion']['onebest']) != lexical(w['native_body']) for w in available),
            empty_onebest=sum(not w['confusion']['onebest'] for w in available),
            truncated_beams=sum(b['truncated'] for w in available for b in w['beams']),
            total_beams=5 * len(available),
            slots=sum(len(w['confusion']['slots']) for w in available),
            alternative_slots=sum(len(s['alternatives']) > 1 for w in available for s in w['confusion']['slots']),
            epsilon_slots=sum(any(not a['text'] and a['mass'] > 0 for a in s['alternatives']) for w in available for s in w['confusion']['slots']),
            zero_mass_alternatives=sum(a['mass'] == 0 for w in available for s in w['confusion']['slots'] for a in s['alternatives']),
            standalone_seconds=source['standalone_seconds'], actual_forwards=source['actual_forwards']))
    result = dict(GT_read=False, scope='descriptive actual inputs; no scoring or selection',
                  lexical_comparison='Unicode word tokens, casefolded; removes punctuation only for this diagnostic', videos=rows)
    output = ROOT / 'runs/20261004_m1_lattice' / ('r1_extract_' + ('smoke' if args.smoke else 'main')) / 'input_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
