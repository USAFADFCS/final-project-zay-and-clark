#!/usr/bin/env python3
import argparse, json, re, sys
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class Entry:
    label: str
    score: float

@dataclass
class Result:
    top_level: Optional[Entry]
    refined: Optional[Entry]
    alternatives: List[Entry]

RESERVED = {"label","labels","score","scores","top_level","tl_result","sequence","Action","Observation","Top-level","Refined","You","Hierarchical DeBERTa Classification"}

def is_classification_label(txt: str) -> bool:
    if not txt: return False
    t = str(txt).strip()
    if t in RESERVED: return False
    if len(t) < 8: return False
    if len(t.split()) < 3: return False
    return True

def try_compact_json(s: str):
    m = re.search(r"\{[\s\S]*\}", s)
    if not m: return None
    block = m.group(0)
    try: return json.loads(block)
    except Exception: return None

def try_top_refined_lines(s: str):
    out = {}
    m1 = re.search(r"Top-level:\s*(.*?)\s*\(score:\s*([0-9.]+)\)", s, flags=re.I)
    m2 = re.search(r"Refined:\s*(.*?)\s*\(score:\s*([0-9.]+)\)", s, flags=re.I)
    if m1: out["top_level"] = Entry(m1.group(1).strip(), float(m1.group(2)))
    if m2: out["refined"] = Entry(m2.group(1).strip(), float(m2.group(2)))
    return out if out else None

def extract_array_after_key(s: str, key: str):
    m = re.search(fr'{key}\s*[:=]\s*\[(.*?)\]', s, flags=re.S)
    if not m: return None
    inside = m.group(1)
    try:
        return json.loads("[" + inside + "]")
    except Exception:
        qs = re.findall(r'"([^"]+)"', inside)
        if qs: return qs
        nums = re.findall(r"-?\d*\.\d+|-?\d+", inside)
        return [float(x) for x in nums] if nums else None

def try_labels_scores_arrays(s: str):
    labels = extract_array_after_key(s, "labels")
    scores = extract_array_after_key(s, "scores")
    if labels and scores and len(labels) == len(scores):
        out = []
        for lab, sc in zip(labels, scores):
            if is_classification_label(str(lab)):
                try: out.append(Entry(str(lab), float(sc)))
                except Exception: pass
        return out
    return None

def guess_pairs(s: str):
    out = []
    for m in re.finditer(r'"([^"]{4,200})"', s):
        label = m.group(1)
        if not is_classification_label(label): continue
        window = s[m.end(): m.end() + 220]
        num = re.search(r"([0-9]*\.[0-9]+)", window)
        if num:
            try: out.append(Entry(label, float(num.group(1))))
            except Exception: pass
    return out

def parse_raw(s: str) -> Result:
    top_level = None; refined = None; alts: List[Entry] = []

    cj = try_compact_json(s)
    if cj:
        if "top_level_label" in cj and "top_level_score" in cj:
            top_level = Entry(str(cj["top_level_label"]), float(cj["top_level_score"]))
        if "refined_label" in cj and "refined_score" in cj:
            refined = Entry(str(cj["refined_label"]), float(cj["refined_score"]))

    tr = try_top_refined_lines(s)
    if tr:
        if not top_level and tr.get("top_level"): top_level = tr["top_level"]
        if not refined and tr.get("refined"): refined = tr["refined"]

    alts = try_labels_scores_arrays(s) or []
    if not alts:
        alts = guess_pairs(s) or []

    # Dedup and sort
    uniq = {}
    for e in alts:
        key = (e.label, round(e.score, 10))
        uniq[key] = e
    alts = list(uniq.values())
    alts.sort(key=lambda x: x.score, reverse=True)

    if not top_level and alts: top_level = alts[0]
    if not refined and len(alts) > 1: refined = alts[1]

    return Result(top_level, refined, alts)

def to_html(result: Result, title: str = "Classification Output"):
    rows = []
    max_score = max([e.score for e in result.alternatives], default=1.0)
    for e in result.alternatives:
        pct = 0 if max_score == 0 else (e.score / max_score) * 100.0
        rows.append(f"<tr><td>{e.label}</td><td>{e.score:.6f}</td><td><div style='height:8px;background:#233044;border-radius:8px;overflow:hidden'><div style='height:8px;width:{pct:.2f}%;background:linear-gradient(90deg,#4da3ff,#74b7ff)'></div></div></td></tr>")
    top = "None" if not result.top_level else f"<strong>{result.top_level.label}</strong> <span style='color:#4da3ff'>({result.top_level.score:.4f})</span>"
    ref = "None" if not result.refined else f"<strong>{result.refined.label}</strong> <span style='color:#4da3ff'>({result.refined.score:.4f})</span>"
    json_block = json.dumps({
        "top_level": asdict(result.top_level) if result.top_level else None,
        "refined": asdict(result.refined) if result.refined else None,
        "alternatives": [asdict(x) for x in result.alternatives],
    }, indent=2)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{background:#0b0f14;color:#e6edf3;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}}.wrap{{max-width:1000px;margin:24px auto;padding:0 16px}}.card{{background:#121822;border:1px solid #1b2433;border-radius:14px;padding:16px;margin-bottom:16px}}table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid #1a2536;padding:8px 6px;text-align:left}}.muted{{color:#9fb3c8}}code,pre{{background:#0e1420;border:1px solid #27344a;border-radius:10px;padding:12px;display:block;overflow:auto}}</style></head>
<body><div class="wrap"><div class="card"><h2 style="margin:0 0 6px 0">{title}</h2><div class="muted">Top level: {top}<br>Refined: {ref}</div></div>
<div class="card"><h3 style="margin:0 0 8px 0">All labels</h3><table><thead><tr><th>Label</th><th>Score</th><th>Confidence</th></tr></thead><tbody>{''.join(rows) if rows else "<tr><td colspan='3' class='muted'>No labels were parsed.</td></tr>"}</tbody></table></div>
<div class="card"><h3 style="margin:0 0 8px 0">JSON</h3><pre><code>{json_block}</code></pre></div></div></body></html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="-", help="Input file path or '-' for stdin")
    ap.add_argument("--json", dest="json_out", help="Write normalized JSON to this file")
    ap.add_argument("--html", dest="html_out", help="Write a ready-to-open HTML report")
    ap.add_argument("--title", default="Classification Output", help="Title of the HTML report")
    args = ap.parse_args()

    text = sys.stdin.read() if args.input == "-" else open(args.input, "r", encoding="utf-8", errors="ignore").read()
    result = parse_raw(text)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "top_level": asdict(result.top_level) if result.top_level else None,
                "refined": asdict(result.refined) if result.refined else None,
                "alternatives": [asdict(x) for x in result.alternatives],
            }, f, indent=2)
    if args.html_out:
        html = to_html(result, title=args.title)
        with open(args.html_out, "w", encoding="utf-8") as f:
            f.write(html)

    if not args.json_out and not args.html_out:
        print(json.dumps({
            "top_level": asdict(result.top_level) if result.top_level else None,
            "refined": asdict(result.refined) if result.refined else None,
            "alternatives": [asdict(x) for x in result.alternatives],
        }, indent=2))

if __name__ == "__main__":
    main()