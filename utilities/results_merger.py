#!/usr/bin/env python3
"""
results_merger.py — run-archiving merger with attack_name + defended support

- Merges current results/*.json (excluding final_results.json)
- Accepts filenames:
    model_usecase.json
    model_usecase_defended.json
- Writes per-run outputs under results/testNNN/:
    - final_results.json  (flat records)
    - summary.txt
- Appends a rolling results/overall_summary.json

Each record fields:
    attack_family, attack_name, attack_prompt, model_response,
    attack_success, latency, model_name, usecase, defended
"""

import argparse, json, re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

OUT_KEYS = [
    "attack_family",
    "attack_name",
    "attack_prompt",
    "model_response",
    "attack_success",
    "latency",
    "model_name",
    "usecase",
    "defended",
]

def parse_filename(filename: str):
    """
    Accept:
      model_usecase.json
      model_usecase_defended.json
    Returns: (model, usecase, defended: bool)
    """
    stem = Path(filename).stem
    defended = False
    if stem.endswith("_defended"):
        defended = True
        stem = stem[: -len("_defended")]
    if "_" not in stem:
        raise ValueError(f"Filename must be model_usecase[_defended].json: {filename}")
    model, usecase = stem.rsplit("_", 1)  # split on LAST underscore
    return model, usecase, defended

def normalize(rec: dict, model: str, usecase: str, defended_from_name: bool) -> dict:
    # Prefer explicit flag from record if present; else derive from filename suffix
    defended = bool(rec.get("defended", rec.get("defence_active", defended_from_name)))
    out = {
        "attack_family": rec.get("attack_family"),
        "attack_name": rec.get("attack_name"),  # NEW
        "attack_prompt": rec.get("attack_prompt"),
        "model_response": rec.get("model_response"),
        "attack_success": bool(rec.get("attack_success", False)),
        "latency": rec.get("latency", rec.get("latency_ms")),
        "model_name": model,
        "usecase": usecase,
        "defended": defended,                    # NEW
    }
    return {k: out.get(k) for k in OUT_KEYS}

def find_next_run_id(runs_root: Path) -> str:
    runs_root.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"^test(\d{3})$")
    n = 0
    for p in runs_root.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                n = max(n, int(m.group(1)))
    return f"test{n+1:03d}"

def load_inputs(results_dir: Path):
    return [
        p for p in sorted(results_dir.glob("*.json"))
        if p.name != "final_results.json"
    ]

def compute_stats(rows):
    total = len(rows)
    success = sum(1 for r in rows if bool(r.get("attack_success")))
    rate = (success / total) if total else 0.0

    by_model = Counter(r["model_name"] for r in rows if r.get("model_name") is not None)
    by_usecase = Counter(r["usecase"] for r in rows if r.get("usecase") is not None)
    by_family = Counter(r["attack_family"] for r in rows if r.get("attack_family") is not None)
    by_defended = Counter("defended" if r.get("defended") else "baseline" for r in rows)

    sbm = defaultdict(lambda: [0,0])  # success by model
    for r in rows:
        m = r.get("model_name")
        if m is None: continue
        sbm[m][1] += 1
        if r.get("attack_success"): sbm[m][0] += 1
    success_by_model = {m: {"success": s, "total": t, "rate": (s/t if t else 0.0)} for m,(s,t) in sbm.items()}

    return {
        "total": total,
        "success": success,
        "success_rate": rate,
        "by_model": dict(by_model),
        "by_usecase": dict(by_usecase),
        "by_family": dict(by_family),
        "by_defended": dict(by_defended),  # NEW
        "success_by_model": success_by_model,
    }

def write_summary_txt(path: Path, run_id: str, stats: dict):
    lines = []
    lines.append(f"Run: {run_id}")
    lines.append(f"Timestamp: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    lines.append("")
    lines.append(f"Total records: {stats['total']}")
    lines.append(f"Total successes: {stats['success']}")
    lines.append(f"Success rate: {stats['success_rate']:.3f}")
    lines.append("")
    lines.append("By model:")
    for m, n in sorted(stats["by_model"].items()):
        s = stats["success_by_model"].get(m, {"success":0,"total":n,"rate":0.0})
        lines.append(f"  - {m}: {n} rows | success {s['success']}/{s['total']} = {s['rate']:.3f}")
    lines.append("")
    lines.append("By usecase:")
    for u, n in sorted(stats["by_usecase"].items()):
        lines.append(f"  - {u}: {n}")
    lines.append("")
    lines.append("By attack_family:")
    for f, n in sorted(stats["by_family"].items()):
        lines.append(f"  - {f}: {n}")
    lines.append("")
    lines.append("Baseline vs Defended:")
    for k, n in sorted(stats["by_defended"].items()):
        lines.append(f"  - {k}: {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def append_overall_summary(runs_root: Path, run_id: str, stats: dict):
    overall = runs_root / "overall_summary.json"
    entry = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total": stats["total"],
        "success": stats["success"],
        "success_rate": round(stats["success_rate"], 6),
        "models": stats["by_model"],
        "usecases": stats["by_usecase"],
        "families": stats["by_family"],
        "baseline_vs_defended": stats["by_defended"],  # NEW
    }
    if overall.exists():
        try:
            data = json.loads(overall.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []
    data.append(entry)
    overall.write_text(json.dumps(data, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", help="Where input JSONs live")
    ap.add_argument("--runs-root",  default="results", help="Where testNNN folders live")
    ap.add_argument("--run-id",     default=None,      help="Optional run id, e.g., test007")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    runs_root   = Path(args.runs_root)

    inputs = load_inputs(results_dir)
    run_id = args.run_id or find_next_run_id(runs_root)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not inputs:
        (run_dir / "final_results.json").write_text("[]\n", encoding="utf-8")
        empty = {"total":0,"success":0,"success_rate":0.0,
                 "by_model":{},"by_usecase":{},"by_family":{},
                 "by_defended":{},"success_by_model":{}}
        write_summary_txt(run_dir / "summary.txt", run_id, empty)
        append_overall_summary(runs_root, run_id, empty)
        print(f"[OK] No inputs found; created empty run at {run_dir}")
        return

    merged = []
    for p in inputs:
        try:
            model, usecase, defended_from_name = parse_filename(p.name)
        except ValueError as e:
            print(f"[WARN] {e} — skipping {p.name}")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                print(f"[WARN] {p.name} is not a JSON list — skipping.")
                continue
        except Exception as e:
            print(f"[WARN] Failed to read {p.name}: {e}")
            continue

        for rec in data:
            if isinstance(rec, dict):
                merged.append(normalize(rec, model, usecase, defended_from_name))

    # Write per-run artifacts
    final_path = run_dir / "final_results.json"
    final_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    stats = compute_stats(merged)
    write_summary_txt(run_dir / "summary.txt", run_id, stats)
    append_overall_summary(runs_root, run_id, stats)

    print(f"[OK] Wrote {len(merged)} records → {final_path}")
    print(f"[OK] Summary → {run_dir/'summary.txt'}")
    print(f"[OK] Updated overall → {runs_root/'overall_summary.json'}")

if __name__ == "__main__":
    main()