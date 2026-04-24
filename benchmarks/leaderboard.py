#!/usr/bin/env python3
"""
Aggregate benchmark CSVs under csvfiles/ and render a local HTML leaderboard.

Usage:
  python leaderboard.py              # write leaderboard.html, print file:// URL
  python leaderboard.py --open       # same, then open in the default browser
"""

import argparse
import csv
import html
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# Minimum required for aggregation; new benchmark CSVs may include stratum, std_cost.
REQUIRED_COLUMNS = ("algo", "function", "avg_cost", "avg_time")
OPTIONAL_DISPLAY = ("stratum", "std_cost")


def _repo_benchmarks_dir() -> Path:
    return Path(__file__).resolve().parent


def load_rows(csv_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(csv_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            fields = [c.strip() for c in reader.fieldnames]
            if not all(c in fields for c in REQUIRED_COLUMNS):
                continue
            for raw in reader:
                row = {k: (raw.get(k) or "").strip() for k in REQUIRED_COLUMNS}
                for opt in OPTIONAL_DISPLAY:
                    if opt in fields:
                        row[opt] = (raw.get(opt) or "").strip()
                if not row["algo"] or not row["function"]:
                    continue
                try:
                    float(row["avg_cost"])
                    float(row["avg_time"])
                except ValueError:
                    continue
                rows.append(row)
    return rows


def best_per_algo_function(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep the row with lowest avg_cost for each (function, algo)."""
    best: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row["function"], row["algo"])
        if key not in best:
            best[key] = row
            continue
        if float(row["avg_cost"]) < float(best[key]["avg_cost"]):
            best[key] = row
    return sorted(
        best.values(),
        key=lambda r: (r["function"].lower(), float(r["avg_cost"]), r["algo"].lower()),
    )


def ranks_by_function(aggregated: List[Dict[str, str]]) -> Dict[Tuple[str, str], int]:
    by_func: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in aggregated:
        by_func[row["function"]].append(row)
    out: Dict[Tuple[str, str], int] = {}
    for func, group in by_func.items():
        ordered = sorted(group, key=lambda r: float(r["avg_cost"]))
        for i, row in enumerate(ordered, start=1):
            out[(func, row["algo"])] = i
    return out


def render_html(aggregated: List[Dict[str, str]], ranks: Dict[Tuple[str, str], int]) -> str:
    rows_html = []
    has_stratum = any(r.get("stratum") for r in aggregated)
    has_std = any(r.get("std_cost") for r in aggregated)
    for row in aggregated:
        rank = ranks[(row["function"], row["algo"])]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
        if not has_stratum:
            stratum_cell = ""
        elif (row.get("stratum") or "").strip():
            stratum_cell = f"<td>{html.escape(row['stratum'])}</td>"
        else:
            stratum_cell = "<td class=\"na\">—</td>"
        std_cell = ""
        if has_std:
            s = row.get("std_cost", "").strip()
            if s:
                try:
                    std_cell = f"<td class=\"num\">{float(s):.6g}</td>"
                except ValueError:
                    std_cell = "<td class=\"na\">—</td>"
            else:
                std_cell = "<td class=\"na\">—</td>"

        rows_html.append(
            "<tr>"
            f"<td>{html.escape(row['function'])}</td>"
            f"<td>{html.escape(row['algo'])}</td>"
            f"{stratum_cell}"
            f"<td class=\"num\">{medal} {rank}</td>"
            f"<td class=\"num\">{float(row['avg_cost']):.6g}</td>"
            f"{std_cell}"
            f"<td class=\"num\">{float(row['avg_time']):.6g}</td>"
            "</tr>"
        )

    ncol = 5 + (1 if has_stratum else 0) + (1 if has_std else 0)
    body = "\n".join(rows_html) if rows_html else ("<tr><td colspan=\"%d\">No benchmark rows found.</td></tr>" % ncol)
    thead_extra = "        <th>Stratum</th>\n" if has_stratum else ""
    thead_std = "        <th class=\"num\">Std cost</th>\n" if has_std else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SwarmOpt benchmark leaderboard</title>
  <style>
    :root {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      line-height: 1.5;
      color: #1a1a1a;
      background: #f4f4f5;
    }}
    body {{
      margin: 0;
      padding: 1.5rem;
      max-width: 56rem;
      margin-inline: auto;
    }}
    h1 {{
      font-size: 1.5rem;
      margin: 0 0 0.25rem;
    }}
    p.meta {{
      margin: 0 0 1.25rem;
      color: #52525b;
      font-size: 0.9rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    th, td {{
      padding: 0.65rem 0.85rem;
      text-align: left;
      border-bottom: 1px solid #e4e4e7;
    }}
    th {{
      background: #fafafa;
      font-weight: 600;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #71717a;
    }}
    tr:last-child td {{ border-bottom: none; }}
    td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
    td.na {{ color: #a1a1aa; text-align: center; }}
    caption {{
      caption-side: bottom;
      padding-top: 0.75rem;
      font-size: 0.8rem;
      color: #71717a;
    }}
  </style>
</head>
<body>
  <h1>SwarmOpt benchmark leaderboard</h1>
  <p class="meta">Best <code>avg_cost</code> per algorithm and test function across all runs in <code>csvfiles/</code> (across run files, keeping the best table entry). Lower cost is better.</p>
  <table>
    <thead>
      <tr>
        <th>Function</th>
        <th>Algorithm</th>
{thead_extra}
        <th class="num">Rank</th>
        <th class="num">Avg cost</th>
{thead_std}
        <th class="num">Avg time (s)</th>
      </tr>
    </thead>
    <tbody>
{body}
    </tbody>
    <caption>Regenerate with <code>python benchmarks/leaderboard.py</code> from the repo root.</caption>
  </table>
</body>
</html>
"""


def file_uri_for_path(path: Path) -> str:
    return path.resolve().as_uri()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local HTML leaderboard from benchmark CSVs.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory of CSV files (default: benchmarks/csvfiles next to this script)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: benchmarks/leaderboard.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the generated page in the default browser",
    )
    args = parser.parse_args()

    base = _repo_benchmarks_dir()
    csv_dir = (args.csv_dir or (base / "csvfiles")).resolve()
    out_path = (args.out or (base / "leaderboard.html")).resolve()

    if not csv_dir.is_dir():
        print(f"No CSV directory: {csv_dir}", file=sys.stderr)
        return 1

    rows = load_rows(csv_dir)
    aggregated = best_per_algo_function(rows)
    ranks = ranks_by_function(aggregated)
    html_doc = render_html(aggregated, ranks)

    out_path.write_text(html_doc, encoding="utf-8")
    uri = file_uri_for_path(out_path)
    print(uri)

    if args.open_browser:
        import webbrowser

        webbrowser.open(uri)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
