"""
Build self-contained HTML pages for runnable examples (browser-openable via file://).

Embeds a PNG as a data URI and renders a sortable-style leaderboard table.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Iterable


def png_to_data_uri(png_path: Path) -> str:
    data = png_path.read_bytes()
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_example_html_page(
    title: str,
    intro_html: str,
    image_data_uri: str,
    leaderboard_rows: Iterable[dict],
    image_alt: str = "Swarm visualization",
) -> str:
    """
    leaderboard_rows: dicts with keys
      rank, algorithm, best_cost, runtime_s, cost_score, time_score, composite_score
    """
    rows_html = []
    for row in leaderboard_rows:
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['rank']))}</td>"
            f"<td><code>{html.escape(str(row['algorithm']))}</code></td>"
            f"<td>{html.escape(str(row['best_cost']))}</td>"
            f"<td>{html.escape(str(row['runtime_s']))}</td>"
            f"<td>{html.escape(str(row['cost_score']))}</td>"
            f"<td>{html.escape(str(row['time_score']))}</td>"
            f"<td><strong>{html.escape(str(row['composite_score']))}</strong></td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf1;
      --muted: #9aa7b2;
      --accent: #3d8bfd;
      --border: #2d3a4d;
    }}
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      line-height: 1.55;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 2rem 1.25rem 3rem;
    }}
    h1 {{
      font-size: 1.65rem;
      font-weight: 650;
      margin: 0 0 0.5rem;
    }}
    .intro {{
      color: var(--muted);
      margin-bottom: 1.5rem;
    }}
    .figure {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      margin-bottom: 2rem;
    }}
    .figure img {{
      width: 100%;
      height: auto;
      border-radius: 6px;
      display: block;
    }}
    .figure figcaption {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-top: 0.75rem;
    }}
    h2 {{
      font-size: 1.2rem;
      margin: 0 0 1rem;
      color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }}
    th, td {{
      padding: 0.65rem 0.85rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: #243044;
      color: var(--text);
      font-weight: 600;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .note {{
      margin-top: 1.25rem;
      font-size: 0.88rem;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <div class="intro">{intro_html}</div>
    <figure class="figure">
      <img src="{image_data_uri}" alt="{html.escape(image_alt)}" />
      <figcaption>3D trajectories (subset), final swarm, delegate sites, and standoff boundary.</figcaption>
    </figure>
    <h2>Algorithm leaderboard</h2>
    <p style="color: var(--muted); font-size: 0.9rem; margin-top: 0;">
      Relative scores: best cost and best runtime each earn 100; others scale proportionally.
      Composite = 60% cost + 40% time (higher is better).
    </p>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Algorithm</th>
          <th>Best cost</th>
          <th>Time (s)</th>
          <th>Cost score</th>
          <th>Time score</th>
          <th>Composite</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
    <p class="note">
      Open this file directly in your browser (<code>file://</code>). Regenerate by running
      <code>python tests/examples/satellite_repair_swarm.py</code> from the SwarmOpt repo root.
    </p>
  </main>
</body>
</html>
"""
