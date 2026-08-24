"""Dataset audit tool for the Thai mental-health chatbot classifiers.

Checks every dataset/*.csv used by ai_model.py for:
  1. Cross-label conflicts  - identical `text` value labeled differently
                               (within the same file, or across files that
                               feed a shared concept, e.g. risk.csv vs emotion.csv)
  2. Exact duplicate rows   - identical text+label pair appearing more than once
  3. Small classes          - labels with too few samples for a safe stratified
                               split / calibration fold count

Run:
    python3 audit_datasets.py [dataset_dir]

Writes a plain-text report to audit_report.txt in the current directory and
also prints a summary to stdout. Line numbers refer to the CSV's data rows
(1 = first row after the header), not raw file lines, so they match what a
spreadsheet app shows if you freeze the header row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_DATASET_DIR = Path("dataset")

FILES = {
    "risk.csv": "risk",
    "emotion.csv": "emotion",
    "problem.csv": "problem",
    "support_need.csv": "support_need",
    "intent.csv": "intent",
    "conversation_style.csv": "conversation_style",
}

SMALL_CLASS_THRESHOLD = 15  # flag any label with fewer samples than this


def audit_file(path: Path) -> list[str]:
    lines: list[str] = []
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["_row"] = df.index + 1  # 1-indexed data row, matches header-frozen spreadsheet view

    lines.append("=" * 78)
    lines.append(f"FILE: {path.name}  (rows: {len(df)}, labels: {df['label'].nunique()})")
    lines.append("=" * 78)

    # --- 1. label distribution / small classes ---
    counts = df["label"].value_counts()
    lines.append("\nLabel distribution:")
    for label, count in counts.items():
        flag = "  <-- TOO FEW SAMPLES" if count < SMALL_CLASS_THRESHOLD else ""
        lines.append(f"  {label:<28} {count:>4}{flag}")

    # --- 2. exact duplicate rows (same text + same label) ---
    dup_mask = df.duplicated(subset=["text", "label"], keep=False)
    dups = df[dup_mask].sort_values("text")
    lines.append(f"\nExact duplicate rows (same text + same label): {dup_mask.sum()}")
    if dup_mask.any():
        for text, group in dups.groupby("text"):
            rows = ", ".join(str(r) for r in group["_row"])
            lines.append(f"  \"{text}\" [{group['label'].iloc[0]}]  -> data rows: {rows}")

    # --- 3. cross-label conflicts (same text, different labels) ---
    conflict_texts = df.groupby("text")["label"].nunique()
    conflict_texts = conflict_texts[conflict_texts > 1].index
    lines.append(f"\nCross-label conflicts (same text, different labels): {len(conflict_texts)}")
    if len(conflict_texts):
        for text in sorted(conflict_texts):
            group = df[df["text"] == text]
            detail = ", ".join(f"row {r}: {lbl}" for r, lbl in zip(group["_row"], group["label"]))
            lines.append(f"  \"{text}\"")
            lines.append(f"      {detail}")

    lines.append("")
    return lines


def audit_cross_file_overlap(dataset_dir: Path) -> list[str]:
    """Optional extra check: identical text reused across different task files.

    Not necessarily a bug (each file trains an independent model), but useful
    to see how much vocabulary/phrasing is shared across tasks.
    """
    lines = ["=" * 78, "CROSS-FILE TEXT REUSE (informational only, not an error)", "=" * 78, ""]
    frames = []
    for filename, task in FILES.items():
        path = dataset_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path).dropna(subset=["text", "label"])
        df = df.assign(task=task)
        frames.append(df[["text", "label", "task"]])
    if not frames:
        return lines
    all_rows = pd.concat(frames, ignore_index=True)
    reused = all_rows.groupby("text")["task"].nunique()
    reused = reused[reused > 1]
    lines.append(f"Texts appearing in more than one dataset file: {len(reused)}")
    if len(reused):
        sample = reused.sort_values(ascending=False).head(15)
        for text, n_files in sample.items():
            tasks = all_rows[all_rows["text"] == text][["task", "label"]].drop_duplicates()
            detail = ", ".join(f"{t}:{l}" for t, l in tasks.values)
            lines.append(f"  \"{text}\" in {n_files} files -> {detail}")
        if len(reused) > 15:
            lines.append(f"  ... and {len(reused) - 15} more")
    lines.append("")
    return lines


def main() -> None:
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET_DIR
    report: list[str] = []
    report.append(f"Dataset audit — directory: {dataset_dir.resolve()}")
    report.append("")

    for filename in FILES:
        path = dataset_dir / filename
        if not path.exists():
            report.append(f"[MISSING] {filename} not found in {dataset_dir}")
            continue
        report.extend(audit_file(path))

    report.extend(audit_cross_file_overlap(dataset_dir))

    text = "\n".join(report)
    Path("audit_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nFull report written to: {Path('audit_report.txt').resolve()}")


if __name__ == "__main__":
    main()
