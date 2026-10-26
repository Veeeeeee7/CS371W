#!/usr/bin/env python3
"""
Table 1 (tab:shift) as LaTeX, from the rows o1_shift.py prints.

Two blocks of rows, accuracy, AUROC and F1 each: the pooled gap (acc_*, the AUROC columns, f1_*)
and the per-tile gap (tile_acc_*, tile_auc_*, tile_f1_*; accuracy and F1 with each tile's ID
points weighted to its Zone A share of positives). ID and Zone A are means over the models (per
tile: over the tiles and models), the gap is ID - Zone A with its 95% tile-bootstrap interval,
and Worse is the share of models with a positive gap. The values come from the printed rows,
which round the full-precision values once to 3 decimals (the CSV keeps 4, and rounding those
again could move a last digit).

Reads   the log of an o1_shift.py run (its printed rows)
Writes  the tabular to stdout

    python aotil/o1_table.py logs/<date>/o1_shift.log --cell FL_band_AE_s0 \\
        > results/<date>/o1_table.tex
"""
from __future__ import annotations

import argparse
import ast
import re
from decimal import ROUND_HALF_UP, Decimal

# (block, row label, ID column, Zone A column, gap column prefix, worse column)
ROWS = (("Pooled", "Accuracy", "acc_ID", "acc_A", "acc_", "acc_worse"),
        ("Pooled", "AUROC", "ID_AUC", "A_AUC", "", "worse"),
        ("Pooled", "F1", "f1_ID", "f1_A", "f1_", "f1_worse"),
        ("Per tile", "Accuracy", "tile_acc_ID", "tile_acc_A", "tile_acc_", "tile_acc_worse"),
        ("Per tile", "AUROC", "tile_auc_ID", "tile_auc_A", "tile_auc_", "tile_auc_worse"),
        ("Per tile", "F1", "tile_f1_ID", "tile_f1_A", "tile_f1_", "tile_f1_worse"))


def printed_rows(log: str) -> list[dict]:
    """The rows o1_shift.py printed (print_row), NaN read back as None."""
    rows = []
    with open(log) as f:
        for line in f:
            if line.startswith("{'cell'"):
                rows.append(ast.literal_eval(re.sub(r"\bnan\b", "None", line)))
    return rows


def number(x: float) -> str:
    """.722, $-$.007: three decimals without the leading zero."""
    text = f"{abs(x):.3f}".lstrip("0")
    return ("$-$" if x < 0 and text != ".000" else "") + text


def percent(share: float) -> str:
    """A share printed to 3 decimals as a whole percent, halves rounded up (0.425 -> 43\\%)."""
    whole = (Decimal(str(share)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return f"{whole}\\%"


def tabular(row: dict) -> str:
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"\textbf{Metric} & \textbf{ID} & \textbf{Zone A} & \textbf{Gap [95\% CI]} & "
             r"\textbf{Worse} \\"]
    block = None
    for name, label, on_id, on_a, gap, worse in ROWS:
        if name != block:
            lines += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{name}}}}} \\"]
            block = name
        interval = f"{number(row[gap + 'gap'])} [{number(row[gap + 'ci_lo'])}, " \
                   f"{number(row[gap + 'ci_hi'])}]"
        lines.append(f"{label} & {number(row[on_id])} & {number(row[on_a])} & {interval} & "
                     f"{percent(row[worse])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="the log of an o1_shift.py run")
    parser.add_argument("--cell", default="FL_band_AE_s0")
    parser.add_argument("--scope", default="all")
    args = parser.parse_args()
    found = [row for row in printed_rows(args.log)
             if row["cell"] == args.cell and row["scope"] == args.scope]
    if not found:
        raise SystemExit(f"no printed row for cell {args.cell}, scope {args.scope}")
    row = found[-1]
    print(f"% Table 1 (tab:shift), cell {args.cell}, scope {args.scope}: pooled and per-tile gaps")
    print(f"% from the printed rows of {args.log} (values rounded once to 3 decimals);")
    print(f"% per tile: {row['tile_acc_n_tiles']} tiles for accuracy and F1, "
          f"{row['tile_auc_n_tiles']} for AUROC. Made by aotil/o1_table.py.")
    print(tabular(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
