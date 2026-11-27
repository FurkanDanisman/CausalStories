"""Gemma homelessness world graph (Causal Stories, grouping procedure, 10% cut): the old causal-edge definition
(grid_v8c) next to the new group headers + Regan et al.'s Without-effect rows (grid_v9), same nodes.
Usage: python benchmark/plot_v9_compare.py -> benchmark/stories_v8/results_aggregated_v9_gemma.pdf"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import plot_results_v7 as P

P.OUT.mkdir(parents=True, exist_ok=True)
run = "homelessness__gemma__llm"
cells = [(P.true_world("homelessness"), r"\textbf{True world graph}")]
for grid, lab in [("grid_v8c", "Causal Stories, old definition"), ("grid_v9", "Causal Stories, new definition")]:
    P.OURS_BY_GEN["gemma"] = grid
    f, cap = P.aggregated(run, 10, f"homelessness_gemma_{grid}")
    cells.append((f, f"\\textbf{{{lab}}}\\\\" + cap))
cells.append((None, ""))
tex = [P.HEAD, r"\section*{Homelessness -- Gemma 4 31B -- Causal Stories at the 10\% cut}",
       P.NOTE + r" Same extracted nodes in both runs; only the causal-edge definition in the arrow question differs "
       r"(new: group headers ``A makes B more likely / less likely \ldots\ even when B does not happen in the end / happens anyway'' "
       r"and the two Without-effect rows of Regan et al.\ 2023, Table 2).", P.grid(cells, "0.38")]
P.build("results_aggregated_v9_gemma", tex)
