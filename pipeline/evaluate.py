"""Score an extracted graph against the gold graph.

Because node names differ between extraction and gold, we cannot compare edge
lists directly. We first ask the LLM to align extracted node ids to gold node ids
(semantic matching), then compare UNLABELED directed edges in the shared gold
vocabulary: an extracted edge (h, t) is a true positive iff align(h) and align(t)
are both non-null and (align(h), align(t)) is a gold edge.

Reported:
  * structural validity (all edge endpoints are declared nodes)
  * edge precision / recall / F1 over unlabeled directed edges
  * a holistic LLM-judge score (1-5)

Caveat we accept for v1: this reuses the LLM for alignment, so eval quality is
bounded by the aligner. That is the same canonicalization problem the full
pipeline exists to solve; here it is a controlled, single-pair instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import Example
from .llm_client import LLMClient
from .prompts import align_nodes_prompt, judge_edge_prompt, judge_prompt
from .schema import CausalGraph, EdgeValidityBatch, JudgeScore, NodeAlignment


@dataclass
class EvalReport:
    valid: bool
    precision: float
    recall: float
    f1: float
    matched: int
    n_pred: int
    n_gold: int
    judge_score: int
    judge_rationale: str
    alignment: dict
    matched_pred: set          # predicted (head, tail) that matched a gold edge
    recovered_gold: set        # gold (head, tail) that were recovered
    paper_precision: float     # fraction of predicted edges judged valid vs text+definition
    n_valid: int               # count of predicted edges judged causally valid


def structural_validity(graph: CausalGraph) -> bool:
    ids = set(graph.node_ids())
    return all(e.head in ids and e.tail in ids for e in graph.edges)


def _prf(tp_pred: int, tp_gold: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp_pred / n_pred if n_pred else 0.0   # fraction of predicted arrows that are correct
    r = tp_gold / n_gold if n_gold else 0.0   # fraction of gold arrows recovered
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(client: LLMClient, graph: CausalGraph, example: Example) -> EvalReport:
    valid = structural_validity(graph)

    align = client.complete(
        task="align_nodes",
        prompt=align_nodes_prompt(graph.node_ids(), example.gold_node_ids),
        schema=NodeAlignment, temperature=0.0,
    ).mapping

    gold = {(e["head"], e["tail"]) for e in example.gold_edges}
    matched_pred, recovered_gold = set(), set()
    for e in graph.edges:
        h, t = align.get(e.head), align.get(e.tail)
        if h is not None and t is not None and (h, t) in gold:
            matched_pred.add((e.head, e.tail))
            recovered_gold.add((h, t))
    matched = len(recovered_gold)
    p, r, f = _prf(len(matched_pred), len(recovered_gold), len(graph.edges), len(gold))

    judge = client.complete(
        task="judge",
        prompt=judge_prompt(example.text, graph.to_gold_like(), example.gold_edges),
        schema=JudgeScore, temperature=0.0,
    )

    # precision-by-paper: is each PREDICTED edge causally valid vs the text + the
    # causal-edge definition (independent of gold). Separates "extra edges that are
    # actually valid" (gold too strict) from "extra edges that are noise".
    n_valid = 0
    if graph.edges:
        pairs = [(e.head, e.tail) for e in graph.edges]
        vb = client.complete(
            task="judge_edges", prompt=judge_edge_prompt(example.text, pairs),
            schema=EdgeValidityBatch, temperature=0.0,
        )
        valid_idx = {v.index for v in vb.verdicts if v.valid}
        n_valid = sum(1 for i in range(len(pairs)) if i in valid_idx)
    paper_precision = n_valid / len(graph.edges) if graph.edges else 0.0

    return EvalReport(
        valid=valid, precision=p, recall=r, f1=f,
        matched=matched, n_pred=len(graph.edges), n_gold=len(gold),
        judge_score=judge.score, judge_rationale=judge.rationale, alignment=align,
        matched_pred=matched_pred, recovered_gold=recovered_gold,
        paper_precision=paper_precision, n_valid=n_valid,
    )
