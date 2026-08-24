"""Load TORQUESTRA and expose one example + its gold graph (labels stripped)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# The dataset is NOT vendored in this repo. Point TORQUESTRA_HUMAN_JSON at the file
# from https://github.com/fd-semantics/causal-schema-public (data/), or clone that
# repo next to this one so the default relative path resolves.
_DEFAULT_PATH = Path(
    os.environ.get(
        "TORQUESTRA_HUMAN_JSON",
        Path(__file__).resolve().parents[2]
        / "causal-schema-public" / "data" / "torquestra-human-2023-02-23.json",
    )
)

# The example we validate on (has both a short train and long dev gold graph).
DEMO_TORQUE_ID = "docid_PRI19980115.2000.0186_sentid_6"


@dataclass
class Example:
    torque_id: str
    split: str
    text: str
    gold_edges: list[dict] = field(default_factory=list)   # [{head, tail}] labels dropped
    gold_node_ids: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def _gold_from(raw: dict) -> tuple[list[dict], list[str]]:
    edges, nodes = [], set()
    for e in raw.get("causal_graph", []):
        edges.append({"head": e["head"], "tail": e["tail"]})
        nodes.add(e["head"])
        nodes.add(e["tail"])
    et = raw.get("event_types")
    if isinstance(et, dict):  # train: node-level typing gives us the node inventory
        nodes.update(et.keys())
    return edges, sorted(nodes)


def load(path: Path | str | None = None) -> list[dict]:
    return json.loads(Path(path or _DEFAULT_PATH).read_text())


def get_example(torque_id: str = DEMO_TORQUE_ID, split: str = "train",
                path: Path | str | None = None) -> Example:
    for raw in load(path):
        if raw.get("torque_id") == torque_id and raw.get("split") == split:
            gold_edges, gold_nodes = _gold_from(raw)
            return Example(
                torque_id=torque_id,
                split=split,
                text=raw["text"].strip(),
                gold_edges=gold_edges,
                gold_node_ids=gold_nodes,
                raw=raw,
            )
    raise KeyError(f"No example with torque_id={torque_id!r} split={split!r}")
