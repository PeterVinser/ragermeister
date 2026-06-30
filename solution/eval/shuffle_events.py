"""Produce constraint-respecting random permutations of an event log.

The event log has hard ordering constraints:
  * A conflict event (implicates=[d1, d2, ...]) must arrive AFTER the first INSERT
    of every implicated doc_id — otherwise the KB has nothing to detect.
  * An UPDATE must arrive AFTER the first INSERT for the same doc_id.

A naive shuffle breaks these. This script builds the implied DAG and generates
valid random linearizations via Kahn's algorithm with shuffled ready-queues, then
re-numbers seq fields sequentially from 1.

Usage:
    python -m solution.eval.shuffle_events [OPTIONS]

Options:
    --input PATH     source JSONL  (default: data/extended_events_v3.jsonl)
    --output-dir DIR where to write shuffled files  (default: data/shuffled/)
    --n INT          number of variants to generate  (default: 5)
    --seed INT       base RNG seed (variant k uses seed+k; default: 42)
    --prefix STR     output filename prefix  (default: events_shuffled)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_dag(events: list[dict]) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Build a DAG of ordering constraints.

    Returns:
        edges:    seq -> [seqs that must come AFTER seq]
        in_degree: seq -> number of predecessors
    """
    seq_list = [e["seq"] for e in events]

    # Map doc_id -> first seq that introduces it
    first_seq_of: dict[str, int] = {}
    for e in events:
        doc = e["payload"]["doc_id"]
        if doc not in first_seq_of:
            first_seq_of[doc] = e["seq"]

    # Map doc_id -> the most recent seq seen so far (for UPDATE chaining)
    last_seq_of: dict[str, int] = {}

    edges: dict[int, list[int]] = {s: [] for s in seq_list}
    in_degree: dict[int, int] = {s: 0 for s in seq_list}

    def add_edge(pred: int, succ: int) -> None:
        if pred == succ:
            return
        edges[pred].append(succ)
        in_degree[succ] += 1

    for e in events:
        seq = e["seq"]
        doc = e["payload"]["doc_id"]
        event_type = e["type"]
        debug = e.get("_debug", {})

        # UPDATE: must come after the prior event for the same doc
        if event_type == "update" and doc in last_seq_of:
            add_edge(last_seq_of[doc], seq)

        # Conflict edges: every implicated doc's FIRST appearance must precede this event
        for imp_doc in debug.get("implicates", []):
            prior = first_seq_of.get(imp_doc)
            if prior is not None and prior != seq:
                add_edge(prior, seq)

        last_seq_of[doc] = seq

    return edges, in_degree


def random_topo_sort(
    events: list[dict],
    edges: dict[int, list[int]],
    in_degree: dict[int, int],
    rng: random.Random,
) -> list[dict]:
    """Kahn's algorithm with a shuffled ready-queue."""
    seq_to_event = {e["seq"]: e for e in events}
    degree = dict(in_degree)  # mutable copy

    ready = [seq for seq, d in degree.items() if d == 0]
    rng.shuffle(ready)
    ready = deque(ready)

    result: list[dict] = []
    while ready:
        seq = ready.popleft()
        result.append(seq_to_event[seq])
        successors = edges[seq][:]
        rng.shuffle(successors)
        for succ in successors:
            degree[succ] -= 1
            if degree[succ] == 0:
                # Insert at a random position in the ready queue for more mixing
                insert_pos = rng.randint(0, len(ready))
                ready.insert(insert_pos, succ)

    if len(result) != len(events):
        raise RuntimeError(
            f"DAG has a cycle or disconnected nodes: "
            f"sorted {len(result)}/{len(events)} events"
        )
    return result


def renumber(events: list[dict], start: int = 1) -> list[dict]:
    """Re-assign seq fields sequentially, preserving all other fields."""
    out = []
    for i, e in enumerate(events):
        new_e = dict(e)
        new_e["seq"] = start + i
        out.append(new_e)
    return out


def write_jsonl(events: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8",
    )


def shuffle_variants(
    input_path: Path,
    output_dir: Path,
    n: int,
    base_seed: int,
    prefix: str,
) -> None:
    events = load_events(input_path)
    print(f"Loaded {len(events)} events from {input_path}")

    edges, in_degree = build_dag(events)
    conflict_events = sum(
        1 for e in events
        if e.get("_debug", {}).get("expected_label", "clean") != "clean"
    )
    print(f"  {conflict_events} conflict events, {len(events) - conflict_events} clean")
    print(f"  DAG edges: {sum(len(v) for v in edges.values())}")

    for k in range(n):
        rng = random.Random(base_seed + k)
        shuffled = random_topo_sort(events, edges, in_degree, rng)
        shuffled = renumber(shuffled)

        out_path = output_dir / f"{prefix}_{k:02d}_seed{base_seed + k}.jsonl"
        write_jsonl(shuffled, out_path)
        print(f"  [{k+1}/{n}] Written {out_path.name}")

    print(f"\nDone. {n} variants written to {output_dir}/")


def _stats(events: list[dict]) -> None:
    """Print basic statistics about a shuffled file (useful for sanity-checking)."""
    seqs = [e["seq"] for e in events]
    labels = defaultdict(int)
    conflict_positions = []
    for i, e in enumerate(events):
        lbl = e.get("_debug", {}).get("expected_label", "clean")
        labels[lbl] += 1
        if lbl != "clean":
            conflict_positions.append(i)
    print(f"  Events: {len(events)}, seq range {seqs[0]}-{seqs[-1]}")
    print(f"  Labels: {dict(labels)}")
    if conflict_positions:
        print(
            f"  Conflict positions: first={conflict_positions[0]}, "
            f"last={conflict_positions[-1]}, "
            f"median={conflict_positions[len(conflict_positions) // 2]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/extended_events_v3.jsonl"),
        help="Source event log",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/shuffled"),
        help="Directory to write shuffled variants",
    )
    parser.add_argument("--n", type=int, default=5, help="Number of variants")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument(
        "--prefix", type=str, default="events_shuffled", help="Output filename prefix"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="After generation, print per-file position statistics",
    )
    args = parser.parse_args()

    shuffle_variants(args.input, args.output_dir, args.n, args.seed, args.prefix)

    if args.stats:
        print("\n--- Position statistics ---")
        for k in range(args.n):
            path = args.output_dir / f"{args.prefix}_{k:02d}_seed{args.seed + k}.jsonl"
            events = load_events(path)
            print(f"\n{path.name}:")
            _stats(events)


if __name__ == "__main__":
    main()
