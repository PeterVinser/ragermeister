"""Side-by-side audit of vector-only, metadata-only, graph-only, and hybrid baselines.

The research question is *candidate discovery*: all baselines feed the SAME conflict
judge and resolution policy; they differ only in how they surface candidate neighbours.
To isolate that variable the judge is an OracleJudge (gold-conditioned, no LLM calls).

Primary output is a raw JSONL event log — one record per (shuffle × baseline × event) —
from which any metric can be computed later. Summary statistics are derived from that log
and printed to stdout; a compact JSON summary is written alongside the JSONL.

Shuffle mode (--shuffles N): generates N constraint-respecting random permutations of the
event log (implicates must precede the event that references them), then runs all baselines
on each permutation in parallel (one thread per baseline).

Usage:
    python -m solution.eval.comparison                          # original order, full audit
    python -m solution.eval.comparison --shuffles 5 --no-audit # log + summary only
    python -m solution.eval.comparison --shuffles 5 --resolve  # recency resolution
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from solution.models.chunk import Chunk
from solution.models.conflict import ConflictLabel, JudgeResult
from solution.models.entity import EntityMention, EntityType
from solution.models.event import EventType, IngestEvent
from solution.services.docstore import Docstore
from solution.services.embedder import Embedder
from solution.services.entity_candidates_judge import EntityCandidatesJudge
from solution.services.entity_extractor import EntityExtractor, LLMEntityExtractor
from solution.services.entity_resolver import EntityResolver
from solution.services.event_log import EventLog
from solution.services.graph_house_keeper import GraphHouseKeeper
from solution.services.hybrid_house_keeper import HybridHouseKeeper
from solution.services.knowledge_base import KnowledgeBase
from solution.services.metadata_house_keeper import MetadataHouseKeeper
from solution.services.vector_house_keeper import VectorHouseKeeper
from solution.services.policies.base import ResolutionPolicy
from solution.services.policies.recency import RecencyPolicy
from solution.services.resolution_manager import ResolutionManager
from solution.services.vector_db import VectorDB
from solution.models.conflict import ConflictReport, Decision, DecisionAction

_DEFAULT_EVENTS = Path(__file__).resolve().parents[2] / "data" / "extended_events_v3.jsonl"

_LABEL_MAP = {
    "clean": ConflictLabel.CLEAN,
    "duplicate": ConflictLabel.DUPLICATE,
    "contradiction": ConflictLabel.CONTRADICTION,
    "supersedes": ConflictLabel.SUPERSEDES,
    "needs_human": ConflictLabel.NEEDS_HUMAN,
}

# --------------------------------------------------------------------------- dataset


@dataclass
class GoldEvent:
    seq: int
    event_type: EventType
    doc_id: str
    text: str
    source_id: str | None
    title: str
    is_artifact: bool
    derived_from: list[str]
    expected_label: ConflictLabel
    implicates: list[str]
    intent: str
    note: str

    def to_ingest(self) -> IngestEvent:
        return IngestEvent(
            event_type=self.event_type,
            doc_id=self.doc_id,
            text=self.text,
            metadata={
                "source_id": self.source_id,
                "title": self.title,
                "is_artifact": self.is_artifact,
                "derived_from": self.derived_from,
            },
        )


def load_events(path: Path = _DEFAULT_EVENTS) -> list[GoldEvent]:
    events: list[GoldEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        p = rec["payload"]
        d = rec.get("_debug", {})
        events.append(
            GoldEvent(
                seq=rec["seq"],
                event_type=EventType(rec["type"]),
                doc_id=p["doc_id"],
                text=p.get("text", ""),
                source_id=p.get("source_id"),
                title=p.get("title", ""),
                is_artifact=bool(p.get("is_artifact", False)),
                derived_from=list(p.get("derived_from", [])),
                expected_label=_LABEL_MAP[d.get("expected_label", "clean")],
                implicates=list(d.get("implicates", [])),
                intent=d.get("intent", ""),
                note=d.get("note", ""),
            )
        )
    return events


# --------------------------------------------------------------------------- shuffling


def _build_dag(events: list[GoldEvent]) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Ordering constraints as a DAG.

    For each conflict event: every implicated doc's FIRST appearance must precede it.
    For each UPDATE: the prior INSERT for the same doc must precede it.
    """
    first_seq_of: dict[str, int] = {}
    for e in events:
        if e.doc_id not in first_seq_of:
            first_seq_of[e.doc_id] = e.seq

    last_seq_of: dict[str, int] = {}
    edges: dict[int, list[int]] = {e.seq: [] for e in events}
    in_degree: dict[int, int] = {e.seq: 0 for e in events}

    def add_edge(pred: int, succ: int) -> None:
        if pred == succ:
            return
        edges[pred].append(succ)
        in_degree[succ] += 1

    for e in events:
        if e.event_type == EventType.UPDATE and e.doc_id in last_seq_of:
            add_edge(last_seq_of[e.doc_id], e.seq)
        for imp_doc in e.implicates:
            prior = first_seq_of.get(imp_doc)
            if prior is not None and prior != e.seq:
                add_edge(prior, e.seq)
        last_seq_of[e.doc_id] = e.seq

    return edges, in_degree


def _random_topo_sort(
    events: list[GoldEvent],
    edges: dict[int, list[int]],
    in_degree: dict[int, int],
    rng: random.Random,
) -> list[GoldEvent]:
    seq_to_event = {e.seq: e for e in events}
    degree = dict(in_degree)
    ready = deque(sorted([s for s, d in degree.items() if d == 0], key=lambda _: rng.random()))
    result: list[GoldEvent] = []
    while ready:
        seq = ready.popleft()
        result.append(seq_to_event[seq])
        for succ in edges[seq]:
            degree[succ] -= 1
            if degree[succ] == 0:
                ready.insert(rng.randint(0, len(ready)), succ)
    if len(result) != len(events):
        raise RuntimeError(f"DAG cycle: sorted {len(result)}/{len(events)}")
    return result


def generate_shuffles(
    events: list[GoldEvent], n: int, base_seed: int
) -> list[tuple[int, list[GoldEvent]]]:
    """Return [(seed, shuffled_and_renumbered_events), ...] for n variants."""
    import copy
    edges, in_degree = _build_dag(events)
    variants = []
    for k in range(n):
        seed = base_seed + k
        shuffled = _random_topo_sort(events, edges, in_degree, random.Random(seed))
        # Re-assign seq 1..N so position == seq in the record
        renumbered = []
        for i, e in enumerate(shuffled):
            new_e = copy.copy(e)
            new_e.seq = i + 1
            renumbered.append(new_e)
        variants.append((seed, renumbered))
    return variants


# --------------------------------------------------------------------------- judge


class OracleJudge:
    """Gold-conditioned perfect judge: returns the gold label iff discovery surfaced
    at least one implicated doc. Verdict differences across baselines are purely due
    to candidate discovery (invariant #3)."""

    def __init__(self) -> None:
        self.current: GoldEvent | None = None

    def judge(self, new_chunk: Chunk, neighbors: list[Chunk]) -> JudgeResult:
        gold = self.current
        if gold is None or gold.expected_label == ConflictLabel.CLEAN:
            return JudgeResult(label=ConflictLabel.CLEAN, implicated_ids=[], proposed_action="insert", rationale="clean (gold)")
        relevant = [n for n in neighbors if n.doc_id in gold.implicates]
        if not relevant:
            return JudgeResult(
                label=ConflictLabel.CLEAN, implicated_ids=[], proposed_action="insert",
                rationale="discovery surfaced no implicated doc → undetected",
            )
        return JudgeResult(
            label=gold.expected_label,
            implicated_ids=[n.chunk_id for n in relevant],
            proposed_action="replace_old",
            rationale=f"gold {gold.expected_label.value}",
        )


@dataclass
class JudgeCall:
    seq: int
    new_doc: str
    candidate_docs: list[str]
    label: ConflictLabel
    implicated_docs: list[str]


class AuditingJudge:
    """Transparent wrapper that records every call for one baseline's run."""

    def __init__(self, inner: OracleJudge, docstore: Docstore) -> None:
        self._inner = inner
        self._docstore = docstore
        self.current_seq = 0
        self.calls: list[JudgeCall] = []

    def judge(self, new_chunk: Chunk, neighbors: list[Chunk]) -> JudgeResult:
        result = self._inner.judge(new_chunk, neighbors)
        implicated_docs = sorted(
            {c.doc_id for cid in result.implicated_ids if (c := self._docstore._chunks.get(cid)) is not None}
        )
        self.calls.append(JudgeCall(
            seq=self.current_seq,
            new_doc=new_chunk.doc_id,
            candidate_docs=sorted({n.doc_id for n in neighbors}),
            label=result.label,
            implicated_docs=implicated_docs,
        ))
        return result


# --------------------------------------------------------------------------- extractors / policy


class CuratedEntityExtractor(EntityExtractor):
    """Rule-based NER stand-in for the university dataset — symmetric 'competent input'
    for structural baselines, paralleling real embeddings for vector-only."""

    _PERSONS: dict[str, tuple[list[str], str]] = {
        "Elena Vance": (["elena vance", "vance"], "dean of admissions"),
        "Marcus Reed": (["marcus reed", "reed"], "admissions staff member"),
        "Priya Sharma": (["priya sharma", "sharma"], "cs department chair"),
        "James Whitfield": (["james whitfield", "whitfield"], "biology chair"),
        "Robert Kim": (["robert kim"], "dean of engineering"),
        "Lena Fischer": (["lena fischer", "fischer"], "physics associate professor"),
        "Michael Torres": (["michael torres", "torres"], "head basketball coach"),
        "Amara Okafor": (["amara okafor", "okafor"], "basketball coach"),
        "Diana Chen": (["diana chen"], "basketball coach"),
        "President Hargrove": (["hargrove", "president hargrove"], "university president"),
    }
    _ORGS: dict[str, tuple[list[str], str]] = {
        "Admissions Office": (
            ["dean of admissions", "admissions office", "admissions leadership",
             "lead northmoor", "admissions head"],
            "the admissions office",
        ),
        "Northmoor University": (["northmoor"], "northmoor university campus"),
        "Hartwell Library": (["hartwell library"], "hartwell library building"),
        "Computer Science Department": (["computer science department", "cs department"], "cs dept"),
        "Biology Department": (["biology department", "department of biology"], "biology dept"),
        "School of Engineering": (["school of engineering", "engineering school"], "engineering"),
        "Physics Department": (["physics department"], "physics dept"),
    }
    _TOPICS: dict[str, list[str]] = {
        "application-deadline": ["application deadline", "deadline", "apply by", "autumn intake",
                                  "submit their application", "applying to northmoor"],
        "application-fee": ["application fee", "application charge", "$50", "$75",
                            "fee remains", "waive the application fee", "fee entirely"],
        "library-hours": ["library", "operates from"],
        "welcome-seminar": ["welcome seminar", "seminar"],
        "tuition": ["tuition", "credit hour", "enrollment fee"],
        "scholarship": ["scholarship", "merit award", "gpa"],
        "financial-aid": ["financial aid", "fafsa", "need-based"],
        "shuttle": ["shuttle", "blue line", "campus transit"],
        "gym": ["fitness center", "henderson", "gym"],
        "dining": ["dining", "commons", "birchwood"],
    }
    _DATES: dict[str, list[str]] = {
        "March 1": ["march 1", "first of march"],
        "April 1": ["april 1"],
        "May 1": ["may 1"],
    }

    def extract(self, chunk: Chunk) -> list[EntityMention]:
        low = chunk.text.lower()
        mentions: list[EntityMention] = []
        for name, (pats, context) in self._PERSONS.items():
            if any(p in low for p in pats):
                mentions.append(EntityMention(surface_form=name, type=EntityType.PERSON, short_context=context))
        for name, (pats, context) in self._ORGS.items():
            if any(p in low for p in pats):
                mentions.append(EntityMention(surface_form=name, type=EntityType.ORG, short_context=context))
        for _slug, pats in self._TOPICS.items():
            matched = next((p for p in pats if p in low), None)
            if matched is not None:
                mentions.append(EntityMention(surface_form=matched, type=EntityType.TOPIC, short_context=""))
        for display, pats in self._DATES.items():
            if any(p in low for p in pats):
                mentions.append(EntityMention(surface_form=display, type=EntityType.DATE, short_context=""))
        return mentions


class InsertOnlyPolicy(ResolutionPolicy):
    """Non-destructive: records detection signal but commits without removing implicated
    docs — isolates candidate discovery from resolution effects."""

    def resolve(self, report: ConflictReport, apply_decision: Callable[[Decision], None]) -> None:
        apply_decision(Decision(
            report_id=report.report_id, action=DecisionAction.INSERT,
            new_chunk=report.new_chunk, chunk_ids_to_remove=[],
        ))


# --------------------------------------------------------------------------- runners


@dataclass
class BaselineRun:
    name: str
    calls: list[JudgeCall]
    final_docs: list[str]
    per_seq: dict[int, JudgeCall] = field(default_factory=dict)


def _aggregate_calls(calls: list[JudgeCall]) -> dict[int, JudgeCall]:
    agg: dict[int, JudgeCall] = {}
    for c in calls:
        cur = agg.get(c.seq)
        if cur is None:
            agg[c.seq] = JudgeCall(c.seq, c.new_doc, list(c.candidate_docs), c.label, list(c.implicated_docs))
            continue
        cur.candidate_docs = sorted(set(cur.candidate_docs) | set(c.candidate_docs))
        cur.implicated_docs = sorted(set(cur.implicated_docs) | set(c.implicated_docs))
        if cur.label == ConflictLabel.CLEAN and c.label != ConflictLabel.CLEAN:
            cur.label = c.label
    return agg


def _active_doc_count(docstore: Docstore) -> int:
    return sum(1 for chunks in docstore._doc_chunks.values() if chunks)


def _build_record(
    run_id: str,
    shuffle_idx: int,
    seed: int | None,
    baseline: str,
    position: int,
    ev: GoldEvent,
    calls: list[JudgeCall],
    corpus_size: int,
    graph_snapshot: dict | None,
) -> dict:
    """Build one raw log record for (shuffle × baseline × event).

    A record is self-contained: every field needed to compute any metric or
    reconstruct the detection timeline is included. graph_snapshot is null for
    non-graph baselines.
    """
    candidate_docs: list[str] = []
    verdict: str | None = None
    implicated_found: list[str] = []

    for call in calls:
        candidate_docs = sorted(set(candidate_docs) | set(call.candidate_docs))
        if call.label != ConflictLabel.CLEAN:
            verdict = call.label.value
        implicated_found = sorted(set(implicated_found) | set(call.implicated_docs))

    if verdict is None and calls:
        verdict = ConflictLabel.CLEAN.value

    detected = bool(
        verdict and verdict != ConflictLabel.CLEAN.value
        and bool(set(ev.implicates) & set(implicated_found))
    )

    return {
        "run_id": run_id,
        "shuffle_idx": shuffle_idx,
        "seed": seed,
        "baseline": baseline,
        "position": position,
        "seq": ev.seq,
        "doc_id": ev.doc_id,
        "event_type": ev.event_type.value,
        "intent": ev.intent,
        "expected_label": ev.expected_label.value,
        "implicates": ev.implicates,
        "n_candidates": len(candidate_docs),
        "candidate_docs": candidate_docs,
        "verdict": verdict,
        "implicated_found": implicated_found,
        "n_implicated_found": len(implicated_found),
        "detected": detected,
        "corpus_size": corpus_size,
        "graph": graph_snapshot,
    }


def run_vector_only(
    events: list[GoldEvent],
    oracle: OracleJudge,
    dim: int,
    policy_factory: Callable[[], ResolutionPolicy],
    records: list[dict] | None = None,
    run_id: str = "",
    shuffle_idx: int = -1,
    seed: int | None = None,
) -> BaselineRun:
    vdb, docstore = VectorDB(dim), Docstore()
    auditor = AuditingJudge(oracle, docstore)
    housekeeper = VectorHouseKeeper(vdb, docstore)
    kb = KnowledgeBase(vdb, docstore, auditor, housekeeper)  # type: ignore[arg-type]
    manager = ResolutionManager(policy_factory(), kb.apply_decision)
    kb.conflict_sink = manager.submit
    for position, ev in enumerate(events):
        n_before = len(auditor.calls)
        oracle.current = ev
        auditor.current_seq = ev.seq
        kb.ingest(ev.to_ingest())
        if records is not None:
            records.append(_build_record(
                run_id, shuffle_idx, seed, "vector-only", position, ev,
                auditor.calls[n_before:], _active_doc_count(docstore), None,
            ))
    final = sorted(d for d in docstore._doc_chunks if docstore._doc_chunks[d])
    return BaselineRun("vector-only", auditor.calls, final, _aggregate_calls(auditor.calls))


def run_graph_only(
    events: list[GoldEvent],
    oracle: OracleJudge,
    dim: int,
    extractor: EntityExtractor,
    embedder: Embedder,
    policy_factory: Callable[[], ResolutionPolicy],
    topic_vocab: dict[str, list[str]] | None = None,
    log_id: str | None = None,
    verbose: bool = True,
    records: list[dict] | None = None,
    run_id: str = "",
    shuffle_idx: int = -1,
    seed: int | None = None,
) -> BaselineRun:
    vdb, docstore = VectorDB(dim), Docstore()
    auditor = AuditingJudge(oracle, docstore)
    resolver_log = EventLog(path=f"graph/resolver/{log_id}.jsonl") if log_id else EventLog()
    hk_log = EventLog(path=f"graph/housekeeper/{log_id}.jsonl") if log_id else EventLog()
    resolver = EntityResolver(
        extractor=extractor, adjudicator=EntityCandidatesJudge(),
        embedder=embedder, event_log=resolver_log, topic_vocab=topic_vocab,
    )
    housekeeper = GraphHouseKeeper(
        vector_db=vdb, docstore=docstore, resolver=resolver, event_log=hk_log,
    )
    kb = KnowledgeBase(vdb, docstore, auditor, housekeeper)  # type: ignore[arg-type]
    manager = ResolutionManager(policy_factory(), kb.apply_decision)
    kb.conflict_sink = manager.submit
    for position, ev in enumerate(events):
        n_before = len(auditor.calls)
        oracle.current = ev
        auditor.current_seq = ev.seq
        kb.ingest(ev.to_ingest())
        if records is not None:
            records.append(_build_record(
                run_id, shuffle_idx, seed, "graph-only", position, ev,
                auditor.calls[n_before:], _active_doc_count(docstore),
                housekeeper._graph.snapshot(),
            ))
    final = sorted(d for d in docstore._doc_chunks if docstore._doc_chunks[d])
    if verbose:
        _print_alias_clusters(resolver)
    return BaselineRun("graph-only", auditor.calls, final, _aggregate_calls(auditor.calls))


def run_metadata_only(
    events: list[GoldEvent],
    oracle: OracleJudge,
    dim: int,
    extractor: EntityExtractor,
    embedder: Embedder,
    policy_factory: Callable[[], ResolutionPolicy],
    topic_vocab: dict[str, list[str]] | None = None,
    records: list[dict] | None = None,
    run_id: str = "",
    shuffle_idx: int = -1,
    seed: int | None = None,
) -> BaselineRun:
    vdb, docstore = VectorDB(dim), Docstore()
    auditor = AuditingJudge(oracle, docstore)
    resolver = EntityResolver(
        extractor=extractor, adjudicator=EntityCandidatesJudge(),
        embedder=embedder, topic_vocab=topic_vocab,
    )
    housekeeper = MetadataHouseKeeper(docstore=docstore, resolver=resolver)
    kb = KnowledgeBase(vdb, docstore, auditor, housekeeper)  # type: ignore[arg-type]
    manager = ResolutionManager(policy_factory(), kb.apply_decision)
    kb.conflict_sink = manager.submit
    for position, ev in enumerate(events):
        n_before = len(auditor.calls)
        oracle.current = ev
        auditor.current_seq = ev.seq
        kb.ingest(ev.to_ingest())
        if records is not None:
            records.append(_build_record(
                run_id, shuffle_idx, seed, "metadata-only", position, ev,
                auditor.calls[n_before:], _active_doc_count(docstore), None,
            ))
    final = sorted(d for d in docstore._doc_chunks if docstore._doc_chunks[d])
    return BaselineRun("metadata-only", auditor.calls, final, _aggregate_calls(auditor.calls))


def _make_resolver(
    extractor: EntityExtractor, embedder: Embedder, topic_vocab: dict[str, list[str]] | None
) -> EntityResolver:
    return EntityResolver(
        extractor=extractor, adjudicator=EntityCandidatesJudge(),
        embedder=embedder, topic_vocab=topic_vocab,
    )


def run_hybrid(
    events: list[GoldEvent],
    oracle: OracleJudge,
    dim: int,
    extractor: EntityExtractor,
    embedder: Embedder,
    policy_factory: Callable[[], ResolutionPolicy],
    topic_vocab: dict[str, list[str]] | None = None,
    records: list[dict] | None = None,
    run_id: str = "",
    shuffle_idx: int = -1,
    seed: int | None = None,
) -> BaselineRun:
    vdb, docstore = VectorDB(dim), Docstore()
    auditor = AuditingJudge(oracle, docstore)
    housekeeper = HybridHouseKeeper(
        vector_db=vdb, docstore=docstore,
        resolver=_make_resolver(extractor, embedder, topic_vocab),
    )
    kb = KnowledgeBase(vdb, docstore, auditor, housekeeper)  # type: ignore[arg-type]
    manager = ResolutionManager(policy_factory(), kb.apply_decision)
    kb.conflict_sink = manager.submit
    for position, ev in enumerate(events):
        n_before = len(auditor.calls)
        oracle.current = ev
        auditor.current_seq = ev.seq
        kb.ingest(ev.to_ingest())
        if records is not None:
            records.append(_build_record(
                run_id, shuffle_idx, seed, "hybrid", position, ev,
                auditor.calls[n_before:], _active_doc_count(docstore),
                housekeeper._graph.snapshot(),
            ))
    final = sorted(d for d in docstore._doc_chunks if docstore._doc_chunks[d])
    return BaselineRun("hybrid", auditor.calls, final, _aggregate_calls(auditor.calls))


# --------------------------------------------------------------------------- parallel dispatch


def _run_all_baselines_parallel(
    events: list[GoldEvent],
    dim: int,
    extractor: EntityExtractor,
    embedder: Embedder,
    policy_factory: Callable[[], ResolutionPolicy],
    topic_vocab: dict[str, list[str]] | None,
    run_id: str,
    shuffle_idx: int,
    seed: int | None,
) -> tuple[list[BaselineRun], list[dict]]:
    """Run all 4 baselines concurrently on the same event list.

    Each baseline gets its own OracleJudge instance (oracle.current is mutated per-event
    inside the loop, so sharing one across threads would race). Each also builds its own
    records list; we merge them at the end. Returns (runs, all_records).
    """
    def _vec() -> tuple[BaselineRun, list[dict]]:
        rec: list[dict] = []
        run = run_vector_only(events, OracleJudge(), dim, policy_factory, rec, run_id, shuffle_idx, seed)
        return run, rec

    def _meta() -> tuple[BaselineRun, list[dict]]:
        rec: list[dict] = []
        run = run_metadata_only(events, OracleJudge(), dim, extractor, embedder, policy_factory, topic_vocab, rec, run_id, shuffle_idx, seed)
        return run, rec

    def _graph() -> tuple[BaselineRun, list[dict]]:
        rec: list[dict] = []
        run = run_graph_only(events, OracleJudge(), dim, extractor, embedder, policy_factory, topic_vocab, log_id=None, verbose=False, records=rec, run_id=run_id, shuffle_idx=shuffle_idx, seed=seed)
        return run, rec

    def _hybrid() -> tuple[BaselineRun, list[dict]]:
        rec: list[dict] = []
        run = run_hybrid(events, OracleJudge(), dim, extractor, embedder, policy_factory, topic_vocab, rec, run_id, shuffle_idx, seed)
        return run, rec

    tasks = {"vector-only": _vec, "metadata-only": _meta, "graph-only": _graph, "hybrid": _hybrid}
    ordered = ("vector-only", "metadata-only", "graph-only", "hybrid")

    results: dict[str, tuple[BaselineRun, list[dict]]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as exc:
                print(f"  [shuffle {shuffle_idx}] {name} FAILED: {exc}")
                raise

    runs = [results[n][0] for n in ordered if n in results]
    all_records: list[dict] = []
    for n in ordered:
        if n in results:
            all_records.extend(results[n][1])
    return runs, all_records


# --------------------------------------------------------------------------- metrics from log


def summarize_records(records: list[dict]) -> dict[str, dict]:
    """Derive summary metrics from raw log records. Groups by baseline.

    Separates conflict events from clean controls and reports:
      - detect_rate, affected_recall
      - false-positive candidate pressure on clean controls
      - per-label breakdown (contradiction / supersedes / duplicate)
      - detection-rate at corpus quartiles (temporal trend)

    Returns a nested dict keyed by baseline name.
    """
    by_baseline: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_baseline[r["baseline"]].append(r)

    summary: dict[str, dict] = {}
    for baseline, recs in by_baseline.items():
        conflicts = [r for r in recs if r["expected_label"] != "clean"]
        controls = [r for r in recs if r["expected_label"] == "clean"]

        detected = sum(1 for r in conflicts if r["detected"])
        recall_num = sum(r["n_implicated_found"] for r in conflicts)
        recall_den = sum(len(r["implicates"]) for r in conflicts)

        by_label: dict[str, dict] = {}
        for r in conflicts:
            lbl = r["expected_label"]
            if lbl not in by_label:
                by_label[lbl] = {"total": 0, "detected": 0}
            by_label[lbl]["total"] += 1
            if r["detected"]:
                by_label[lbl]["detected"] += 1
        for v in by_label.values():
            v["rate"] = round(v["detected"] / v["total"], 4) if v["total"] else 0.0

        fp_counts = [r["n_candidates"] for r in controls]
        fp_mean = statistics.mean(fp_counts) if fp_counts else 0.0

        # Temporal trend: detection rate in each corpus-size quartile
        max_pos = max((r["position"] for r in conflicts), default=0)
        quartile_size = max(max_pos // 4, 1)
        temporal: dict[str, dict] = {}
        for q in range(4):
            lo, hi = q * quartile_size, (q + 1) * quartile_size if q < 3 else max_pos + 1
            bucket = [r for r in conflicts if lo <= r["position"] < hi]
            label = f"Q{q+1}_pos_{lo}-{hi}"
            temporal[label] = {
                "n_conflicts": len(bucket),
                "detected": sum(1 for r in bucket if r["detected"]),
                "rate": round(sum(1 for r in bucket if r["detected"]) / len(bucket), 4) if bucket else 0.0,
            }

        summary[baseline] = {
            "n_conflict_events": len(conflicts),
            "n_clean_events": len(controls),
            "detect_rate": round(detected / len(conflicts), 4) if conflicts else 0.0,
            "detected_count": detected,
            "affected_recall": round(recall_num / recall_den, 4) if recall_den else 0.0,
            "fp_pressure_mean": round(fp_mean, 4),
            "fp_pressure_total": sum(fp_counts),
            "by_label": by_label,
            "temporal_detection": temporal,
        }
    return summary


def aggregate_summaries(per_shuffle: list[dict[str, dict]]) -> dict[str, dict]:
    """Fold N per-shuffle summaries into mean ± std / min / max per baseline."""
    if not per_shuffle:
        return {}
    baseline_names = list(per_shuffle[0].keys())
    scalar_keys = ["detect_rate", "affected_recall", "fp_pressure_mean"]

    agg: dict[str, dict] = {}
    for name in baseline_names:
        runs = [s[name] for s in per_shuffle if name in s]
        entry: dict = {}
        for key in scalar_keys:
            vals = [r[key] for r in runs if key in r]
            if not vals:
                continue
            entry[key] = {
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }
        all_labels = sorted({lbl for r in runs for lbl in r.get("by_label", {})})
        entry["by_label"] = {}
        for lbl in all_labels:
            rates = [r["by_label"][lbl]["rate"] for r in runs if lbl in r.get("by_label", {})]
            if rates:
                entry["by_label"][lbl] = {
                    "detect_rate_mean": round(statistics.mean(rates), 4),
                    "detect_rate_std": round(statistics.stdev(rates) if len(rates) > 1 else 0.0, 4),
                }
        agg[name] = entry
    return agg


# --------------------------------------------------------------------------- persistence


def write_logs(
    run_id: str,
    meta: dict,
    all_records: list[dict],
    per_shuffle_summaries: list[dict[str, dict]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the raw JSONL event log and a compact JSON summary. Returns (jsonl_path, summary_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"run_{run_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_path = output_dir / f"run_{run_id}_summary.json"
    summary = {
        "meta": meta,
        "per_shuffle": [
            {"shuffle_idx": i, "seed": meta["base_seed"] + i, "metrics": s}
            for i, s in enumerate(per_shuffle_summaries)
        ],
        "aggregate": aggregate_summaries(per_shuffle_summaries),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return jsonl_path, summary_path


# --------------------------------------------------------------------------- reporting


def _fmt(items: list[str], width: int = 34) -> str:
    s = ", ".join(items) if items else "-"
    return s if len(s) <= width else s[: width - 1] + "~"


def print_audit(events: list[GoldEvent], runs: list[BaselineRun]) -> None:
    name_w = max(len(r.name) for r in runs)
    print("=" * 100)
    print("PER-EVENT AUDIT  (candidates = docs surfaced by discovery; OK = conflict detected)")
    print("=" * 100)
    for ev in events:
        gold_conf = ev.expected_label != ConflictLabel.CLEAN
        print(f"\n[seq {ev.seq}] {ev.event_type.value.upper():6} doc={ev.doc_id!r}  intent={ev.intent}")
        print(f"    gold: {ev.expected_label.value:13} implicates={_fmt(ev.implicates, 50)}")
        for run in runs:
            call = run.per_seq.get(ev.seq)
            if call is None:
                print(f"    {run.name:<{name_w}}: (no judge call)")
                continue
            detected = call.label != ConflictLabel.CLEAN
            mark = "OK  " if detected == gold_conf and (
                not gold_conf or set(ev.implicates) & set(call.implicated_docs)
            ) else "MISS"
            print(f"    {run.name:<{name_w}}: cand=[{_fmt(call.candidate_docs)}] verdict={call.label.value:13} {mark}")
    _print_summary_from_runs(events, runs)


def _print_summary_from_runs(events: list[GoldEvent], runs: list[BaselineRun]) -> None:
    conflicts = [e for e in events if e.expected_label != ConflictLabel.CLEAN]
    controls = [e for e in events if e.expected_label == ConflictLabel.CLEAN]
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nConflict events: {len(conflicts)}  |  Clean controls: {len(controls)}")
    print(f"\n{'baseline':16} {'detected':>10} {'detect-rate':>12} {'affected-recall':>16}")
    print("-" * 58)
    for run in runs:
        det = recall_n = recall_d = 0
        for e in conflicts:
            call = run.per_seq.get(e.seq)
            cand = set(call.candidate_docs) if call else set()
            found = set(e.implicates) & cand
            recall_n += len(found)
            recall_d += len(e.implicates)
            if call and call.label != ConflictLabel.CLEAN and found:
                det += 1
        dr = det / len(conflicts) if conflicts else 0.0
        ar = recall_n / recall_d if recall_d else 0.0
        print(f"{run.name:16} {det:>5}/{len(conflicts):<4} {dr:>11.0%} {ar:>15.0%}")

    col_w = max(6, max(len(r.name) for r in runs))
    print(f"\nPer-conflict detection:")
    header = f"{'seq':>4} {'doc':20} {'label':14} " + " ".join(f"{r.name:>{col_w}}" for r in runs)
    print(header)
    print("-" * len(header))
    for e in conflicts:
        marks = " ".join(
            f"{('OK' if (c := r.per_seq.get(e.seq)) and c.label != ConflictLabel.CLEAN and (set(e.implicates) & set(c.implicated_docs)) else 'MISS'):>{col_w}}"
            for r in runs
        )
        print(f"{e.seq:>4} {e.doc_id:20} {e.expected_label.value:14} {marks}")


def _print_aggregate_summary(agg: dict) -> None:
    baseline_names = list(agg.keys())
    metrics_keys = ["detect_rate", "affected_recall", "fp_pressure_mean"]
    print("\n" + "=" * 100)
    print("AGGREGATE ACROSS SHUFFLES  (mean ± std  [min – max])")
    print("=" * 100)
    col_w = max(14, max(len(n) for n in baseline_names))
    for m in metrics_keys:
        print(f"\n  {m}:")
        for name in baseline_names:
            v = agg.get(name, {}).get(m, {})
            if v:
                print(f"    {name:<{col_w}}  {v['mean']:.1%} ± {v['std']:.1%}  [{v['min']:.0%} – {v['max']:.0%}]")

    all_labels = sorted({lbl for e in agg.values() for lbl in e.get("by_label", {})})
    if all_labels:
        print(f"\n  detect-rate by label (mean ± std):")
        for lbl in all_labels:
            print(f"    {lbl}:")
            for name in baseline_names:
                v = agg.get(name, {}).get("by_label", {}).get(lbl, {})
                if v:
                    print(f"      {name:<{col_w}}  {v['detect_rate_mean']:.1%} ± {v['detect_rate_std']:.1%}")


def _print_alias_clusters(resolver: EntityResolver) -> None:
    clusters = resolver.alias_clusters()
    print("\n" + "-" * 60)
    print(f"ENTITY RESOLUTION — {len(clusters)} canonical entities")
    print("-" * 60)
    for cid, info in sorted(clusters.items()):
        print(f"  {cid:14} [{info['type']:6}] {info['canonical_name']:22} aliases={info['aliases']}")


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_DEFAULT_EVENTS)
    parser.add_argument(
        "--shuffles", type=int, default=5,
        help="Random permutations to run (0 = original order only)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"),
        help="Directory for JSONL event log and JSON summary",
    )
    parser.add_argument("--llm-extractor", action="store_true")
    parser.add_argument("--resolve", action="store_true", help="Use recency resolution instead of insert-only")
    parser.add_argument("--no-audit", action="store_true", help="Skip per-event audit table")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    events = load_events(args.input)
    print(f"Loaded {len(events)} events from {args.input}")

    embedder = Embedder()
    dim = int(embedder.embed(["dimension probe"]).shape[1])
    print(f"Embedding dim = {dim}")

    extractor: EntityExtractor = LLMEntityExtractor() if args.llm_extractor else CuratedEntityExtractor()
    topic_vocab = None if args.llm_extractor else CuratedEntityExtractor._TOPICS

    policy_factory: Callable[[], ResolutionPolicy]
    if args.resolve:
        policy_factory = RecencyPolicy
        mode = "recency (last-write-wins)"
    else:
        policy_factory = InsertOnlyPolicy
        mode = "insert-only (isolates candidate discovery)"

    print(f"Extractor : {type(extractor).__name__}")
    print(f"Resolution: {mode}")

    conflicts = [e for e in events if e.expected_label != ConflictLabel.CLEAN]
    label_counts: dict[str, int] = defaultdict(int)
    for e in conflicts:
        label_counts[e.expected_label.value] += 1
    meta = {
        "run_id": run_id,
        "input_file": str(args.input),
        "n_shuffles": max(args.shuffles, 1),
        "base_seed": args.seed,
        "use_llm_extractor": args.llm_extractor,
        "resolve_mode": mode,
        "timestamp": datetime.now().isoformat(),
        "total_events": len(events),
        "conflict_events": len(conflicts),
        "clean_events": len(events) - len(conflicts),
        "label_counts": dict(label_counts),
    }

    all_records: list[dict] = []
    per_shuffle_summaries: list[dict[str, dict]] = []

    # ---- single run (original order, shuffles=0)
    if args.shuffles == 0:
        print("\nRunning 4 baselines on original event order...")
        oracle = OracleJudge()
        vec_rec: list[dict] = []
        meta_rec: list[dict] = []
        graph_rec: list[dict] = []
        hybrid_rec: list[dict] = []
        vec = run_vector_only(events, oracle, dim, policy_factory, vec_rec, run_id, -1, None)
        metadata = run_metadata_only(events, oracle, dim, extractor, embedder, policy_factory, topic_vocab, meta_rec, run_id, -1, None)
        graph = run_graph_only(
            events, oracle, dim, extractor, embedder, policy_factory, topic_vocab,
            log_id=run_id, verbose=True,
            records=graph_rec, run_id=run_id, shuffle_idx=-1, seed=None,
        )
        hybrid = run_hybrid(events, oracle, dim, extractor, embedder, policy_factory, topic_vocab, hybrid_rec, run_id, -1, None)
        runs = [vec, metadata, graph, hybrid]
        all_records = vec_rec + meta_rec + graph_rec + hybrid_rec
        per_shuffle_summaries = [summarize_records(all_records)]
        if not args.no_audit:
            print_audit(events, runs)
        else:
            _print_summary_from_runs(events, runs)

    # ---- shuffle batch
    else:
        print(f"\nGenerating {args.shuffles} shuffles (base seed {args.seed})...")
        variants = generate_shuffles(events, args.shuffles, args.seed)
        for shuffle_idx, (seed, shuffled_events) in enumerate(variants):
            print(f"\n[Shuffle {shuffle_idx + 1}/{args.shuffles}] seed={seed} — 4 baselines in parallel...")
            runs, records = _run_all_baselines_parallel(
                shuffled_events, dim, extractor, embedder, policy_factory, topic_vocab,
                run_id=run_id, shuffle_idx=shuffle_idx, seed=seed,
            )
            all_records.extend(records)
            per_shuffle_summaries.append(summarize_records(records))
            if not args.no_audit:
                print_audit(shuffled_events, runs)
            else:
                _print_summary_from_runs(shuffled_events, runs)
        _print_aggregate_summary(aggregate_summaries(per_shuffle_summaries))

    jsonl_path, summary_path = write_logs(run_id, meta, all_records, per_shuffle_summaries, args.log_dir)
    print(f"\nEvent log : {jsonl_path}  ({len(all_records)} records)")
    print(f"Summary   : {summary_path}")


if __name__ == "__main__":
    main()
