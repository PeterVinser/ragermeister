"""Side-by-side audit of the vector-only (v1) and graph-only baselines over a labelled
event stream (``data/debug_events.jsonl``).

The research question is *candidate discovery*: both baselines feed the SAME conflict
judge and run the SAME resolution policy — they differ only in how they surface candidate
neighbours for an arriving chunk. To isolate that variable we hold everything else
constant and competent:

  * Judge  -> ``OracleJudge``: a perfect classifier conditioned on the dataset's gold.
             A baseline detects a conflict iff its discovery surfaced an implicated doc.
             This makes the judge a non-variable upper bound (it never mislabels).
  * Vector-only input -> real embeddings (high quality).
  * Graph-only input  -> ``CuratedEntityExtractor``: a transparent rule-based, typed NER
             stand-in feeding the shared entity resolver — the symmetric "competent input"
             for the structural baseline. Swap in the raw ``LLMEntityExtractor``
             (``use_llm_extractor=True``) to audit unaided behaviour.
  * Resolution -> recency (last-write-wins), deterministic, no human in the loop.

So the headline metric — did FAISS-KNN vs graph-PPR each surface the implicated docs? —
is judge-independent and fully auditable. We print every judge call and a summary.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

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
from solution.services.rrf_house_keeper import RRFHouseKeeper
from solution.services.vector_house_keeper import VectorHouseKeeper
from solution.services.policies.base import ResolutionPolicy
from solution.services.policies.recency import RecencyPolicy
from solution.services.resolution_manager import ResolutionManager
from solution.services.vector_db import VectorDB

from typing import Callable

from solution.models.conflict import ConflictReport, Decision, DecisionAction

_DEFAULT_EVENTS = Path(__file__).resolve().parents[2] / "data" / "debug_events.jsonl"

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


# --------------------------------------------------------------------------- judge


class OracleJudge:
    """Gold-conditioned perfect judge. Set ``current`` to the event's gold before ingest.
    Returns the gold conflict label iff at least one *implicated* doc is among the
    candidate neighbours; otherwise CLEAN (discovery missed → cannot detect). Identical
    logic for every baseline, so any verdict difference is attributable purely to which
    candidates the baseline's discovery surfaced."""

    def __init__(self) -> None:
        self.current: GoldEvent | None = None

    def judge(self, new_chunk: Chunk, neighbors: list[Chunk]) -> JudgeResult:
        gold = self.current
        if gold is None or gold.expected_label == ConflictLabel.CLEAN:
            return JudgeResult(label=ConflictLabel.CLEAN, implicated_ids=[], proposed_action="insert", rationale="clean (gold)")
        relevant = [n for n in neighbors if n.doc_id in gold.implicates]
        if not relevant:
            return JudgeResult(
                label=ConflictLabel.CLEAN,
                implicated_ids=[],
                proposed_action="insert",
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
    """Transparent wrapper around the SHARED judge that records every call for one
    baseline's run. The underlying classifier is untouched — only observed."""

    def __init__(self, inner: OracleJudge, docstore: Docstore) -> None:
        self._inner = inner
        self._docstore = docstore
        self.current_seq = 0
        self.calls: list[JudgeCall] = []

    def judge(self, new_chunk: Chunk, neighbors: list[Chunk]) -> JudgeResult:
        result = self._inner.judge(new_chunk, neighbors)
        implicated_docs = sorted(
            {
                c.doc_id
                for cid in result.implicated_ids
                if (c := self._docstore._chunks.get(cid)) is not None
            }
        )
        self.calls.append(
            JudgeCall(
                seq=self.current_seq,
                new_doc=new_chunk.doc_id,
                candidate_docs=sorted({n.doc_id for n in neighbors}),
                label=result.label,
                implicated_docs=implicated_docs,
            )
        )
        return result


# --------------------------------------------------------------------------- keys


class CuratedEntityExtractor(EntityExtractor):
    """Rule-based, typed NER stand-in for the admissions dataset — the symmetric "competent
    input" for the structural baseline (parallel to giving vector-only real embeddings).
    Emits the SAME schema as the LLM extractor ({surface_form, type, short_context}) so the
    resolver pipeline is exercised end to end. Surface forms are canonicalised to a stable
    name per real-world entity and paired with a distinctive context, so identical mentions
    embed identically (auto-merge at tau_high) while distinct entities stay apart. Purely
    lexical in mechanism; the vocabulary is tuned to this corpus."""

    # name -> (surface patterns, disambiguating context)
    _PERSONS: dict[str, tuple[list[str], str]] = {
        "Elena Vance": (["elena vance", "vance"], "dean of admissions"),
        "Marcus Reed": (["marcus reed", "reed"], "admissions staff member"),
    }
    _ORGS: dict[str, tuple[list[str], str]] = {
        "Admissions Office": (
            ["dean of admissions", "admissions office", "admissions leadership",
             "lead northmoor", "admissions head"],
            "the admissions office",
        ),
        "Northmoor University": (["northmoor"], "northmoor university campus"),
        "Hartwell Library": (["hartwell library"], "hartwell library building"),
    }
    _TOPICS: dict[str, list[str]] = {
        "application-deadline": [
            "application deadline", "deadline", "apply by", "autumn intake",
            "submit their application", "applying to northmoor",
        ],
        "application-fee": [
            "application fee", "application charge", "application fee for",
            "$50", "$75", "fee remains", "waive the application fee", "fee entirely",
        ],
        "library-hours": ["library", "operates from"],
        "welcome-seminar": ["welcome seminar", "seminar"],
    }
    _DATES: dict[str, list[str]] = {
        "March 1": ["march 1", "first of march"],
        "April 1": ["april 1"],
    }

    def extract(self, chunk: Chunk) -> list[EntityMention]:
        low = chunk.text.lower()
        mentions: list[EntityMention] = []

        for name, (pats, context) in self._PERSONS.items():
            if any(p in low for p in pats):
                mentions.append(
                    EntityMention(surface_form=name, type=EntityType.PERSON, short_context=context)
                )
        for name, (pats, context) in self._ORGS.items():
            if any(p in low for p in pats):
                mentions.append(
                    EntityMention(surface_form=name, type=EntityType.ORG, short_context=context)
                )
        for _slug, pats in self._TOPICS.items():
            matched = next((p for p in pats if p in low), None)
            if matched is not None:
                # Emit the matched phrase; the resolver snaps it to the controlled slug.
                mentions.append(
                    EntityMention(surface_form=matched, type=EntityType.TOPIC, short_context="")
                )
        for display, pats in self._DATES.items():
            if any(p in low for p in pats):
                mentions.append(
                    EntityMention(surface_form=display, type=EntityType.DATE, short_context="")
                )
        return mentions

# --------------------------------------------------------------------------- policy


class InsertOnlyPolicy(ResolutionPolicy):
    """Non-destructive resolution: record the detection (signal already emitted via the
    auditing judge) but commit the new chunk WITHOUT removing implicated docs. Keeps the
    corpus monotonic so every prior doc stays discoverable — this isolates candidate
    discovery from resolution policy. Detection still always runs (invariant)."""

    def resolve(
        self, report: ConflictReport, apply_decision: Callable[[Decision], None]
    ) -> None:
        apply_decision(
            Decision(
                report_id=report.report_id,
                action=DecisionAction.INSERT,
                new_chunk=report.new_chunk,
                chunk_ids_to_remove=[],
            )
        )


# --------------------------------------------------------------------------- runner


@dataclass
class BaselineRun:
    name: str
    calls: list[JudgeCall]
    final_docs: list[str]  # doc_ids still committed after the full stream
    # aggregated per-seq view (union over a doc's chunks)
    per_seq: dict[int, JudgeCall] = field(default_factory=dict)


def _aggregate(calls: list[JudgeCall]) -> dict[int, JudgeCall]:
    """Collapse multiple per-chunk judge calls for one event into one per-seq summary."""
    agg: dict[int, JudgeCall] = {}
    for c in calls:
        cur = agg.get(c.seq)
        if cur is None:
            agg[c.seq] = JudgeCall(
                c.seq, c.new_doc, list(c.candidate_docs), c.label, list(c.implicated_docs)
            )
            continue
        cur.candidate_docs = sorted(set(cur.candidate_docs) | set(c.candidate_docs))
        cur.implicated_docs = sorted(set(cur.implicated_docs) | set(c.implicated_docs))
        # any non-clean verdict wins
        if cur.label == ConflictLabel.CLEAN and c.label != ConflictLabel.CLEAN:
            cur.label = c.label
    return agg


def run_vector_only(
    events: list[GoldEvent],
    oracle: OracleJudge,
    dim: int,
    policy_factory: Callable[[], ResolutionPolicy],
) -> BaselineRun:
    vdb, docstore = VectorDB(dim), Docstore()
    auditor = AuditingJudge(oracle, docstore)
    housekeeper = VectorHouseKeeper(vdb, docstore)
    kb = KnowledgeBase(vdb, docstore, auditor, housekeeper)  # type: ignore[arg-type]
    manager = ResolutionManager(policy_factory(), kb.apply_decision)
    kb.conflict_sink = manager.submit
    for ev in events:
        oracle.current = ev
        auditor.current_seq = ev.seq
        kb.ingest(ev.to_ingest())
    final = sorted(d for d in docstore._doc_chunks if docstore._doc_chunks[d])
    return BaselineRun("vector-only", auditor.calls, final, _aggregate(auditor.calls))


def run_graph_only(
    events: list[GoldEvent],
    oracle: OracleJudge,
    dim: int,
    extractor: EntityExtractor,
    embedder: Embedder,
    policy_factory: Callable[[], ResolutionPolicy],
    topic_vocab: dict[str, list[str]] | None = None,
) -> BaselineRun:
    vdb, docstore = VectorDB(dim), Docstore()
    auditor = AuditingJudge(oracle, docstore)
    # Shared ER infrastructure: blocking tuned for recall, adjudicator (mock -> create_new
    # on the gray band) for precision. Swap in LLMEntityAdjudicator to use a real model.
    event_log_filename = f"snapshot_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.json"
    resolver = EntityResolver(
        extractor=extractor,
        adjudicator=EntityCandidatesJudge(),
        embedder=embedder,
        event_log=EventLog(path=f"graph/resolver/{event_log_filename}"),
        topic_vocab=topic_vocab,
    )
    housekeeper = GraphHouseKeeper(
        vector_db=vdb,
        docstore=docstore,
        resolver=resolver,
        event_log=EventLog(path=f"graph/housekeeper/{event_log_filename}"),
    )
    kb = KnowledgeBase(vdb, docstore, auditor, housekeeper)  # type: ignore[arg-type]
    manager = ResolutionManager(policy_factory(), kb.apply_decision)
    kb.conflict_sink = manager.submit
    for ev in events:
        oracle.current = ev
        auditor.current_seq = ev.seq
        kb.ingest(ev.to_ingest())
    final = sorted(d for d in docstore._doc_chunks if docstore._doc_chunks[d])  # noqa: SLF001
    _print_alias_clusters(resolver)
    return BaselineRun("graph-only", auditor.calls, final, _aggregate(auditor.calls))


def run_metadata_only(
    events: list[GoldEvent],
    oracle: OracleJudge,
    dim: int,
    extractor: EntityExtractor,
    embedder: Embedder,
    policy_factory: Callable[[], ResolutionPolicy],
    topic_vocab: dict[str, list[str]] | None = None,
) -> BaselineRun:
    vdb, docstore = VectorDB(dim), Docstore()
    auditor = AuditingJudge(oracle, docstore)
    # Same shared ER infrastructure graph-only uses (its own instance, but identical
    # extractor/adjudicator/embedder -> identical canonical entities and quality).
    resolver = EntityResolver(
        extractor=extractor,
        adjudicator=EntityCandidatesJudge(),
        embedder=embedder,
        topic_vocab=topic_vocab,
    )
    housekeeper = MetadataHouseKeeper(docstore=docstore, resolver=resolver)
    kb = KnowledgeBase(vdb, docstore, auditor, housekeeper)  # type: ignore[arg-type]
    manager = ResolutionManager(policy_factory(), kb.apply_decision)
    kb.conflict_sink = manager.submit
    for ev in events:
        oracle.current = ev
        auditor.current_seq = ev.seq
        kb.ingest(ev.to_ingest())
    final = sorted(d for d in docstore._doc_chunks if docstore._doc_chunks[d])  # noqa: SLF001
    return BaselineRun("metadata-only", auditor.calls, final, _aggregate(auditor.calls))


def _make_resolver(
    extractor: EntityExtractor,
    embedder: Embedder,
    topic_vocab: dict[str, list[str]] | None,
) -> EntityResolver:
    """Shared ER infrastructure. Each baseline gets its own instance, but identical
    extractor/adjudicator/embedder -> identical canonical entities and resolution quality
    (the cap is shared, not baseline-specific)."""
    return EntityResolver(
        extractor=extractor,
        adjudicator=EntityCandidatesJudge(),
        embedder=embedder,
        topic_vocab=topic_vocab,
    )


def run_hybrid(
    events: list[GoldEvent],
    oracle: OracleJudge,
    dim: int,
    extractor: EntityExtractor,
    embedder: Embedder,
    policy_factory: Callable[[], ResolutionPolicy],
    topic_vocab: dict[str, list[str]] | None = None,
) -> BaselineRun:
    vdb, docstore = VectorDB(dim), Docstore()
    auditor = AuditingJudge(oracle, docstore)
    housekeeper = HybridHouseKeeper(
        vector_db=vdb,
        docstore=docstore,
        resolver=_make_resolver(extractor, embedder, topic_vocab),
    )
    kb = KnowledgeBase(vdb, docstore, auditor, housekeeper)  # type: ignore[arg-type]
    manager = ResolutionManager(policy_factory(), kb.apply_decision)
    kb.conflict_sink = manager.submit
    for ev in events:
        oracle.current = ev
        auditor.current_seq = ev.seq
        kb.ingest(ev.to_ingest())
    final = sorted(d for d in docstore._doc_chunks if docstore._doc_chunks[d])  # noqa: SLF001
    return BaselineRun("hybrid", auditor.calls, final, _aggregate(auditor.calls))


def run_rrf(
    events: list[GoldEvent],
    oracle: OracleJudge,
    dim: int,
    extractor: EntityExtractor,
    embedder: Embedder,
    policy_factory: Callable[[], ResolutionPolicy],
    topic_vocab: dict[str, list[str]] | None = None,
) -> BaselineRun:
    vdb, docstore = VectorDB(dim), Docstore()
    auditor = AuditingJudge(oracle, docstore)
    # RRF fuses the two PARENT strategies' independent rankings — vote, not guidance.
    vector_hk = VectorHouseKeeper(vdb, docstore)
    graph_hk = GraphHouseKeeper(
        vector_db=vdb,
        docstore=docstore,
        resolver=_make_resolver(extractor, embedder, topic_vocab),
    )
    housekeeper = RRFHouseKeeper(vector_hk, graph_hk)
    kb = KnowledgeBase(vdb, docstore, auditor, housekeeper)  # type: ignore[arg-type]
    manager = ResolutionManager(policy_factory(), kb.apply_decision)
    kb.conflict_sink = manager.submit
    for ev in events:
        oracle.current = ev
        auditor.current_seq = ev.seq
        kb.ingest(ev.to_ingest())
    final = sorted(d for d in docstore._doc_chunks if docstore._doc_chunks[d])  # noqa: SLF001
    return BaselineRun("rrf(v+g)", auditor.calls, final, _aggregate(auditor.calls))


def _print_alias_clusters(resolver: EntityResolver) -> None:
    """Metric hook: alias clusters per canonical node. Compared against a gold entity set
    this yields the false-discovery-rate of unresolved entities, which caps graph-only and
    metadata-only recall."""
    clusters = resolver.alias_clusters()
    print("\n" + "-" * 60)
    print(f"ENTITY RESOLUTION — {len(clusters)} canonical entities")
    print("-" * 60)
    for cid, info in sorted(clusters.items()):
        print(f"  {cid:14} [{info['type']:6}] {info['canonical_name']:22} "
              f"aliases={info['aliases']}")


# --------------------------------------------------------------------------- reporting


def _fmt(items: list[str], width: int = 34) -> str:
    s = ", ".join(items) if items else "-"
    return s if len(s) <= width else s[: width - 1] + "~"


def _detected_with_implication(call: JudgeCall | None, implicates: list[str]) -> bool:
    return bool(
        call
        and call.label != ConflictLabel.CLEAN
        and (set(implicates) & set(call.implicated_docs))
    )


def print_audit(events: list[GoldEvent], runs: list[BaselineRun]) -> None:
    name_w = max(len(r.name) for r in runs)
    print("=" * 100)
    print("PER-EVENT AUDIT  (candidates = docs surfaced by discovery; OK = conflict detected)")
    print("=" * 100)
    for ev in events:
        gold_conf = ev.expected_label != ConflictLabel.CLEAN
        print(
            f"\n[seq {ev.seq}] {ev.event_type.value.upper():6} doc={ev.doc_id!r}  "
            f"intent={ev.intent}"
        )
        print(
            f"    gold: {ev.expected_label.value:13} implicates={_fmt(ev.implicates, 50)}"
        )
        for run in runs:
            call = run.per_seq.get(ev.seq)
            if call is None:
                print(f"    {run.name:<{name_w}}: (no judge call — empty/no chunks)")
                continue
            detected = call.label != ConflictLabel.CLEAN
            mark = "OK " if detected == gold_conf and (
                not gold_conf or set(ev.implicates) & set(call.implicated_docs)
            ) else "MISS"
            print(
                f"    {run.name:<{name_w}}: cand=[{_fmt(call.candidate_docs)}] "
                f"verdict={call.label.value:13} {mark}"
            )
    _print_summary(events, runs)


def _print_summary(events: list[GoldEvent], runs: list[BaselineRun]) -> None:
    conflicts = [e for e in events if e.expected_label != ConflictLabel.CLEAN]
    controls = [e for e in events if e.expected_label == ConflictLabel.CLEAN]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    def stats(run: BaselineRun) -> tuple[float, float, int]:
        detected = 0
        recall_num = recall_den = 0
        for e in conflicts:
            call = run.per_seq.get(e.seq)
            cand = set(call.candidate_docs) if call else set()
            found = set(e.implicates) & cand
            recall_num += len(found)
            recall_den += len(e.implicates)
            if call and call.label != ConflictLabel.CLEAN and found:
                detected += 1
        det_rate = detected / len(conflicts) if conflicts else 0.0
        recall = recall_num / recall_den if recall_den else 0.0
        return det_rate, recall, detected

    print(f"\nConflict events: {len(conflicts)}  |  Clean controls: {len(controls)}")
    print(f"\n{'baseline':14} {'detected':>10} {'detect-rate':>12} {'affected-recall':>16}")
    print("-" * 56)
    for run in runs:
        det_rate, recall, detected = stats(run)
        print(
            f"{run.name:14} {detected:>5}/{len(conflicts):<4} "
            f"{det_rate:>11.0%} {recall:>15.0%}"
        )

    # Per-conflict detection grid: one column per baseline (where do they diverge?).
    col_w = max(6, max(len(r.name) for r in runs))
    print(f"\nPer-conflict detection (OK = detected w/ correct implication, MISS = missed):")
    header = f"{'seq':>4} {'doc':16} {'intent':30} " + " ".join(
        f"{r.name:>{col_w}}" for r in runs
    )
    print(header)
    print("-" * len(header))
    for e in conflicts:
        marks = " ".join(
            f"{('OK' if _detected_with_implication(r.per_seq.get(e.seq), e.implicates) else 'MISS'):>{col_w}}"
            for r in runs
        )
        print(f"{e.seq:>4} {e.doc_id:16} {e.intent[:30]:30} {marks}")

    # False-positive pressure on clean controls: candidates an imperfect judge might flag.
    print(f"\nClean-control candidate pressure (oracle stays clean; real judge might not):")
    header = f"{'seq':>4} {'doc':16} " + " ".join(f"{r.name:>{col_w}}" for r in runs)
    print(header)
    print("-" * len(header))
    for e in controls:
        counts = " ".join(
            f"{(len(c.candidate_docs) if (c := r.per_seq.get(e.seq)) else 0):>{col_w}}"
            for r in runs
        )
        print(f"{e.seq:>4} {e.doc_id:16} {counts}")

    print("\nNotes:")
    print("  - 'detect-rate' = conflict events where discovery surfaced >=1 implicated doc")
    print("    AND the oracle therefore flagged it. Under the oracle this is exactly a")
    print("    test of candidate discovery.")
    print("  - 'affected-recall' = fraction of all gold-implicated docs that discovery")
    print("    surfaced (micro-averaged over conflict events).")
    print("  - Clean-control candidate counts show over-surfacing: harmless under the")
    print("    oracle, but a real judge could turn surfaced candidates into false flags.")
    print("  - metadata-only is single-hop faceted match: it should catch seq 4/5 (shared")
    print("    facets) and seq 7 (single shared entity, if resolution links them), but MISS")
    print("    the transitive tail of seq 10 and the derived_from staleness of seq 8 - no")
    print("    traversal. seq 9 (shares the Vance entity, clean) is its precision stress.")


def main(use_llm_extractor: bool = False, resolve: bool = False) -> None:
    events = load_events()
    print(f"Loaded {len(events)} events from {_DEFAULT_EVENTS}")

    embedder = Embedder()  # real Azure embeddings (shared by both baselines)
    dim = int(embedder.embed(["dimension probe"]).shape[1])
    print(f"Embedding dim = {dim}")

    oracle = OracleJudge()  # SHARED across both baselines

    extractor: EntityExtractor = (
        LLMEntityExtractor() if use_llm_extractor else CuratedEntityExtractor()
    )
    topic_vocab = None if use_llm_extractor else CuratedEntityExtractor._TOPICS
    if resolve:
        policy_factory: Callable[[], ResolutionPolicy] = RecencyPolicy
        mode = "recency (destructive; shows real end-to-end dynamics + v1 leak)"
    else:
        policy_factory = InsertOnlyPolicy
        mode = "insert-only (monotonic corpus; isolates candidate discovery)"

    print(
        f"Graph entity extractor: {type(extractor).__name__}"
        f"{'  (raw/unaided)' if use_llm_extractor else '  (curated NER stand-in)'}"
    )
    print(f"Resolution mode       : {mode}\n")

    vec = run_vector_only(events, oracle, dim, policy_factory)
    metadata = run_metadata_only(
        events, oracle, dim, extractor, embedder, policy_factory, topic_vocab
    )
    graph = run_graph_only(
        events, oracle, dim, extractor, embedder, policy_factory, topic_vocab
    )
    rrf = run_rrf(
        events, oracle, dim, extractor, embedder, policy_factory, topic_vocab
    )
    hybrid = run_hybrid(
        events, oracle, dim, extractor, embedder, policy_factory, topic_vocab
    )

    print_audit(events, [vec, metadata, graph, rrf, hybrid])
