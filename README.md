# RAGermeister

**Graph-augmented housekeeping for dynamic RAG knowledge bases.**

Most RAG research focuses on *how to retrieve better context*. This project asks a different question: **how does a knowledge base keep itself consistent as documents are added, updated, and removed over time?**

The deliverable is a *housekeeper* that sits above a vector store, detects when a change introduces a contradiction, duplication, or supersession, and routes it to a resolution policy. The central experimental question is whether a **typed dependency graph** over chunks and entities can surface better conflict candidates than raw FAISS nearest-neighbour search — and whether a hybrid of both wins over either alone.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [High-Level Architecture](#high-level-architecture)
3. [Core Components](#core-components)
   - [KnowledgeBase](#knowledgebase)
   - [HouseKeeper (abstract)](#housekeeper-abstract)
   - [ConflictJudge](#conflictjudge)
   - [ResolutionManager & Policies](#resolutionmanager--policies)
4. [Housekeeper Baselines](#housekeeper-baselines)
   - [VectorHouseKeeper](#vectorhousekeepervector-only)
   - [MetadataHouseKeeper](#metadatahousekeepermetadata-only)
   - [GraphHouseKeeper](#graphhousekeepergraph-only)
   - [HybridHouseKeeper](#hybridhousekeeperhybrid)
5. [Entity Resolution Pipeline](#entity-resolution-pipeline)
6. [Graph Layer](#graph-layer)
7. [Storage Layer](#storage-layer)
8. [Evaluation Harness](#evaluation-harness)
9. [Hyperparameter Reference](#hyperparameter-reference)
10. [Key Design Invariants](#key-design-invariants)
11. [Data Flow Walkthrough](#data-flow-walkthrough)

---

## Project Goal

A live knowledge base receives a stream of insert/update/delete events. When a new chunk arrives it may:

- **contradict** an existing chunk (incompatible claims)
- **duplicate** an existing chunk (same information restated)
- **supersede** an existing chunk (a newer version replaces an older one)
- be entirely **clean** (no conflict)

The system must detect which case applies and route the event to a policy (auto-resolve or queue for human review). The detection quality depends entirely on *which existing chunks are surfaced as candidates* — this is what the four baselines compete on.

---

## High-Level Architecture

```
                 ┌────────────────────────────────────────────┐
  IngestEvent ──►│             KnowledgeBase                  │
                 │                                            │
                 │  embed ──► HouseKeeper.find_candidates()   │
                 │                        │                   │
                 │                        ▼                   │
                 │              ConflictJudge.judge()  ◄──────┼── SHARED (LLM)
                 │                        │                   │
                 │          ┌─────────────┴──────────────┐    │
                 │      CLEAN                         CONFLICT │
                 │          │                             │    │
                 │     commit()                   conflict_sink│
                 │     on_commit()                         │    │
                 └─────────────────────────────────────────┘    │
                                                                │
                 ┌──────────────────────────────────────────────┘
                 │
                 ▼
        ResolutionManager
                 │
         ┌───────┴────────┐
    RecencyPolicy     HumanPolicy
         │                │
         └──────┬──────────┘
                ▼
        KnowledgeBase.apply_decision()
                │
         ┌──────┴──────────────────┐
    commit / retire          on_retire / on_resolution
         │                         │
    VectorDB + Docstore        HouseKeeper auxiliary state
```

The key inversion: the `KnowledgeBase` is **baseline-agnostic**. It always embeds and stores chunks — it is the retrieval ground truth. A swappable `HouseKeeper` monitors arrivals and surfaces *candidates*; the KB feeds those to the same shared judge for every baseline. Any benchmark difference is attributable only to candidate discovery, not to a different judge.

---

## Core Components

### KnowledgeBase

**File:** `solution/services/knowledge_base.py`

The central orchestrator. It owns the embedder, vector index, docstore, conflict judge, and housekeeper. It never reasons about consistency itself — it delegates candidate discovery to the injected housekeeper.

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector_db` | `VectorDB` | FAISS-backed vector store |
| `docstore` | `Docstore` | chunk metadata and index |
| `judge` | `ConflictJudge` | shared LLM conflict classifier |
| `housekeeper` | `HouseKeeper` | swappable candidate-discovery strategy |
| `conflict_sink` | `Callable` | where to send `ConflictReport`s (wired to `ResolutionManager.submit`) |
| `revalidate_on_retire` | `bool` | if `True`, re-judges chunks that shared a semantic edge with a retired chunk (diachronic re-validation; default `False`) |

**Key methods:**

- `query(text, k=5)` — embed + KNN retrieve top-k chunks.
- `ingest(event)` — the main entry point. Routes INSERT/UPDATE/DELETE events through the full detect → commit/route pipeline. For UPDATE, it first retires the old doc's chunks, then processes the new content.
- `apply_decision(decision)` — called by the resolution manager after a conflict is resolved. Commits or removes chunks, notifies the housekeeper, and optionally triggers diachronic re-validation.

**Mutual-dependency note:** `KnowledgeBase` and `ResolutionManager` are mutually dependent. The recommended wiring order: create the KB with `conflict_sink=None`, create the manager pointing at `kb.apply_decision`, then set `kb.conflict_sink = manager.submit`.

---

### HouseKeeper (abstract)

**File:** `solution/services/house_keeper.py`

The single interface that all baselines implement. The only **required** method is `find_candidates`; all lifecycle hooks default to no-ops.

```python
class HouseKeeper(ABC):
    def find_candidates(chunk, embedding) -> list[Chunk]  # REQUIRED
    def on_commit(chunk, embedding) -> None               # no-op default
    def on_retire(chunk_ids) -> list[Chunk]               # returns [] default
    def on_resolution(decision, new_chunk) -> None        # no-op default
```

- `find_candidates` — called **before** the chunk is committed. Return the existing chunks the arriving chunk might conflict with. This is the only thing that differs between baselines.
- `on_commit` — called **after** the base has committed the chunk (it already has a `vec_id`). Stateful housekeepers mirror the commit into their auxiliary structures here.
- `on_retire` — called **before** the base removes chunks. Stateful housekeepers tombstone nodes and return any live chunks whose consistency the removal may have broken (for diachronic re-validation).
- `on_resolution` — called when a resolution commits. Records provenance edges in the graph.

---

### ConflictJudge

**File:** `solution/services/conflict_judge.py`

An LLM-powered classifier (backed by `LLM`, which calls Azure OpenAI). It is **shared across all baselines** — the housekeeper is never allowed to specialise or bypass it.

**Input:** one arriving chunk + its candidate neighbours.

**Output:** a `JudgeResult` with fields:
- `label` — one of `clean | duplicate | contradiction | supersedes | needs_human`
- `implicated_ids` — chunk IDs of existing chunks involved in the conflict
- `proposed_action` — `insert | replace_old | skip | flag_for_human`
- `rationale` — one sentence

When there are no candidates at all the judge short-circuits to `CLEAN` without an LLM call.

The judge is behind an interface so tests can inject an `OracleJudge` (gold-conditioned, no LLM calls) to isolate candidate-discovery quality from judge quality.

---

### ResolutionManager & Policies

**File:** `solution/services/resolution_manager.py`, `solution/services/policies/`

The `ResolutionManager` receives `ConflictReport`s from the KB's `conflict_sink` and immediately drains them through the configured `ResolutionPolicy`.

**Policies:**

| Class | Behaviour |
|-------|-----------|
| `RecencyPolicy` | Auto-resolves every conflict as UPDATE — newest chunk supersedes all implicated ones (last-write-wins). |
| `HumanPolicy` | Holds reports in a pending dict until `submit_human_decision(decision)` is called by an external caller. |

Detection always runs and always emits a signal regardless of which policy is in use. This is invariant #2 — the detection event stream feeds the consistency-score metric even when resolution is fully automatic.

---

## Housekeeper Baselines

### VectorHouseKeeper (vector-only)

**File:** `solution/services/vector_house_keeper.py`

The simplest baseline. Candidate discovery is a live KNN of the arriving chunk's embedding against the FAISS index. Holds no auxiliary state; every lifecycle hook is the inherited no-op.

**Hyperparameter:**

| Name | Default | Effect |
|------|---------|--------|
| `k` | `5` | Number of nearest neighbours to surface as candidates |

**Limitation (banked):** `on_retire` returns `[]` — it cannot find the stale chunks left behind by a supersession because it has no graph to traverse.

---

### MetadataHouseKeeper (metadata-only)

**File:** `solution/services/metadata_house_keeper.py`

Candidate discovery by faceted attribute matching. Builds one inverted index per facet type from the entity resolver's output. Discovery is single-hop: no edges, no traversal.

**Facet types:**
- `entity` — canonical entity IDs (post-resolution, not raw strings)
- `date` — ISO-8601 date strings
- `topic` — controlled-vocabulary slugs
- `source` — `source_id` from event metadata
- `title` — tokenized title (stopwords removed)

**Scoring:** candidates are ranked by the sum of the IDF scores of the facets they share with the arriving chunk. Rare shared facets (a specific person's ID) score higher than common ones (a shared date).

**IDF formula:** `log((N+1)/(df+1)) + 1.0` (smoothed, entity-type scoped)

**Hyperparameter:**

| Name | Default | Effect |
|------|---------|--------|
| `top_n` | `5` | Max candidates returned to the judge |

**Limitation (banked):** structurally cannot follow supersession chains or `derived_from` dependencies — no edges.

---

### GraphHouseKeeper (graph-only)

**File:** `solution/services/graph_house_keeper.py`

Maintains a typed dependency graph as an auxiliary structure beside the KB. Candidate discovery is structural: extract + resolve canonical entities/dates/topics → look them up as anchor nodes in the graph → run Personalized PageRank outward from those anchors.

**On commit (`on_commit`):**
1. Mutating entity resolve: mint/merge canonical entities into the `EntityResolver`.
2. Build-time KNN over FAISS (size `_SIM_FREEZE_K + 1`, excluding self): freeze the top-K similar chunks as `similar_to` edges with confidence = cosine score.
3. Add a `CHUNK` node with structural edges: `CONTAINS` (doc→chunk), `MENTIONS` (chunk→entity), `ON_DATE`, `HAS_TOPIC`.
4. Add `DERIVED_FROM` edges for artifact provenance.
5. Append a `commit` record to the event log.

**On retire (`on_retire`):**
1. Tombstone the retiring chunk nodes (never hard-delete).
2. Revoke incident semantic edges (belief pruned, structure kept).
3. Return the still-live chunks on the other end of revoked edges — these are diachronic re-validation candidates.

**Discovery algorithm:**
1. Read-only resolve the arriving chunk (no entity minting — an arriving chunk that gets flagged must not create entities).
2. Look up canonical entity IDs, dates, topics, and source in the graph's symbolic indexes → get anchor node IDs.
3. Personalized PageRank seeded at those anchors → top-N active chunk nodes.

**Hyperparameters:**

| Name | Default | Effect |
|------|---------|--------|
| `_GRAPH_TOP_N` | `5` | Max candidates from PPR |
| `_SIM_FREEZE_K` | `5` | KNN size for freezing `similar_to` edges at build time |
| `_CHECKPOINT_EVERY` | `50` | Write a graph checkpoint every N events |

**Persistence:** the append-only `EventLog` is the source of truth. The graph is a materialised view over it. Calling `rebuild_from_log()` replays the log deterministically without re-embedding, re-judging, or re-resolving.

---

### HybridHouseKeeper (hybrid)

**File:** `solution/services/hybrid_house_keeper.py`

Subclasses `GraphHouseKeeper` — it builds the identical graph (all lifecycle hooks are inherited unchanged). The **only** difference is how candidate discovery seeds the PageRank walk: instead of pure symbolic anchors, it mixes three seed populations into a single teleport vector.

**Seeding (the teleport vector is built by ADDITION, not score fusion):**

| Seed source | Weight | Rationale |
|-------------|--------|-----------|
| **Vector seeds** | `sim²` (cosine squared) | Live KNN against FAISS; `sim²` sharpens toward confident matches |
| **Entity seeds** | `w_entity / (1 + degree(node))` | Hub down-weighting prevents a generic entity from dominating |
| **Update-identity seeds** | `w_old` (fixed, high) | On an update, the prior version's now-tombstoned chunk nodes are structural certainty about what the update affects |

After seeding, a single PPR diffusion over the **directed walk graph** (not the undirected projection used by graph-only) is run with teleport probability `restart`.

**Hyperparameters:**

| Name | Default | Effect |
|------|---------|--------|
| `top_n` | `5` | Max candidates returned |
| `k_vec` | `5` | KNN size for vector seeds |
| `w_entity` | `1.0` | Base weight for entity seeds |
| `w_old` | `3.0` | Weight for update-identity seeds |
| `restart` | `0.2` | PPR teleport probability (locality/precision dial) |

The directed walk graph is built fresh per call from the stored graph using `build_walk_digraph()`, where each relation gets both a forward and a typed reverse edge (so `derived_from` and `supersedes` chains are traversable in both directions).

---

## Entity Resolution Pipeline

**Files:** `solution/services/entity_resolver.py`, `solution/services/entity_extractor.py`, `solution/services/entity_index.py`

Used by graph-only, metadata-only, and hybrid. Built once and shared; resolution quality must not vary across baselines.

### Pipeline stages

```
chunk.text
    │
    ▼
EntityExtractor.extract()          ← LLM pass (or CuratedEntityExtractor for eval)
    │  list[EntityMention]
    ▼
For each PERSON/ORG mention:
    embed(surface_form + type + short_context)
    EntityIndex.block(embedding, type, k=10, floor=0.30)   ← cosine KNN within type
        │  list[EntityCandidate] sorted by score
        ▼
    Three-band adjudication on top score:
        score >= 0.86  → auto-merge   (high band)
        score <  0.55  → auto-create  (low band)
        in between     → EntityCandidatesJudge.judge() → MERGE or CREATE_NEW  (gray band)
    │
    ▼
MentionResolution(canonical_id or None)

For DATE mentions   → regex-normalize to ISO-8601
For TOPIC mentions  → snap to controlled vocabulary slug
```

**Two-phase design:**
- `resolve(chunk)` — read-only. Extracts, embeds, adjudicates, caches. Does **not** mint or update canonical entities. Safe to call on a chunk that might later be rejected.
- `commit(chunk)` — mutating. Applies the resolution: appends aliases on merge, mints a new canonical node on create. Reuses the `resolve` cache (no extra extraction/embedding).

**Adjudication caches:** verdicts are cached on `(mention_signature, top_candidate_id)`, so no mention pair is ever adjudicated twice. Re-ingesting the same chunk is idempotent.

**EntityIndex blocking:** dense per-type cosine scan (no ANN needed at corpus scale ~100-200 docs). Returns top-K above `block_floor`. This is the recall ceiling — the adjudicator provides precision.

**Hyperparameters:**

| Name | Default | Effect |
|------|---------|--------|
| `tau_high` | `0.86` | Auto-merge threshold |
| `tau_low` | `0.55` | Auto-create threshold |
| `block_k` | `10` | Max blocking candidates per mention |
| `block_floor` | `0.30` | Minimum cosine score to enter blocking |

---

## Graph Layer

**Files:** `solution/models/graph.py`, `solution/services/graph_db.py`

### Node types

| Type | Prefix | Description |
|------|--------|-------------|
| `DOCUMENT` | `doc:` | One per logical document |
| `CHUNK` | `chunk:` | One per committed chunk |
| `ENTITY` | `ent:` | Canonical entity (person/org); keyed by `canonical_id` |
| `DATE` | `date:` | ISO-8601 date anchor |
| `TOPIC` | `topic:` | Controlled-vocab topic slug |
| `ARTIFACT` | `art:` | Generated artifact (e.g. a summary) |
| `DECISION` | `dec:` | A conflict resolution decision, for provenance |

Nodes are **never hard-deleted**. When a chunk is retired its node is tombstoned (`status = TOMBSTONED`). PPR walks through tombstoned nodes (so supersession chains stay intact) but excludes them from the returned candidate set.

### Edge tiers

| Tier | Examples | Revocable? |
|------|----------|------------|
| **Structural** | `CONTAINS`, `MENTIONS`, `ON_DATE`, `HAS_TOPIC`, `DERIVED_FROM` | Never |
| **Semantic** | `CONTRADICTS`, `DUPLICATES`, `SUPERSEDES_CONTENT`, `SIMILAR_TO` | Yes (on retire) |

Structural edges are deterministic and permanent. Semantic edges carry belief — when a node is tombstoned, its incident semantic edges are revoked (their assertion no longer holds).

### Edge weights (for PPR)

| Relation | Weight | Rationale |
|----------|--------|-----------|
| `SUPERSEDES` / `SUPERSEDES_CONTENT` / `VERSION_OF` | 3.0 | Strong structural certainty |
| `DERIVED_FROM` | 2.5 | Artifact dependencies |
| `CONTRADICTS` / `DUPLICATES` | 2.0 | High-signal conflicts |
| `CONTAINS` | 1.5 | Structural containment |
| `SIMILAR_TO` | 1.0 × confidence | Build-time frozen cosine similarity |
| `HAS_TOPIC` | 0.7 | Weak topical link |
| `MENTIONS` | 0.5 | Entity mention (cheap anchor) |
| `ON_DATE` | 0.4 | Very weak (dates are hubs) |

### Personalized PageRank kernel

**File:** `solution/services/graph_db.py`, function `_personalized_pagerank`

Power-iteration PPR implemented in NumPy (no SciPy dependency). Runs over a small dense matrix — cheap at corpus scale.

```
Algorithm:
  1. Build transition matrix W (row-normalized, weighted by edge weights)
  2. Initialize r = p  (personalization vector)
  3. Iterate:
       leaked = alpha * r[dangling_nodes].sum()
       r_new  = (1-alpha)*p + alpha*(W^T @ r) + leaked*p
     until ||r_new - r||₁ < tol or max_iter reached
  4. Return top-N active CHUNK nodes by score
```

**PPR parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `alpha` | `0.85` (graph-only), `1 - restart` (hybrid) | Damping factor |
| `max_iter` | `100` | Power-iteration cap |
| `tol` | `1e-9` | Convergence threshold |

**Graph-only vs. hybrid graphs:**
- **Graph-only** uses an undirected projection of the stored graph (`_ppr_graph()`), with revoked edges dropped and hub-down-weighted personalization.
- **Hybrid** uses a directed walk graph (`build_walk_digraph()`), where every stored relation gets an explicit typed reverse edge and `similar_to` (already stored bidirectionally) is copied as-is.

---

## Storage Layer

### VectorDB

**File:** `solution/services/vector_db.py`

Thin wrapper around `faiss.IndexIDMap2(faiss.IndexFlatIP(dim))`.
- **Index type:** flat exact inner-product search. No approximation needed at corpus scale.
- Embeddings must be **normalized** before insertion so inner product = cosine similarity.
- Auto-assigns monotonically increasing integer IDs (`vec_id`).

### Docstore

**File:** `solution/services/docstore.py`

In-memory identity layer. Maintains three dicts:
- `chunk_id → Chunk`
- `vec_id → chunk_id` (reverse lookup from FAISS ID to chunk)
- `doc_id → [chunk_ids]` (all chunks belonging to a document)

Needed because FAISS stores only vectors, not text or metadata.

### Embedder

**File:** `solution/services/embedder.py`

Thin wrapper around Azure OpenAI `text-embedding-3-large`. Returns a `(n, dim)` float32 NumPy array. Embeddings are not normalized here — callers normalize before inserting into FAISS.

### LLM

**File:** `solution/services/llm.py`

Azure OpenAI client wrapper with token counting. `get_structured_response` uses OpenAI's structured output (schema-constrained parsing) to guarantee JSON matching a Pydantic model class.

**Default model:** `gpt-5.4`. Set `reasoning_effort="none"` and `verbosity="low"` for latency.

### EventLog

**File:** `solution/services/event_log.py`

Append-only JSONL file (or in-memory list when `path=None`). Each entry is a `LogEvent(seq, type, payload)`. The graph housekeeper uses one log for commit/retire/resolution events; the entity resolver uses a separate log for entity_create/entity_merge events. The graph view is fully rebuildable by replaying the log.

---

## Evaluation Harness

**File:** `solution/eval/comparison.py`

Runs all four baselines over a labelled event stream (`data/debug_events.jsonl`) and prints a side-by-side audit.

### Key design choices

**OracleJudge** — replaces the real LLM judge. Given the gold label for the current event, it returns the correct conflict label **iff** the discovery surfaced at least one implicated doc. This makes the benchmark judge-independent: any verdict difference is attributable only to which candidates the housekeeper surfaced.

**CuratedEntityExtractor** — a rule-based, typed NER stand-in for the eval corpus. Emits the same schema as `LLMEntityExtractor`, with stable surface forms so identical mentions embed identically (auto-merge). Gives the graph/metadata baselines "competent input" without requiring LLM calls per chunk, paralleling giving the vector baseline high-quality embeddings.

**InsertOnlyPolicy** — non-destructive resolution: records the detection signal but commits the new chunk without removing implicated docs. Keeps the corpus monotonic so every prior doc stays discoverable — isolates candidate discovery from resolution effects.

### Metrics

| Metric | Definition |
|--------|-----------|
| **detect-rate** | Fraction of conflict events where discovery surfaced ≥1 implicated doc AND the oracle flagged it |
| **affected-recall** | Fraction of all gold-implicated docs surfaced by discovery (micro-averaged over conflict events) |
| **clean-control candidate pressure** | Number of candidates surfaced on clean events — harmless under oracle, but a real judge would turn these into false positives |

---

## Hyperparameter Reference

| Component | Parameter | Default | File |
|-----------|-----------|---------|------|
| VectorHouseKeeper | `k` (KNN candidates) | `5` | `vector_house_keeper.py` |
| GraphHouseKeeper | `_GRAPH_TOP_N` (PPR top-N) | `5` | `graph_house_keeper.py` |
| GraphHouseKeeper | `_SIM_FREEZE_K` (build-time KNN) | `5` | `graph_house_keeper.py` |
| GraphHouseKeeper | `_CHECKPOINT_EVERY` | `50` | `graph_house_keeper.py` |
| HybridHouseKeeper | `top_n` | `5` | `hybrid_house_keeper.py` |
| HybridHouseKeeper | `k_vec` (vector seed KNN) | `5` | `hybrid_house_keeper.py` |
| HybridHouseKeeper | `w_entity` (entity seed weight) | `1.0` | `hybrid_house_keeper.py` |
| HybridHouseKeeper | `w_old` (update-identity seed weight) | `3.0` | `hybrid_house_keeper.py` |
| HybridHouseKeeper | `restart` (PPR teleport probability) | `0.2` | `hybrid_house_keeper.py` |
| MetadataHouseKeeper | `top_n` | `5` | `metadata_house_keeper.py` |
| EntityResolver | `tau_high` (auto-merge threshold) | `0.86` | `entity_resolver.py` |
| EntityResolver | `tau_low` (auto-create threshold) | `0.55` | `entity_resolver.py` |
| EntityResolver | `block_k` (blocking candidates) | `10` | `entity_resolver.py` |
| EntityResolver | `block_floor` (min cosine for blocking) | `0.30` | `entity_resolver.py` |
| PPR kernel | `alpha` (damping) | `0.85` | `graph_db.py` |
| PPR kernel | `max_iter` | `100` | `graph_db.py` |
| PPR kernel | `tol` | `1e-9` | `graph_db.py` |
| Edge weight | SUPERSEDES / VERSION_OF | `3.0` | `graph_db.py` |
| Edge weight | DERIVED_FROM | `2.5` | `graph_db.py` |
| Edge weight | CONTRADICTS / DUPLICATES | `2.0` | `graph_db.py` |
| Edge weight | CONTAINS | `1.5` | `graph_db.py` |
| Edge weight | SIMILAR_TO (× confidence) | `1.0` | `graph_db.py` |
| Edge weight | HAS_TOPIC | `0.7` | `graph_db.py` |
| Edge weight | MENTIONS | `0.5` | `graph_db.py` |
| Edge weight | ON_DATE | `0.4` | `graph_db.py` |

---

## Key Design Invariants

1. **Detection and resolution stay decoupled.** Resolution policy is swappable without touching detection code.

2. **Detection always runs and always emits a signal** — even when resolution is fully automatic. The detection event stream feeds the consistency-score metric. Fusing detection into resolution would destroy the metric the moment a human is switched off.

3. **The LLM conflict judge is a shared component.** Every baseline feeds the same judge. Any benchmark difference must be attributable to candidate retrieval alone.

4. **The housekeeper never mutates the base.** It reads the vector index and docstore but never writes to them. It only maintains its own auxiliary structures (graph, event log, inverted indexes).

5. **Identity layer exists from day one.** The `Docstore` maps `doc_id → chunk_ids → vec_ids` so UPDATE can retire old chunks before committing new ones.

6. **One event at a time.** No batching. No concurrency guards. The leak rate under concurrent arrivals is an intentional headline metric, not a bug.

7. **DELETE is assumed always consistent.** No detection needed for deletes. Graph orphan detection (finding dependents of a deleted doc) is a roadmap item.

---

## Data Flow Walkthrough

**Inserting a new document:**

```
1. kb.ingest(IngestEvent(INSERT, doc_id, text))
2.   → chunk = Chunk(chunk_id=uuid(), doc_id, text)
3.   → emb = embedder.embed([chunk.text])[0]
4.   → candidates = housekeeper.find_candidates(chunk, emb)
          VectorHouseKeeper:    FAISS.search(emb, k=5) → top-5 existing chunks
          GraphHouseKeeper:     resolve entities → anchor lookup → PPR top-5
          HybridHouseKeeper:    seed(vec + entity + update-id) → directed PPR top-5
          MetadataHouseKeeper:  facet extract → inverted-index union → IDF rank top-5
5.   → result = judge.judge(chunk, candidates)
          if no candidates: return CLEAN (no LLM call)
          else: LLM classifies as clean/duplicate/contradiction/supersedes/needs_human
6a. CLEAN:    commit(chunk, emb)
                → vec_id = vdb.add(emb)
                → docstore.add(chunk)
                → housekeeper.on_commit(chunk, emb)
                     GraphHouseKeeper: add chunk node, freeze similar_to, log event
6b. CONFLICT: pending_embeddings[chunk.chunk_id] = emb
              conflict_sink(ConflictReport(report_id, chunk, result))
7.   → ResolutionManager.submit(report)
8.   → policy.resolve(report, apply_decision)
          RecencyPolicy: Decision(UPDATE, new_chunk=chunk, remove=implicated_ids)
          HumanPolicy:   queue and wait
9.   → kb.apply_decision(decision)
10.  → retire_chunks(implicated_ids)
          housekeeper.on_retire(chunk_ids) → revoke semantic edges → return successors
          vdb.remove + docstore.remove_chunk
11.  → commit(new_chunk, emb)  [emb from pending_embeddings cache]
     → housekeeper.on_resolution(decision, new_chunk)
          GraphHouseKeeper: add Decision node, assert SUPERSEDES_CONTENT edge
12.  [if revalidate_on_retire] re-judge successors from step 10
```
