# ADR 0007: Auditable diversity-aware post-selection

- Status: Accepted
- Date: 2026-09-10

## Context

Slice discovery can return equivalent or heavily overlapping rules. Showing
all of them as independent findings consumes attention without adding much
population coverage. Removing or reranking them in the fitted estimator would,
however, destroy search evidence and make downstream results hard to audit.

Membership equivalence and overlap depend on a reference population. Two rules
that are semantically different can be empirically equivalent on one frame and
different on another.

## Decision

Ginsu exposes post-selection as a separate immutable ``SliceSelection`` view.
It never changes ``slices_``, ``slice_statistics_``, legacy result attributes,
or discovery ranks.

Three deterministic methods are supported:

- ``score`` selects the first ``k`` discovery-ranked rules without using
  overlap as an exclusion criterion.
- ``unique_membership`` follows discovery rank and retains the first rule for
  each nonempty exact membership mask on the supplied reference frame.
- ``diverse`` follows discovery rank and greedily retains a nonempty rule only
  when its Jaccard overlap with every selected rule is at most
  ``max_jaccard``. Equality with the threshold is allowed.

The ordering therefore balances discovery score and overlap through a clear
lexicographic policy: score order is primary, and overlap is a hard acceptance
constraint. Ginsu does not synthesize an undocumented combined score.

Every discovered rule remains in the decision table. The table records
selection status and rank, the selected representative or overlap blocker,
maximum selected-rule Jaccard, reference support, candidate incremental
coverage, and cumulative selected coverage. Unique/diverse methods label empty
reference membership rather than selecting it. Capacity exclusions remain
visible.

Slice count, row-by-slice membership cells, and worst-case pair comparisons are
bounded before membership or pairwise work. The caller chooses and identifies
the reference population outside this API.

## Consequences

Reports can offer compact views without presenting excluded rules as absent
from discovery. Results are deterministic for a fixed fitted model, reference
row set, order, and configuration.

Selection does not establish statistical validity, causality, fairness, or
coverage on another population. A changed reference population requires a new
selection result. Stability across partitions and serialization of selection
evidence remain separate planned work.
