Diversity-aware selection
=========================

Post-selection creates a compact, auditable view of discovered rules without
changing the fitted estimator or its raw ranking.

.. code:: python

   diverse = finder.select_slices(
       X_reference,
       method="diverse",
       k=20,
       max_jaccard=0.80,
   )

   print(diverse.selected_slices)
   print(diverse.decisions)

The supplied frame defines empirical membership, equivalence, overlap, and
incremental coverage. Use a clearly identified reference population; results
can change when that population changes.

Selection methods
-----------------

``score`` selects the first ``k`` discovery-ranked rules. It preserves the
existing score-order behavior and does not exclude empty or overlapping
reference memberships.

``unique_membership`` walks discovery order and keeps the first rule for each
nonempty exact membership mask. Equivalent later rules identify the retained
representative.

``diverse`` also walks discovery order. It keeps a nonempty candidate only when
its Jaccard overlap with every already-selected rule is at most
``max_jaccard``. Equality with the threshold is accepted. This is a documented
lexicographic policy—discovery rank first, overlap constraint second—not a new
opaque score.

Audit table
-----------

``SliceSelection.decisions`` contains every discovered candidate in original
rank order. It records:

- reference support count and fraction;
- whether the candidate was selected and its selection rank;
- an explicit selection or exclusion status;
- its representative or blocking selected rule where applicable;
- maximum Jaccard overlap with selected rules;
- candidate incremental support beyond already covered rows; and
- cumulative support covered by accepted rules.

Exclusion statuses distinguish equivalence, excessive overlap, empty reference
membership, and capacity. ``selected_slices`` is a convenience view ordered by
selection rank. Neither table modifies ``finder.slices_`` or
``finder.slice_statistics_``.

Resource limits
---------------

``SelectionLimits`` bounds discovered slice count, membership cells, and
worst-case pair comparisons before expensive work begins:

.. code:: python

   from ginsu import SelectionLimits

   diverse = finder.select_slices(
       X_reference,
       method="diverse",
       limits=SelectionLimits(
           max_slices=500,
           max_membership_cells=5_000_000,
           max_pair_comparisons=100_000,
       ),
   )

Exceeding a limit raises ``AnalysisLimitError`` with a stable diagnostic code.
Post-selection is a reporting and prioritization tool; it is not statistical
validation, a causal claim, or evidence that the same coverage holds on a
different population.
