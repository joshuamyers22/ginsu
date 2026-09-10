# Python 3.13 readiness

## Status

The inherited OptBinning blocker was removed on 2026-09-10 when the maintained
notebooks migrated to Ginsu's Polars-native ``DiscretizationPlan``. OptBinning,
Matplotlib, and their solver stack are no longer project dependencies.

This removes the known notebook dependency blocker but does not itself declare
Python 3.13 supported. Ginsu currently advertises and tests Python 3.10 through
3.12.

## Evidence required before adding Python 3.13

- Add Python 3.13 to the CI matrix with and without Numba.
- Run the complete correctness, type, documentation, coverage, notebook, and
  benchmark gates against the frozen lock.
- Build wheel and source distributions and smoke-test core, plotting, notebook,
  PyArrow, and pandas-interchange environments.
- Confirm the supported NumPy, SciPy, scikit-learn, Polars, Numba, and packaging
  ranges resolve on every supported operating system.
- Update classifiers, contributor documentation, and the release-readiness
  record in one reviewed change.

Until that evidence exists, Python 3.13 remains unclaimed rather than blocked
on a removed dependency.
