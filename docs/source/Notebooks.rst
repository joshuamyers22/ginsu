Examples and notebooks
======================

The two maintained notebooks are deterministic and network-independent. They
use synthetic Titanic-style classification and California-Housing-style
regression fixtures; they do not download or reproduce those source datasets.
This keeps the examples executable from a clean checkout and makes their seeded
behavior suitable for continuous integration.

Both examples keep model training, slice discovery, and fixed-rule validation
in disjoint partitions. Fitted ``DiscretizationPlan`` objects learn only from
discovery features and are reused unchanged on validation rows. The notebooks
do not use error-supervised binning or present discovery lift as significance.

Classification: diagnose a Titanic-style model
----------------------------------------------

``1. Implementing Ginsu on Titanic dataset.ipynb`` uses a synthetic
passenger-style classification fixture. It covers canonical Polars inputs,
fixed-rule validation, diversity selection, stable-ID membership, impact,
observed error dependence, and search diagnostics.

:download:`Download the classification notebook
<../../notebooks/1. Implementing Ginsu on Titanic dataset.ipynb>`.

Regression: compare California-Housing-style models
---------------------------------------------------

``2. Implementing Ginsu on California housing dataset.ipynb`` uses a synthetic
housing-style regression fixture. It compares two exhaustive
``SliceAnalysis`` objects, builds caller-declared discovery subsamples, checks
stability, and renders bounded comparison and search-diagnostic views.

:download:`Download the regression notebook
<../../notebooks/2. Implementing Ginsu on California housing dataset.ipynb>`.

Run the examples
----------------

Run the maintained examples with:

.. code:: sh

   make execute-notebooks

Execution uses the ``notebooks`` and ``plot`` extras and writes generated
copies beneath ignored ``docs/build/notebooks``. Checked-in notebooks retain
empty outputs and null execution counts. The dedicated notebook dependency set
does not include pandas, PyArrow, OptBinning, or Matplotlib.

Real projects must replace the synthetic fixtures with governed data and define
their own sampling unit, leakage boundary, validation population, privacy
controls, row identity, and provenance. Notebook results are descriptive API
examples, not deployment guarantees.
