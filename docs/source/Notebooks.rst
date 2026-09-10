Offline notebook tutorials
==========================

The two maintained notebooks are deterministic and network-independent. They
use synthetic Titanic-style classification and California-Housing-style
regression fixtures; they do not download or reproduce those source datasets.
This keeps the examples executable from a clean checkout and makes their seeded
behavior suitable for continuous integration.

Both examples keep model training, slice discovery, and fixed-rule validation
in disjoint partitions. Fitted ``DiscretizationPlan`` objects learn only from
discovery features and are reused unchanged on validation rows. The notebooks
do not use error-supervised binning or present discovery lift as significance.

The classification tutorial covers canonical Polars inputs, fixed-rule
validation, diversity selection, stable-ID membership, impact, observed error
dependence, and search diagnostics. The regression tutorial covers two-model
``SliceAnalysis`` comparison, declared discovery subsamples, stability, and
their bounded visualizations.

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
