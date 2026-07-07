# Licensing Notes

## This Package

`flaggam` is released under the **Apache License 2.0** (see `LICENSE` in the repository root).

The implementation is an independent from-scratch realisation of the method described in the
papers listed below.  No source code, text, figures, or tables from either paper are reproduced
in this repository.

## FlagGAM Paper

The FlagGAM method is described in:

> Zhao, Z. and Welsch, R. E. (2026). *FlagGAM: Rule-Basis Generalized Additive Models for
> Explainable Tabular Prediction.* arXiv:2605.31189.

The preprint PDF (`FlagGAM.pdf`) is **not** committed to this repository and is treated as
all-rights-reserved.  No expression from that document has been reused in this codebase,
including documentation strings, comments, or variable names derived from verbatim PDF text.

## Univariate Flagging Algorithm (UFA) Paper

The Univariate Flagging Algorithm used in the screening step is described in:

> Sheth, M., Gerovitch, A., Welsch, R. E., Markuzon, N. (2019). *The Univariate Flagging
> Algorithm (UFA): An interpretable approach for predictive modeling.* PLOS ONE 14(10):
> e0223161. https://doi.org/10.1371/journal.pone.0223161

This paper is published under the **Creative Commons Attribution (CC BY) 4.0** licence.
The implementation follows the method as described; no text has been copied verbatim.

## Datasets

Dataset licences are the responsibility of the caller.  Each dataset loader (shipped in the
separate benchmarks plan) documents the applicable licence in its module docstring.  Users
must verify that their use of any dataset complies with its licence terms before running
experiments.
