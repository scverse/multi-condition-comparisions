> [!NOTE]
> The functionality developed in this repo has now been integrated into [pertpy](https://github.com/theislab/pertpy).
>
> This repo holds prototypes for a differential expression interface for scverse that has been developed at the [scverse hackathon in Cambridge, UK in November 2023](https://scverse.org/events/2023_11_hackathon/). It has now been archived as read-only. 

# multi-condition-comparisions

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/grst/multi-condition-comparisions/test.yaml?branch=main
[link-tests]: https://github.com/scverse/multi-condition-comparisons/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/multi-condition-comparisions

Functions for analyzing and visualizing multi-condition single-cell data (prototypes created at Cambridge hackathon)

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install multi-condition-comparisions:

<!--
1) Install the latest release of `multi-condition-comparisions` from `PyPI <https://pypi.org/project/multi-condition-comparisions/>`_:

```bash
pip install multi-condition-comparisions
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/scverse/multi-condition-comparisions.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/grst/multi-condition-comparisions/issues
[changelog]: https://multi-condition-comparisions.readthedocs.io/latest/changelog.html
[link-docs]: https://multi-condition-comparisions.readthedocs.io
[link-api]: https://multi-condition-comparisions.readthedocs.io/latest/api.html
