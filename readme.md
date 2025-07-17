#### Online DCHAIN Tools documentation: [lindt8.github.io/DCHAIN-Tools/](https://lindt8.github.io/DCHAIN-Tools/)

# DCHAIN Tools
[![Documentation](https://img.shields.io/badge/Documentation-brightgreen)](https://lindt8.github.io/DCHAIN-Tools/)
[![status](https://joss.theoj.org/papers/ef67acccadb883867ba60dc9e018ff70/status.svg)](https://joss.theoj.org/papers/ef67acccadb883867ba60dc9e018ff70)
[![PyPI - Version](https://img.shields.io/pypi/v/PHITS-Tools?logo=pypi&logoColor=fff&label=PyPI)](https://pypi.org/project/PHITS-Tools/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14267236.svg)](https://doi.org/10.5281/zenodo.14267236)

## Purpose

This module is a collection of Python 3 functions which serve to automatically process output from the DCHAIN-PHITS code, the nuclide activation, buildup, burnup, and decay code which is coupled to and distributed with the PHITS general purpose Monte Carlo particle transport code. These codes can be obtained at [https://phits.jaea.go.jp/](https://phits.jaea.go.jp/).

## Installation

DCHAIN Tools is distributed as a submodule of [PHITS Tools](https://github.com/Lindt8/PHITS-Tools) and is primarily intended to be accessed as such, though it is still fully functional on its own.

### With `pip` (Python >= 3.10) _(recommended)_

Install PHITS Tools:

`pip install PHITS-Tools`

Import DCHAIN Tools as a submodule of PHITS Tools like any other Python submodule:

`from PHITS_tools import dchain_tools` / `from PHITS_tools.dchain_tools import *`

### Manually 

One may use the functions by first placing the `dchain_tools.py` Python script into a folder in their `PYTHONPATH` system variable or in the active directory and then just importing them normally ( `import dchain_tools` / `from dchain_tools import *` ).

## Usage

Aside from the main DCHAIN output parsing function [**`process_dchain_simulation_output()`**](https://lindt8.github.io/DCHAIN-Tools/#dchain_tools.process_dchain_simulation_output), `dchain_tools.py` also contains additional functions which one may find useful.  The `dchain_tools_manual.pdf` document primarily covers usage of this main function but provides brief descriptions of the other available functions.  All of these functions are documented online at [lindt8.github.io/DCHAIN-Tools/](https://lindt8.github.io/DCHAIN-Tools/).

I have also written a similar module for (nearly universally) parsing and processing general PHITS output files (both normal tally output and tally dump files) called [PHITS Tools](https://github.com/Lindt8/PHITS-Tools/), which will also call DCHAIN Tools if provided with DCHAIN-related files to process and includes DCHAIN Tools as a submodule.

-----

These functions are just tools I have developed over time to speed up my usage of PHITS and DCHAIN-PHITS; they are not officially supported by the PHITS development team. 

All of the professionally-relevant Python modules I have developed are summarized [here](https://lindt8.github.io/professional-code-projects/), and more general information about me and the work I do / have done can be found on [my personal webpage](https://lindt8.github.io/).
