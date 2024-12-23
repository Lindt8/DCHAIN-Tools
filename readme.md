#### Online documentation: [lindt8.github.io/DCHAIN-Tools/](https://lindt8.github.io/DCHAIN-Tools/)

# DCHAIN Tools
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14267236.svg)](https://doi.org/10.5281/zenodo.14267236)

This module is a collection of Python 3 functions which serve to automatically process output from the DCHAIN-PHITS code, the nuclide activation, buildup, burnup, and decay code which is coupled to and distributed with the PHITS general purpose Monte Carlo particle transport code. These codes can be obtained at [https://phits.jaea.go.jp/](https://phits.jaea.go.jp/).

These functions are just tools I have developed over time to speed up my usage of PHITS and DCHAIN-PHITS; they are not officially supported by the PHITS development team.  One may use the functions by first placing the dchain_tools.py Python script into a folder in their PYTHONPATH system variable or in the active directory and then just importing them normally ( `from dchain_tools import *` ).

Aside from the main DCHAIN output parsing function **`process_dchain_simulation_output()`**, dchain_tools.py also contains a handful other functions which individuals may find useful.  The dchain_tools_manual.pdf document primarily covers usage of this main function but provides brief descriptions of the other available functions.  All of these functions are documented online at [lindt8.github.io/DCHAIN-Tools/](https://lindt8.github.io/DCHAIN-Tools/).

I have also written a similar module for (nearly universally) parsing and processing general PHITS output files (both normal tally output and tally dump files) called [PHITS Tools](https://github.com/Lindt8/PHITS-Tools/), which will also call DCHAIN Tools if provided with DCHAIN-related files to process and includes DCHAIN Tools as a submodule.

All of the professionally-relevant Python modules I have developed are summarized [here](https://lindt8.github.io/professional-code-projects/), and more general information about me and the work I do / have done can be found on [my personal webpage](https://lindt8.github.io/).
