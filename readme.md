#### Online documentation: [lindt8.github.io/DCHAIN-Tools/](https://lindt8.github.io/DCHAIN-Tools/)

# DCHAIN Tools

This module is just a collection of Python 3 functions which serve to automatically process output from the DCHAIN-PHITS code, the nuclide activation, buildup, burnup, and decay code which is coupled to and distributed with the PHITS general purpose Monte Carlo particle transport code. These codes can be obtained at [https://phits.jaea.go.jp/](https://phits.jaea.go.jp/).

These functions are just tools I have developed over time to speed up my usage of PHITS and DCHAIN-PHITS; they are not officially supported by the PHITS development team.  They were developed to serve my own needs, and I am just publicly sharing them because others may also find utility in them.  I may more professionally repackage and redistribute these functions in the future in a more standard way.  For now, one may use the functions by first placing the dchain_tools.py Python script into a folder in their PYTHONPATH system variable or in the active directory and then just importing them normally ( from dchain_tools import * ).

Aside from the main DCHAIN output parsing function **process_dchain_simulation_output()**, dchain_tools.py also contains a handful other functions which individuals may find useful.  The dchain_tools_manual.pdf document primarily covers usage of this main function but provides brief descriptions of the other available functions.  All of these functions are documented online at [lindt8.github.io/DCHAIN-Tools/](lindt8.github.io/DCHAIN-Tools/).
