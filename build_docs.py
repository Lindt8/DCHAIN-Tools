'''
This script serves to automate production of HTML documentation using [pdoc](https://github.com/pdoc3/pdoc) 
and organizing of files for builing of a release of DCHAIN Tools.

'''

import pdoc
import os

# DCHAIN Tools version number appearing in the documentation.
# https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-releases
VERSION_NUMBER = '1.5.1'

output_dir = "build/docs"
os.makedirs(output_dir, exist_ok=True)

# Build documentation following pdoc instructions: https://pdoc3.github.io/pdoc/doc/pdoc/#programmatic-usage
modules = [pdoc.import_module('dchain_tools.py')]
context = pdoc.Context()
modules = [pdoc.Module(mod, context=context) for mod in modules]
pdoc.link_inheritance(context)

def recursive_htmls(mod):
    yield mod.name, mod.html(sort_identifiers=False)
    for submod in mod.submodules():
        yield from recursive_htmls(submod)

for mod in modules:
    for module_name, html in recursive_htmls(mod):
        if module_name == 'dchain_tools':
            html_file_name = 'index.html'
        else:
            html_file_name = module_name + '.html'
        html = html.replace('width:70%;max-width:100ch;', 'width:70%;max-width:120ch;',1)  # make page contents wider
        if module_name == 'manage_mc_materials':
            html = html.replace('<h1 class="title">Module <code>dchain_tools</code></h1>','<h1 class="title">Submodule <code>dchain_tools</code></h1>',1)
        # add version number
        html = html.replace('</code></h1>','</code> <i><small>(v'+VERSION_NUMBER+')</small></i></h1>', 1)
        with open(os.path.join(output_dir, html_file_name), 'w') as f:
            f.write(html)

# Make CNAME file for use with custom domain name
with open(os.path.join(output_dir, "CNAME"), "w") as f:
    f.write("hratliff.com\n")