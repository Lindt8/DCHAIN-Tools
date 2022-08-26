'''
Created on Jun 12, 2020

@author: Hunter
'''
'''
This code serves as means of converting DCHAIN's .node, .ele, and .foam files into one .vtk file
'''

import numpy as np
import os
import time

import pyvista as pv
from pyvista import examples

# Timer start
start = time.time()

# Enter the path to the folder containing PHITS and DCHAIN files
data_file_folder = r"C:\path\to\your\DCHAIN_and_PHITS\simulation\files\\"

# Enter the filenames of the node (.node) and element (.ele) tetrahedral geometry files used by PHITS and the FOAM (.foam) file output by DCHAIN.
node_file    = 'bunny.node'
element_file = 'bunny.ele'
foam_file    = 'dchain_bunny_foamout1_t1.foam'

# The output and CSV files have the same basename as the .foam file, so this just replaces the extension
out_file     = foam_file.replace('foam','vtk')
csv_file     = foam_file.replace('.foam','.csv').replace('foamout1','foamout2')


node_file    = data_file_folder + node_file
element_file = data_file_folder + element_file
foam_file    = data_file_folder + foam_file
csv_file     = data_file_folder + csv_file
out_file     = data_file_folder + out_file

use_csv_values = True  # If True, this code will read the CSV file produced by DCHAIN.  If False, it will read the .foam file instead.
normalize_to_volume = True

# Select the index of the value you want to be included with the VTK file
# 0-11, value/rerr: 0/1 tot heat, 2/3 gamma heat, 4/5 beta heat, 6/7 alpha heat, 8/9 photon dose rate, 10/11 activity
ifoamval = 10  # 10 = activity


vtk_text = ''
vtk_text += '# vtk DataFile Version 2.0' + '\n'
vtk_text += 'Unstructured Grid' + '\n'
vtk_text += 'ASCII' + '\n'
vtk_text += 'DATASET UNSTRUCTURED_GRID' + '\n'

# Node data
x,y,z = [],[],[] # node coordinates
f = open(node_file)
lines = f.readlines()
first_line_read = False
for li,line in enumerate(lines):
    if len(line)<2: continue 
    if line[0]=='#': continue
    if not first_line_read: 
        n_nodes = int(line.split()[0])
        first_line_read = True
        continue 
    vals = line.split()
    x.append(vals[1])
    y.append(vals[2])
    z.append(vals[3])
f.close()

vtk_text += 'POINTS {:d} double'.format(n_nodes) + '\n'

for i in range(n_nodes):
    vtk_text += '{} {} {}'.format(x[i],y[i],z[i]) + '\n'

vtk_text += '\n'

# Cell/element data
vtc = [ [], [], [], [] ] # vertices of each node
f = open(element_file)
lines = f.readlines()
first_line_read = False
for li,line in enumerate(lines):
    if len(line)<2: continue 
    if line[0]=='#': continue
    if not first_line_read: 
        n_elements = int(line.split()[0])
        first_line_read = True
        continue 
    vals = line.split()
    for i in range(4):
        vtc[i].append(int(vals[i+1]))
f.close()


vtk_text += 'CELLS {:d} {:d}'.format(n_elements,5*n_elements) + '\n'
for i in range(n_elements):
    vtk_text += '4  {:5d} {:5d} {:5d} {:5d}'.format(vtc[0][i],vtc[1][i],vtc[2][i],vtc[3][i]) + '\n'

vtk_text += '\n'
vtk_text += 'CELL_TYPES {:d}'.format(n_elements) + '\n'
for i in range(n_elements):
    vtk_text += '10\n'


# Scalars / OpenFoam output from DCHAIN
s = [] # scalars
if use_csv_values:
    f = open(csv_file)
    lines = f.readlines()
    for li,line in enumerate(lines):
        if li==0: continue
        if len(line)<3: continue 
        vals = line.split(',')
        if ifoamval%2==0: # even, therefore value, not error
            if normalize_to_volume and ifoamval==8:
                val = np.float(vals[5+ifoamval])/np.float(vals[4])
            elif not normalize_to_volume and ifoamval!=8:
                val = np.float(vals[5+ifoamval])*np.float(vals[4])
            else:
                val = np.float(vals[5+ifoamval])
            s.append('{:11.4E}'.format(val))
        else:
            s.append(vals[5+ifoamval])
    f.close()
else:
    f = open(foam_file)
    lines = f.readlines()
    for li,line in enumerate(lines):
        if li==0: continue
        if len(line)<3: continue 
        s.append(line.strip())
    f.close()

vtk_text += '\n'
vtk_text += 'CELL_DATA {:d}'.format(n_elements) + '\n'
vtk_text += 'SCALARS decay_heat float' + '\n'
vtk_text += 'LOOKUP_TABLE default' + '\n'
for i in range(n_elements):
    vtk_text += '{}'.format(s[i]) + '\n'
vtk_text = vtk_text[:-1] # remove last newline character

f2 = open(out_file,"w+")
f2.write(vtk_text)
f2.close()


visualize_vtk = True
if visualize_vtk:
    pv.rcParams["use_panel"] = False
    vtkpath = out_file
    mesh = pv.UnstructuredGrid(vtkpath)
    # read the data
    grid = pv.read(vtkpath)
    
    # plot the data with an automatically created Plotter
    grid.plot(show_scalar_bar=True, show_axes=False)
    
    '''
    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True, color='white')
    p.add_mesh(pv.PolyData(mesh.points), color='red',
           point_size=1, render_points_as_spheres=True)
    p.camera_position = [(0.02, 0.30, 0.73),
                     (0.02, 0.03, -0.022),
                     (-0.03, 0.94, -0.34)]
    p.show(screenshot='bunny_nodes.png')
    '''

print("\nDone.   ({:0.2f} seconds elapsed)".format(time.time()-start)) 