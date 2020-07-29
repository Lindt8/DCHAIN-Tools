from dchain_tools import *

 
test_folder = r'' # in this case, Python script is in same folder as DCHAIN output; otherwise specify path to DCHAIN output folder
test_basename = 'example_Na22' # base name of DCHAIN output files without extension

output = process_dchain_simulation_output(test_folder,test_basename)

print('\n')
 
print('Output times relative to start (in seconds):',output['time']['from_start_sec'])
print('Output times relative to EOB (in seconds):  ',output.time.from_EOB_sec)

print('\n')

print('Total neutron flux in first region (in n/cm^2/sec):',output['region']['neutron_flux'][0])
print('Fractional uncertainty in total neutron flux in first region:',output['neutron']['total_flux']['error'][0]/output['neutron']['total_flux']['value'][0])

print('\n')

print('All produced nuclides (in DCHAIN-formatted strings):',output['nuclides']['names'])
#print(np.shape(np.array(output['gamma']['spectra']['flux']['value'])))
#print(output['gamma']['spectra']['flux']['value'][0][0,:])

#pprint.pprint(output['top10']['activity']['rank'][0][0,:])

print('\n')

print('[T-Yield] high energy reaction-produced nuclides:',output['yields']['names'][0])
print('[T-Yield] nuclide yields per source particle:',output['yields']['unit_rate']['value'][0])

print('\n')

library_file = r'C:\phits\dchain-sp\data\JEFF-3-3--_n_act_xs_lib'
#xs_out, rxn_str_out = retrieve_rxn_xs_from_lib(library_file,'F-19','p')
xs_1g = calc_one_group_nrxn_xs_dchain(output['neutron']['unit_spectra']['flux']['value'][0],output['neutron']['unit_spectra']['flux']['error'][0],library_file,'F-19','p')
print('Single group cross section (and its absolute uncertainty) for 19F(n,p)19O reaction in provided neutron flux (in barns): ',xs_1g)


figs = plot_top10_nuclides(output,region_indices=1,rank_cutoff=5,xaxis_val='time',xaxis_type='indices',xscale='linear')

plt.figure(2)
region_index = 0
Na22_index = output['nuclides']['names'][region_index].index(nuclide_plain_str_to_Dname('Na-22')) # determine index in nuclide list for desired nuclide, Na-22.
A_Na22 = output.nuclides.activity.value[region_index][:,Na22_index]
times = output['time']['from_start_sec']
plt.plot(times,A_Na22)
plt.xlabel('time from start [seconds]',fontsize=12)
plt.ylabel(nuclide_plain_str_to_latex_str('Na-22')+r' activity [Bq/cm$^3$]',fontsize=12)
plt.xscale('log')
plt.yscale('log')


plt.show()