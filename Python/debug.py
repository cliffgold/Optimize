import numpy as np
import pandas as pd
from shared import df_int_to_float, nrgs
global debug_enabled

def setup(debug_option):
    """
    Setup function to initialize based on debug options.
    """
    debug_enabled = True # default is true, unless 'None'
    debug_filename = ''
    debug_matrix = pd.DataFrame()  # Initialize an empty DataFrame for debug matrix
    one_case_nrgs = pd.DataFrame()  # Initialize an empty DataFrame for one case energies

    match debug_option:
        case 'None':
            debug_enabled = False
        case 'debug_one_case':
            one_case_nrgs = pd.read_csv('Analysis/debug_knobs.csv')
            one_case_nrgs = df_int_to_float(one_case_nrgs)
            debug_matrix_columns = pd.Series \
                (['Year', 'Supply_var', 'Supply_matrix', 'Demand_var', 'Demand_matrix'])
            debug_filename       = 'Debug_One_Case'
            debug_matrix = pd.DataFrame(columns=debug_matrix_columns)    
        case 'debug_step_minimizer':
            debug_matrix_columns = pd.Series (['Year'] )
            for nrg in nrgs:
                debug_matrix_columns = pd.concat([debug_matrix_columns, pd.Series(['Knob_' + nrg])])

            debug_matrix_columns    = pd.concat([debug_matrix_columns, pd.Series(['Outage', 'Cost'])])
            debug_filename       = 'Debug_Step'
            debug_matrix = pd.DataFrame(columns=debug_matrix_columns)
        case 'debug_final_hourly':
    # Note that you can change the debug_matrix_columns to include more or less data
            debug_matrix_columns = pd.Series(['Hour', 'Year', 'Path', 'Hour_of_Need', 'Gas_Max', 'Gas_Used', 
                                        'Battery_Max','Battery_Used', 'Excess'])
            debug_filename = 'debug_final_hourly'
            debug_matrix = pd.DataFrame(columns=debug_matrix_columns)
        case 'debug_final_hourly':
            pass     
        case _:
            raise ValueError(f"Unknown debug option: {debug_option}")
    return debug_matrix, debug_filename, debug_enabled, one_case_nrgs

  
def debug_one_case_odd(debug_matrix, year, supply_MWh_nrgs, demand_MWh_nrgs, output_matrix):
    """
    Debug function for a single case.
    """
    debug_matrix.at[year * 2 + 1, 'Supply_var'] = supply_MWh_nrgs['Solar']
    debug_matrix.at[year * 2 + 1, 'Supply_matrix'] = \
        output_matrix.at[year, 'Solar_Supply_MWh']
    debug_matrix.at[year * 2 + 1, 'Demand_var'] = demand_MWh_nrgs['Solar']
    debug_matrix.at[year * 2 + 1, 'Demand_matrix'] = \
        output_matrix.at[year, 'Solar_Demand_MWh']
    
    return debug_matrix 

def debug_one_case_even(debug_matrix, year, supply_MWh_nrgs, demand_MWh_nrgs):
      # 'Year', 'Supply_Var', 'Supply_matrix', 'Demand_var', 'Demand_matrix']
    debug_matrix.at[year * 2 + 0, 'Year']          = year
    debug_matrix.at[year * 2 + 0, 'Supply_var']    = supply_MWh_nrgs['Solar'] 
    debug_matrix.at[year * 2 + 0, 'Supply_matrix'] = 0 
    debug_matrix.at[year * 2 + 0, 'Demand_var']    = demand_MWh_nrgs['Solar'] 
    debug_matrix.at[year * 2 + 0, 'Supply_matrix'] = 0

def debug_minimizer_add1(debug_matrix, results, fatol, xatol, end_time, call_time):
    """
    Debug function for the minimizer.
    """
    debug_matrix = pd.concat([debug_matrix, pd.Series(f'fatol {fatol} xatol {xatol}')])
    debug_matrix = pd.concat([debug_matrix, pd.Series(f'Knobs  {results.x}')])
    debug_matrix = pd.concat([debug_matrix, pd.Series(f'Results {results.fun:,.3f} Time {end_time - call_time:,.2f} with {results.nfev} runs')])
    
    return debug_matrix 
def debug_one_case_init(knobs_nrgs):
    """
    Init for each year's run
    """
    max_add_nrgs = pd.Series(999.,index=nrgs, dtype=float)
    start_knobs  = knobs_nrgs
    return knobs_nrgs, max_add_nrgs, start_knobs
 
def debug_minimizer_add2(debug_matrix, knobs, max_add_nrgs, bnds):
    """
    Adds debug information for the minimizer.
    """
    debug_matrix = pd.concat([debug_matrix, pd.Series([f'Start Knobs = {knobs}'])])
    debug_matrix = pd.concat([debug_matrix, pd.Series([f'Max Knobs = {max_add_nrgs}'])])
    debug_matrix = pd.concat([debug_matrix, pd.Series(bnds)])
    
    return debug_matrix 

def debug_step_minimizer(debug_matrix, max_add_nrgs, knobs_nrgs, year):
    row_debug = len(debug_matrix)
    debug_matrix.at[row_debug, 'Year'] = year * 100
    for nrg in nrgs:
        debug_matrix.at[row_debug, 'Knob_' + nrg] = knobs_nrgs[nrg]
        
    row_debug += 1
    debug_matrix.at[row_debug, 'Year'] = year * 100 + 1
    for nrg in nrgs:
        debug_matrix.at[row_debug, 'Knob_' + nrg] = max_add_nrgs[nrg]
