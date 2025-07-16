import numpy as np
import pandas as pd
from shared import df_int_to_float, nrgs


def setup(debug_option):
    """
    Setup function to initialize based on debug options.
    """
    global debug_enabled
    debug_enabled = True # default is true, unless 'None'
    one_case_nrgs  = pd.DataFrame(columns = nrgs)
    debug_matrix   = pd.DataFrame()
    debug_filename = ''
    
    match debug_option:
        case 'None':
            debug_enabled = False
            debug_matrix = pd.DataFrame()
            debug_filename = ''

        case 'debug_one_case':
            one_case_nrgs        = pd.read_excel('Analysis/One_Case_Nrgs.xlsx')
    #        one_case_nrgs        = df_int_to_float(one_case_nrgs)
            debug_matrix_columns = pd.Series \
                (['Year', 'Excess MWh', 'Total_Curtailed'])
            debug_filename       = 'Debug_One_Case'
            debug_matrix         = pd.DataFrame(columns=debug_matrix_columns)

        case 'debug_step_minimizer':
            debug_matrix_columns= pd.Series(['Year','Outage', 'Cost'])
            for nrg in nrgs:
                debug_matrix_columns = pd.concat([debug_matrix_columns, pd.Series(['Knob_' + nrg])])
                debug_matrix_columns = pd.concat([debug_matrix_columns, pd.Series(['MWh_' + nrg])])
            debug_filename       = 'Debug_Step'
            debug_matrix = pd.DataFrame(columns=debug_matrix_columns)
            
        case 'debug_final_hourly':
    # Note that you can change the debug_matrix_columns to include more or less data
            debug_matrix_columns = pd.Series(['Hour', 'Year', 'Path', 'Hour_of_Need', 
                                                'Gas_Max', 'Gas_Used', 
                                                'Battery_Max','Battery_Used', 'Excess'])
            debug_filename = 'debug_final_hourly'
            debug_matrix = pd.DataFrame(columns=debug_matrix_columns)
   
        case _:
            raise ValueError(f"Unknown debug option: {debug_option}")

    return debug_matrix, debug_filename, debug_enabled, one_case_nrgs

def debug_one_case_year(debug_matrix, year, excess_MWh, total_curtailed):
    debug_matrix.at[year, 'Excess MWh']      = excess_MWh
    debug_matrix.at[year, 'Total Curtailed'] = total_curtailed
    
    return debug_matrix 

def debug_minimizer_add1(debug_matrix, results, fatol, xatol, end_time, region):
    """
    Debug function for the minimizer.
    """
    debug_matrix = pd.concat([debug_matrix, pd.Series(f'fatol {fatol} xatol {xatol}')])
    debug_matrix = pd.concat([debug_matrix, pd.Series(f'Knobs  {results.x}')])
    debug_matrix = pd.concat([debug_matrix, pd.Series(f'Results {results.fun:,.3f} Time {end_time:,.2f} with {results.nfev} runs')])
    return debug_matrix 

def debug_minimizer_add2(debug_matrix, knobs, max_add_nrgs, bnds):
    """
    Adds debug information for the minimizer.
    """
    debug_matrix = pd.concat([debug_matrix, pd.Series([f'Start Knobs = {knobs}'])])
    debug_matrix = pd.concat([debug_matrix, pd.Series([f'Max Knobs = {max_add_nrgs}'])])
    debug_matrix = pd.concat([debug_matrix, pd.Series(bnds)])
    
    return debug_matrix 

def debug_step_minimizer(debug_matrix, year, outage_MWh, knobs_nrgs, MWh_nrgs, cost):
    row_debug = len(debug_matrix)
    for nrg in nrgs:
        debug_matrix.at[row_debug, 'Knob_' + nrg] = knobs_nrgs[nrg]
        debug_matrix.at[row_debug, 'MWh_' + nrg] = MWh_nrgs[nrg]        

    debug_matrix.at[row_debug, 'Year']      = year    
    debug_matrix.at[row_debug, 'Outage']    = outage_MWh
    debug_matrix.at[row_debug, 'Cost']      = cost
    if (row_debug > 100):
        abort = True
    else:
        abort = False
    return debug_matrix, abort


