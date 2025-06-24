import numpy as np
import pandas as pd
import numpy_financial as npf
from scipy.optimize import minimize, Bounds
from math import log10, floor
import os 
import time
import warnings
import multiprocessing as mp
import traceback

warnings.filterwarnings('error',module=r'.*Optimize.*')

# Naming Rules:
# ending in _nrgs: vector in nrgs below
# Starting in hourly_: vector/matrix including sample years of hourly data

#Energy is in MWh
#Power is in MW.
#Cost is in M$
#CO2 is MTonne

#Globals

dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname + '/..')

# had to add to deletechars, they got inserted at the beginning of the first genfromtext entry.
delete_chars   = " !#$%&'()*+, -./:;<=>?@[\\]^{|}~﻿ï»¿"

nrgs           = np.array(['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal', 'Battery'])

others         = np.array(['Hydro', 'Oil', 'Other'])

# Output Matrix Columns
output_header = pd.Series(['Year', 'CO2_Price', 'Outage', 'Total_MW', 'Total_MWh', 'Total_Target', 'MW_Cost', 'MWh_Cost', 'Outage_Cost','CO2_Cost', 'MW_MWh_Outage_Cost', 'Including_CO2_Cost','Demand', 'Molten_Capacity', 'Molten_Used'])
param_order   = pd.Series(['MW','Demand_MWh', 'Supply_MWh', 'Cost', 'CO2_Cost', 'CO2_MTon', 'MW_Cost', 'MWh_Cost', 'Start_Knob', 'Knob', 'Max_Knob'])
tweaked_globals_order = pd.Series(['CO2_Price', 'Demand', 'Interest', 'Molten_Rate'])
tweaked_nrgs_order    = pd.Series(['Capital','Fixed', 'perMW', 'perMWh', 'Max_PCT', 'Lifetime', 'CO2_gen'])

# Sample years in the EIA data.  Will be set in get_eia_data().
sample_years = 0
sample_hours = 0
# used by debug, and other places 
def df_int_to_float(df):
# Identify integer columns and convert them to float
    int_cols = df.select_dtypes(include=['int']).columns
    df[int_cols] = df[int_cols].astype(float)
    return df

#************ Debug Options ************
# True = do not run parallel processes
kill_parallel = False

# run one set of knobs/Year - no minimize()
debug_one_case = True
if debug_one_case:
    one_case_nrgs = pd.read_csv('Analysis/debug_knobs.csv')
    one_case_nrgs = df_int_to_float(one_case_nrgs)
    debug_matrix_columns = pd.Series(['Count', 'Hour of Need', 'Gas_MWh', 'Battery_MWh', 'Molten_MWh', 'Outage_MWh'])
    print(one_case_nrgs['Solar'])
# True = print minimize results
debug_minimizer = False

# Print out on each step of the minimizer
debug_step_minimizer = True
if debug_step_minimizer:
    debug_matrix_columns = pd.Series(['Year'])
    for nrg in nrgs:
        debug_matrix_columns = pd.concat([debug_matrix_columns, pd.Series(['Knob_' + nrg])])

    debug_matrix_columns    = pd.concat([debug_matrix_columns, pd.Series(['Outage', 'Cost'])])
    debug_filename       = 'Debug_Step'


# Print out numbers that should not change in each year
debug_unexpected_change = False

# Save every hour in fig_gas_and_storage loop
debug_final_run = False
if debug_final_run:
    debug_matrix_columns = pd.Series(['Year', 'Path', 'Hour_of_Need', 'Gas_Max', 'Gas_Used', 
                                        'Battery_Max','Battery_Used', 'Excess'])
    debug_filename = 'Debug_final_run'
    
if(debug_one_case or debug_step_minimizer or debug_final_run):
    debug_matrix = pd.DataFrame(columns=debug_matrix_columns)

#******************** End of Globals ***************
#

# Save debug matrix
def save_matrix(file_name, matrix):
    if (matrix.size > 1):
        file_path = './Python/Mailbox/Outbox/' + file_name + '.csv'
        if os.path.exists(file_path):
            os.remove(file_path)
        matrix.to_csv(file_path)
      

# Get price, CO2 generated, etc for each nrg
def get_specs_nrgs():
    specs_nrgs = pd.read_csv('./csv/Specs.csv',
                         header=0, 
                         skiprows=1, 
                         index_col=0)
   
    specs_nrgs = df_int_to_float(specs_nrgs)
          
    return specs_nrgs

# Get parameters set in overlord by user
def get_inbox():
    inbox = pd.read_csv('./Python/Mailbox/Inbox.csv', 
                         header=0, 
                         index_col=0)
    inbox = df_int_to_float(inbox)
    
    return inbox

# Get list of regions
def get_all_regions():
    regions_temp = np.genfromtxt('./csv/Regions.csv', 
                              delimiter=',',
                              dtype=('U5, U20'), 
                              names=True,
                              deletechars=delete_chars)  
    return regions_temp['Abbr']
    
#There are two sets of EIA data for each region:
#   1. hourly_cap_pct_nrgs: Hourly capacity percentages used historicly for each nrg.  
#       These percentages do not change for each year run.
#   2. MW_nrgs: Maximum MW for each nrg.  
#       This starts at the maximum MWh for the historical data.
#       Then, it is adjusted each year as plants are retired, and new plants are built

#   Note that nrg sources "Hydro", "Oil" and "Other" are not used in the optimization.
#      They are assumed to grow with demand.
#   Also note that gas and battery are figured as fill-in at the end of each test run.
def get_eia_data(region):
    global sample_years, sample_hours

    eia_filename = f'{region}_master.csv'
    csv_path = f'./csv/Eia_Hourly/Latest/Hourly_Capacity_values/{eia_filename}'
    eia_cap_csv = pd.read_csv(csv_path,
                         header=0, 
                         skiprows=0)
    
    eia_filename = f'{region}_max_vals.csv'
    csv_path = f'./csv/Eia_Hourly/Latest/Max_MWh_values_yearly/{eia_filename}'  
    eia_max_csv = pd.read_csv(csv_path,
                         header=0, 
                         skiprows=0)
     
    # Set up the hourly capacity percentages and max MW for each nrg
    # Also total demand for each hour
    hourly_cap_pct_nrgs = pd.DataFrame(0,index=eia_cap_csv.index, columns=nrgs, dtype=float)
    MW_nrgs             = pd.Series(0, index=nrgs, dtype=float)
    sample_years        = len(eia_cap_csv)/(365.25 * 24) 
    sample_hours        = len(eia_cap_csv)
   
    for nrg in nrgs:
        if nrg == 'Battery':
            hourly_cap_pct_nrgs['Battery'] = pd.Series(0, index=eia_cap_csv.index)
            MW_nrgs['Battery']             = 0
        else:
            hourly_cap_pct_nrgs[nrg] = eia_cap_csv[nrg]
            # This .max() is to pick the maximum value for all years
            # Usually, this is the last year
            MW_nrgs[nrg]             = eia_max_csv[nrg].max()

    return hourly_cap_pct_nrgs, MW_nrgs

def init_output_matrix():
    output_header_loc = output_header    
    for nrg in nrgs:
        for param in param_order:
            output_header_loc = pd.concat([output_header_loc,pd.Series([nrg + '_' + param])], axis=0, ignore_index=True)                 
    output_matrix = pd.DataFrame(columns=output_header_loc, dtype=float)
    return output_matrix

# Initial values for year zero - Spec Numbers
def init_tweaks(specs_nrgs,inbox):
    tweaked_nrgs = pd.DataFrame(1,columns=nrgs, index=tweaked_nrgs_order, dtype=float) 
    for nrg in nrgs: 
        tweaked_nrgs.at['Capital', nrg] = specs_nrgs.at['Capital', nrg]
        tweaked_nrgs.at['Fixed', nrg]   = specs_nrgs.at['Fixed', nrg]   
        tweaked_nrgs.at['perMWh', nrg]  = specs_nrgs.at['Variable', nrg]
        tweaked_nrgs.at['perMW', nrg]   = specs_nrgs.at['Fixed', nrg] + \
                     (-4 * npf.pmt(inbox.at['Interest','Initial']/4, specs_nrgs.at['Lifetime', nrg]*4,specs_nrgs.at['Capital', nrg]))
        tweaked_nrgs.at['Lifetime', nrg] = specs_nrgs.at['Lifetime', nrg]
        tweaked_nrgs.at['Max_PCT', nrg]  = specs_nrgs.at['Max_PCT', nrg]
        tweaked_nrgs.at['CO2_gen', nrg]  = specs_nrgs.at['CO2_gen', nrg]
        
    tweaked_globals = pd.Series(0,index=tweaked_globals_order, dtype=float)
    tweaked_globals['CO2_Price']    = 0.
    tweaked_globals['Demand']       = 1
    tweaked_globals['Interest']     = 0.
    tweaked_globals['Molten_Rate']  = 0.

    return tweaked_globals, tweaked_nrgs

# Figure next year's info
def fig_tweaks(    
                tweaked_nrgs,
                tweaked_globals,
                inbox,
                year):

    if year == 1:
        loc_ = 'Initial'
        tweaked_globals['CO2_Price']   = inbox.at['CO2_Price', loc_]
        tweaked_globals['Demand']      = inbox.at['Demand', loc_] 
        tweaked_globals['Interest']    = inbox.at['Interest', loc_]
        tweaked_globals['Molten_Rate'] = inbox.at['Molten_Rate', loc_]
    else:
        loc_ = 'Yearly'
        tweaked_globals['CO2_Price']   += inbox.at['CO2_Price', loc_]
        tweaked_globals['Demand']      *= inbox.at['Demand', loc_] 
        tweaked_globals['Interest']    *= inbox.at['Interest', loc_] 
        tweaked_globals['Molten_Rate'] *= inbox.at['Molten_Rate', loc_] 
    
    for nrg in nrgs: 
        tweaked_nrgs.at['Capital', nrg]  *= inbox.at[nrg + '_Capital', loc_]
        tweaked_nrgs.at['Fixed', nrg]    *= inbox.at[nrg + '_Fixed', loc_]
        tweaked_nrgs.at['Lifetime', nrg] *= inbox.at[nrg + '_Lifetime', loc_]
        tweaked_nrgs.at['Max_PCT', nrg]  *= inbox.at[nrg + '_Max_PCT', loc_]
        tweaked_nrgs.at['perMWh', nrg]   *= inbox.at[nrg + '_Variable', loc_]
        
        tweaked_nrgs.at['perMW', nrg]     = tweaked_nrgs.at['Fixed', nrg] + \
                         (-4 * npf.pmt(tweaked_globals['Interest']/4, tweaked_nrgs.at['Lifetime', nrg]*4,tweaked_nrgs.at['Capital', nrg]))
                         # Note that this figures a quarterly payoff, 4 payments per year  
    return tweaked_globals, tweaked_nrgs

# Figure loss due to lifetime of plant
def fig_decadence(MW_nrgs, tweaked_nrgs):
    for nrg in nrgs:
        MW_nrgs[nrg]         *= 1 - (1/tweaked_nrgs.at['Lifetime', nrg])
    return MW_nrgs
    
# Gas fills any leftover need.  If not enough, storage.  If not enough, outage (VERY expensive)
def fig_gas_and_storage(hourly_MWh_needed,   
                        nuclear_hourly,  
                        gas_max, 
                        battery_max,
                        molten_max,
                        battery_stored,
                        molten_stored,
                        supply_MWh_nrgs,
                        tweaked_globals,
                        after_optimize, 
                        supercharge,
                        year):
    global debug_matrix
    global sample_years
    
    # This is for debugging.  Want final run of each year.
    if(after_optimize):
        break_me = 1
    
    gas_used     = 0.
    battery_used = 0.
    molten_used  = 0.
    outage_MWh   = 0.
    excess       = 0.
    hour         = 0.
    # Separate cases for each for loop    
    if (molten_max > 0 and supercharge):
            # Case of Molten with Supercharge  
            for hour_of_need in hourly_MWh_needed:
                #Already have too much NRG
                chargeable_molten = min(molten_max - molten_stored, nuclear_hourly[hour])
                if(hour_of_need < 0):
                    path                        = 'Molt + Super - Excess'
                    # How much can molten take?
                    molten_chargeable           = min(molten_max - molten_stored, nuclear_hourly[hour])
                    molten_charge               = min(molten_chargeable, -hour_of_need)
                    molten_stored              += molten_charge
                    supply_MWh_nrgs['Nuclear'] -= molten_charge
                    hour_of_need               += molten_charge
                    # Can battery take all the remaining excess
                    battery_charge              = min(battery_max - battery_stored, -hour_of_need)
                    excess                     += -hour_of_need - battery_charge
                        
                 # Enough gas for everybody - most common, does it help to go first?
                elif (hour_of_need <= gas_max):
                    path      = 'Molt + Super - Enough_Gas'
                    gas_used += hour_of_need
                    gas_left = gas_max - hour_of_need
                    battery_stored += min(battery_max-battery_stored, gas_left)
                    
               # Enough gas + molten to meet need
                elif (hour_of_need < gas_max + molten_stored):
                    path           = 'Molt + Super - Use_Molten'
                    gas_used      += gas_max
                    molten_stored -= hour_of_need - gas_max
                    molten_used   += hour_of_need - gas_max
                    
                # Enough gas + molten + battery to meet need
                elif (hour_of_need < gas_max + molten_stored + battery_stored):
                    path             = 'Molt + Super - Use_Molten+Battery'
                    gas_used        += gas_max
                    molten_used     += molten_stored
                    battery_stored  -= hour_of_need - gas_max - molten_stored
                    battery_used    += hour_of_need - gas_max - molten_stored
                    molten_stored    = 0.
                    
                # Not enough to meet need
                else:
                    path           = 'Molt + Super - UhOh'
                    outage_MWh    += hour_of_need - gas_max - battery_stored - molten_stored        
                    gas_used      += gas_max
                    battery_used  += battery_stored
                    molten_used   += molten_stored
                    battery_stored = 0.
                    molten_stored  = 0.
                                    
                if(debug_final_run and after_optimize):
                    row_debug_matrix = len(debug_matrix)
                    
                    debug_matrix.at[row_debug_matrix, 'Year']           = year
                    debug_matrix.at[row_debug_matrix, 'Path']           = path
                    debug_matrix.at[row_debug_matrix, 'Hour_of_Need']   = hour_of_need
                    debug_matrix.at[row_debug_matrix, 'Gas_Max']        = gas_max
                    debug_matrix.at[row_debug_matrix, 'Gas_Used']       = gas_used
                    debug_matrix.at[row_debug_matrix, 'Battery_Max']    = battery_max
                    debug_matrix.at[row_debug_matrix, 'Battery_Used']   = battery_used
                    debug_matrix.at[row_debug_matrix, 'Battery_Stored'] = battery_stored
                    debug_matrix.at[row_debug_matrix, 'Excess']         = excess
               
    elif (molten_max > 0):
            # Case of Molten without Supercharge  
            for hour_of_need in hourly_MWh_needed:
                #Already have too much NRG
                chargeable_molten = min(molten_max - molten_stored, nuclear_hourly[hour])
                if(hour_of_need < 0):
                    path                        = 'Molten - Excess'
                    # How much can molten take?
                    molten_chargeable           = min(molten_max - molten_stored, nuclear_hourly[hour])
                    molten_charge               = min(molten_chargeable, -hour_of_need)
                    molten_stored              += molten_charge
                    supply_MWh_nrgs['Nuclear'] -= molten_charge
                    hour_of_need               += molten_charge
                    # Can battery take all the remaining excess
                    battery_charge              = min(battery_max - battery_stored, -hour_of_need)
                    excess                     += -hour_of_need - battery_charge
                        
                 # Enough gas for everybody - most common, does it help to go first?
                elif (hour_of_need <= gas_max):
                    path = 'Molten - Enough_Gas'
                    gas_used += hour_of_need
                    
               # Enough gas + molten to meet need
                elif (hour_of_need < gas_max + molten_stored):
                    path           = 'Molten - Use_Molten'
                    gas_used      += gas_max
                    molten_stored -= hour_of_need - gas_max
                    molten_used   += hour_of_need - gas_max
                    
                # Enough gas + molten + battery to meet need
                elif (hour_of_need < gas_max + molten_stored + battery_stored):
                    path             = 'Molten - Use_Molten+Battery'
                    gas_used        += gas_max
                    molten_used     += molten_stored
                    battery_stored  -= hour_of_need - gas_max - molten_stored
                    battery_used    += hour_of_need - gas_max - molten_stored
                    molten_stored    = 0.
                    
                # Not enough to meet need
                else:
                    path           = 'Molten - UhOh'
                    outage_MWh    += hour_of_need - gas_max - battery_stored - molten_stored        
                    gas_used      += gas_max
                    battery_used  += battery_stored
                    molten_used   += molten_stored
                    battery_stored = 0.
                    molten_stored  = 0.
                                    
                if(debug_final_run and after_optimize):
                    row_debug_matrix = len(debug_matrix)
                    
                    debug_matrix.at[row_debug_matrix, 'Year']           = year
                    debug_matrix.at[row_debug_matrix, 'Path']           = path
                    debug_matrix.at[row_debug_matrix, 'Hour_of_Need']   = hour_of_need
                    debug_matrix.at[row_debug_matrix, 'Gas_Max']        = gas_max
                    debug_matrix.at[row_debug_matrix, 'Gas_Used']       = gas_used
                    debug_matrix.at[row_debug_matrix, 'Battery_Max']    = battery_max
                    debug_matrix.at[row_debug_matrix, 'Battery_Used']   = battery_used
                    debug_matrix.at[row_debug_matrix, 'Battery_Stored'] = battery_stored
                    debug_matrix.at[row_debug_matrix, 'Excess']         = excess

    elif (supercharge):
            # Case if No molten, with supercharge
            for hour_of_need in hourly_MWh_needed:
                #Already have too much NRG
                if(hour_of_need < 0):
                    path              = 'Super - Excess'
                    # Can battery take all the remaining excess, with some left over?
                    battery_charge = min(battery_max - battery_stored, -hour_of_need)
                    battery_stored += battery_charge
                    excess         += -excess - battery_charge
                        
                 # Enough gas for everybody - most common, does it help to go first?
                elif (hour_of_need <= gas_max):
                    path = 'Super - Enough_Gas'
                    gas_used += hour_of_need
                    battery_charge = min(battery_max - battery_stored, gas_max - hour_of_need)
                    battery_stored += battery_charge                    
                    
                # Enough gas + battery to meet need
                elif (hour_of_need < gas_max + battery_stored):
                    path             = 'Super - Use_Battery'
                    gas_used        += gas_max
                    battery_stored  -= hour_of_need - gas_max
                    battery_used    += hour_of_need - gas_max
                    molten_stored    = 0.
                    
                # Not enough to meet need
                else:
                    path           = 'Super - UhOh'
                    outage_MWh    += hour_of_need - gas_max - battery_stored - molten_stored        
                    gas_used      += gas_max
                    battery_used  += battery_stored
                    molten_used   += molten_stored
                    battery_stored = 0.
                    molten_stored  = 0.
                    
                if(debug_final_run and after_optimize):
                    row_debug_matrix = len(debug_matrix)
                   
                    debug_matrix.at[row_debug_matrix, 'Year']           = year
                    debug_matrix.at[row_debug_matrix, 'Path']           = path
                    debug_matrix.at[row_debug_matrix, 'Hour_of_Need']   = hour_of_need
                    debug_matrix.at[row_debug_matrix, 'Gas_Max']        = gas_max
                    debug_matrix.at[row_debug_matrix, 'Gas_Used']       = gas_used
                    debug_matrix.at[row_debug_matrix, 'Battery_Max']    = battery_max
                    debug_matrix.at[row_debug_matrix, 'Battery_Used']   = battery_used
                    debug_matrix.at[row_debug_matrix, 'Battery_Stored'] = battery_stored
                    debug_matrix.at[row_debug_matrix, 'Excess']         = excess

    else:
            # Case if No molten, no supercharge
            for hour_of_need in hourly_MWh_needed:
                hour = hour + 1
                #Already have too much NRG
                if(hour_of_need < 0):
                    path              = 'None - Excess'
                    # Can battery take all the remaining excess, with some left over?
                    battery_charge = min(battery_max - battery_stored, -hour_of_need)
                    battery_stored += battery_charge
                    excess         += -excess - battery_charge
                        
                 # Enough gas for everybody - most common, does it help to go first?
                elif (hour_of_need <= gas_max):
                    path = 'None - Enough_Gas'
                    gas_used += hour_of_need
                    
                # Enough gas + battery to meet need
                elif (hour_of_need < gas_max + battery_stored):
                    path             = 'None - Use_Battery'
                    gas_used        += gas_max
                    battery_stored  -= hour_of_need - gas_max
                    battery_used    += hour_of_need - gas_max
                    molten_stored    = 0.
                    
                # Not enough to meet need
                else:
                    path           = 'None - UhOh'
                    outage_MWh    += hour_of_need - gas_max - battery_stored - molten_stored        
                    gas_used      += gas_max
                    battery_used  += battery_stored
                    molten_used   += molten_stored
                    battery_stored = 0.
                    molten_stored  = 0.
                    
                if(debug_final_run and after_optimize):
                    row_debug_matrix = len(debug_matrix)
                    
                    debug_matrix.at[row_debug_matrix, 'Year']           = year
                    debug_matrix.at[row_debug_matrix, 'Path']           = path
                    debug_matrix.at[row_debug_matrix, 'Hour_of_Need']   = hour_of_need
                    debug_matrix.at[row_debug_matrix, 'Gas_Max']        = gas_max
                    debug_matrix.at[row_debug_matrix, 'Gas_Used']       = gas_used
                    debug_matrix.at[row_debug_matrix, 'Battery_Max']    = battery_max
                    debug_matrix.at[row_debug_matrix, 'Battery_Used']   = battery_used
                    debug_matrix.at[row_debug_matrix, 'Battery_Stored'] = battery_stored
                    debug_matrix.at[row_debug_matrix, 'Excess']         = excess

    supply_MWh_nrgs['Gas']     = gas_used     / sample_years
    supply_MWh_nrgs['Battery'] = battery_used / sample_years
    molten_used                = molten_used  / sample_years
    excess_MWh                 = excess       / sample_years
    outage_MWh                 = outage_MWh   / sample_years
   
    return supply_MWh_nrgs,     \
           outage_MWh,          \
           molten_stored,       \
           battery_stored,      \
           molten_used,         \
           excess_MWh

def fig_excess(supply_MWh_nrgs, excess_MWh, after_optimize):
    # This is for debugging.  Want final run of each year.
    if(after_optimize):
        break_me = 1

    demand_MWh_nrgs = supply_MWh_nrgs.copy()
    total_curtailable = 0.
    for nrg in ['Solar', 'Wind', 'Coal']:
        total_curtailable += supply_MWh_nrgs[nrg]

    for nrg in ['Solar', 'Wind', 'Coal']:
        excess = excess_MWh * supply_MWh_nrgs[nrg] / total_curtailable
        demand_MWh_nrgs[nrg]  -= excess
            
    return demand_MWh_nrgs
    
# add another year to the output matrix
def add_output_year(
                  demand_MWh_nrgs,            
                  MW_nrgs,
                  supply_MWh_nrgs,
                  tweaked_globals,
                  tweaked_nrgs,
                  expensive,
                  outage_MWh,
                  molten_max,
                  molten_used,
                  output_matrix,
                  year,
                  start_knobs,
                  knobs_nrgs,
                  max_add_nrgs,
                  target_hourly
                  ):
    global sample_years

    output_matrix.at[year, 'Year']            = year
    output_matrix.at[year, 'CO2_Price']       = tweaked_globals['CO2_Price']
    output_matrix.at[year, 'Outage']          = outage_MWh
    output_matrix.at[year, 'Demand']          = tweaked_globals['Demand']
    output_matrix.at[year, 'Molten_Capacity'] = molten_max
    output_matrix.at[year, 'Molten_Used']     = molten_used
    
    
    MW_cost     = 0.
    MWh_cost    = 0.
    total_CO2   = 0.
    total_MW    = 0.
    total_MWh   = 0.
    
    for nrg in nrgs:
        output_matrix.at[year, nrg + '_MW']         = MW_nrgs[nrg]  
        output_matrix.at[year, nrg + '_Demand_MWh'] = demand_MWh_nrgs[nrg].sum() / sample_years  
        output_matrix.at[year, nrg + '_Supply_MWh'] = supply_MWh_nrgs[nrg]
        output_matrix.at[year, nrg + '_MW_Cost']    = MW_nrgs[nrg]  * tweaked_nrgs.at['perMW', nrg]
        output_matrix.at[year, nrg + '_MWh_Cost']   = supply_MWh_nrgs[nrg] * tweaked_nrgs.at['perMWh', nrg]
        output_matrix.at[year, nrg + '_Cost']       = MW_nrgs[nrg]  * tweaked_nrgs.at['perMW', nrg]
        output_matrix.at[year, nrg + '_Cost']      += supply_MWh_nrgs[nrg] * tweaked_nrgs.at['perMWh', nrg]
        output_matrix.at[year, nrg + '_CO2_MTon']  = supply_MWh_nrgs[nrg] * tweaked_nrgs.at['CO2_gen', nrg]
        output_matrix.at[year, nrg + '_CO2_Cost']  = supply_MWh_nrgs[nrg] * tweaked_nrgs.at['CO2_gen', nrg] \
                                                       * tweaked_globals['CO2_Price']
        output_matrix.at[year, nrg + '_Start_Knob'] = start_knobs[nrg]
        output_matrix.at[year, nrg + '_Knob']       = knobs_nrgs[nrg]
        output_matrix.at[year, nrg + '_Max_Knob']   = max_add_nrgs[nrg]
        

        MW_cost   += MW_nrgs[nrg]          * tweaked_nrgs.at['perMW', nrg]
        MWh_cost  += supply_MWh_nrgs[nrg]  * tweaked_nrgs.at['perMWh', nrg]
        total_CO2 += supply_MWh_nrgs[nrg]  * tweaked_nrgs.at['CO2_gen', nrg]
        # Storage is really not a producer, and its MW is really MWh of capacity
        if (nrg != 'Battery'):
            total_MW    += MW_nrgs[nrg]
            total_MWh   += supply_MWh_nrgs[nrg]
    
    output_matrix.at[year, 'MW_Cost']            = MW_cost
    output_matrix.at[year, 'MWh_Cost']           = MWh_cost
    output_matrix.at[year, 'Outage_Cost']        = outage_MWh * expensive
    output_matrix.at[year, 'CO2_Cost']           = total_CO2  * tweaked_globals['CO2_Price'] 
    
    output_matrix.at[year, 'MW_MWh_Outage_Cost'] = output_matrix[['MW_Cost','MWh_Cost','Outage_Cost']].loc[year].sum()
    output_matrix.at[year, 'Including_CO2_Cost'] = output_matrix[['MW_MWh_Outage_Cost', 'CO2_Cost']].loc[year].sum()
    
    output_matrix.at[year, 'Total_MW']    = total_MW 
    output_matrix.at[year, 'Total_MWh']   = total_MWh
    output_matrix.at[year, 'Total_Target']= target_hourly.sum()
    return output_matrix

 # Save Output file.  Also called if minimizer error
def output_close(output_matrix, inbox, region):   
    file_name = f'{inbox.at["SubDir", "Text"]}-{region}'
    # minimized returned a really really small number for outage.  Excel couldn't handle it.
    # So rounding it to make that number 0.  Careful if you use really small numbers here.
    output_matrix_t = output_matrix.round(8).transpose()
    save_matrix(file_name, output_matrix_t)

# Cost function used by minimizer
def cost_function(     
                  MW_nrgs, 
                  supply_MWh_nrgs,
                  tweaked_globals,
                  tweaked_nrgs,  
                  expensive,     
                  outage_MWh,    
                  adj_zeros):    
    cost = 0.
    total_MWh_nrgs = pd.Series(0, index=nrgs, dtype=float)
    for nrg in nrgs:
        total_MWh_nrgs[nrg] = supply_MWh_nrgs[nrg].sum()

    for nrg in nrgs:
        cost += MW_nrgs[nrg]  * tweaked_nrgs.at['perMW', nrg]
        cost += total_MWh_nrgs[nrg] * tweaked_nrgs.at['perMWh', nrg]
        cost += total_MWh_nrgs[nrg] * tweaked_nrgs.at['CO2_gen', nrg] * tweaked_globals['CO2_Price']
        
    cost += outage_MWh * expensive
    cost += adj_zeros  * expensive
    return cost
    
# This updates the data based on knob settings.
#   Solar, Wind, Nuclear and Coal have nrg total multiplied by knob
#   Gas and (if necessary) storage is used to fill up to target value
#   Any excess is used to recharge storage. 

# If Supercharge is set, any available gas is used for storage 

# Note that all values must be passed as copies, so that
#   next run of optimize starts from the beginning

def update_data(
               knobs_nrgs,       
               hourly_cap_pct_nrgs,
               cap_pct_nrgs,        
               MW_nrgs,
               tweaked_globals,
               tweaked_nrgs,
               battery_stored,
               molten_stored,
               molten_max,      
               target_hourly, 
               zero_nrgs,
               after_optimize,
               supercharge,
               year):    
    
    # This is for debugging.  Want final run of each year.
    if(after_optimize):
        break_me = 1
        
    hourly_MWh_needed      = target_hourly.copy()
    MW_total               = MW_nrgs.sum()
    adj_zeros              = 0.
    hourly_supply_MWh_nrgs = hourly_cap_pct_nrgs.copy()
    supply_MWh_nrgs        = pd.Series(0, index=nrgs, dtype=float)

    for nrg in ['Solar','Wind','Nuclear','Coal']:
        if (zero_nrgs[nrg] == 0):
            MW_nrgs[nrg]                *= knobs_nrgs[nrg]
            hourly_supply_MWh_nrgs[nrg] *= MW_nrgs[nrg]
            supply_MWh_nrgs[nrg]         = cap_pct_nrgs[nrg] * MW_nrgs[nrg]
            if (knobs_nrgs[nrg] > 1):
                MW_nrgs[nrg] *= knobs_nrgs[nrg]
                if nrg == 'Nuclear':
                    molten_max += \
                        MW_nrgs['Nuclear'] * (knobs_nrgs['Nuclear'] - 1) * tweaked_globals.at['Molten_Rate']
                
            hourly_MWh_needed -= hourly_supply_MWh_nrgs[nrg]

        else:
            adj_zeros += knobs_nrgs[nrg]

    if (knobs_nrgs['Battery'] > 1):      
        MW_nrgs['Battery'] += (tweaked_nrgs.at['Max_PCT', 'Battery'] * MW_total * (knobs_nrgs['Battery'] - 1))
        battery_stored     += (tweaked_nrgs.at['Max_PCT', 'Battery'] * MW_total * (knobs_nrgs['Battery'] - 1))
        
    if (knobs_nrgs['Gas'] > 1):
        MW_nrgs['Gas']     = MW_nrgs['Gas'] * knobs_nrgs['Gas']

    supply_MWh_nrgs,     \
    outage_MWh,          \
    molten_stored,       \
    battery_stored,      \
    molten_used,         \
    excess_MWh           \
        =                \
        fig_gas_and_storage(
                hourly_MWh_needed   = hourly_MWh_needed,                   
                nuclear_hourly      = hourly_cap_pct_nrgs['Nuclear'],
                gas_max             = MW_nrgs['Gas'],
                battery_max         = MW_nrgs['Battery'],
                molten_max          = molten_max,
                battery_stored      = battery_stored,
                molten_stored       = molten_stored,
                supply_MWh_nrgs     = supply_MWh_nrgs,
                tweaked_globals     = tweaked_globals,
                after_optimize      = after_optimize,
                supercharge         = supercharge,
                year                = year)  
          
    if (after_optimize):
        demand_MWh_nrgs = fig_excess(supply_MWh_nrgs, excess_MWh, after_optimize)
    else:
        demand_MWh_nrgs = pd.Series(0, index=nrgs, dtype=float)
        
    return \
           demand_MWh_nrgs,\
           MW_nrgs,        \
           battery_stored, \
           molten_stored,  \
           molten_max,     \
           molten_used,    \
           adj_zeros,      \
           outage_MWh,     \
           supply_MWh_nrgs
          

# Main function used by minimizer              
def solve_this(
               knobs,               # Guess from minimizer                  
               hourly_cap_pct_nrgs, # Percentage of capacity used. Does not change
               cap_pct_nrgs,        # sum of percentages per nrg          
               MW_nrgs,             # Total capacity from last year
               battery_stored,      # battery stored from last year 
               molten_stored,       # molten stored from last year
               target_hourly,       # Target hourly demand for this year
               tweaked_globals,     # Global tweaks
               tweaked_nrgs,        # Tweaks for each nrg
               molten_max,          # Molten capacity from last year
               expensive,           # Cost of outage      
               zero_nrgs,           # Zeroed out nrgs
               supercharge,         # charge storage with gas if possible]
               year):               # Current year
               
    global debug_matrix
    knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)

# Must make a separate copy of these.  Otherwise, Python modifies the original.
# We need it to stay the same for the next minimize run 
         
    new_MW_nrgs             = MW_nrgs.copy()                 
    new_battery_stored      = battery_stored
    new_molten_stored       = molten_stored
    new_molten_max          = molten_max

    new_demand_MWh_nrgs, \
    new_MW_nrgs,         \
    new_battery_stored,  \
    new_molten_stored,   \
    new_molten_max,      \
    new_molten_used,     \
    adj_zeros,           \
    outage_MWh,          \
    supply_MWh_nrgs      \
        = update_data(
                      knobs_nrgs          = knobs_nrgs,           
                      hourly_cap_pct_nrgs = hourly_cap_pct_nrgs,
                      cap_pct_nrgs        = cap_pct_nrgs,    
                      MW_nrgs             = new_MW_nrgs,
                      tweaked_globals     = tweaked_globals,
                      tweaked_nrgs        = tweaked_nrgs,
                      battery_stored      = new_battery_stored,
                      molten_stored       = new_molten_stored,
                      molten_max          = new_molten_max,
                      target_hourly       = target_hourly,
                      zero_nrgs           = zero_nrgs,
                      after_optimize      = False,
                      supercharge         = supercharge,
                      year                = year)
                                 
    cost = cost_function(
               MW_nrgs         = new_MW_nrgs,
               supply_MWh_nrgs = supply_MWh_nrgs,
               tweaked_globals = tweaked_globals,
               tweaked_nrgs    = tweaked_nrgs,
               expensive       = expensive,
               outage_MWh      = outage_MWh,
               adj_zeros       = adj_zeros)

    if (debug_step_minimizer and year == 27):
        row_debug_matrix = len(debug_matrix)
        for nrg in nrgs:
            debug_matrix.at[row_debug_matrix, 'Knob_' + nrg]  = knobs_nrgs[nrg]
            
        debug_matrix.at[row_debug_matrix, 'Outage'] = outage_MWh
        debug_matrix.at[row_debug_matrix, 'Year']   = year
        debug_matrix.at[row_debug_matrix, 'Cost']   = cost
        
    return cost

# Initialize for year 1 starting place
def init_knobs(tweaked_globals, tweaked_nrgs):
    knobs_nrgs = pd.Series(1,index=nrgs, dtype=float)
    for nrg in nrgs:
        knobs_nrgs[nrg] = tweaked_globals['Demand'] + (1/tweaked_nrgs.at['Lifetime', nrg])
    return knobs_nrgs

def run_minimizer(    
                  hourly_cap_pct_nrgs,                 
                  cap_pct_nrgs,
                  MW_nrgs,
                  battery_stored,
                  molten_stored,
                  target_hourly,
                  tweaked_globals,
                  tweaked_nrgs,
                  molten_max,
                  expensive,               
                  zero_nrgs,
                  knobs_nrgs,
                  inbox,
                  region,
                  output_matrix,
                  year):
    
    global debug_matrix
    #This is total energy produced - Storage is excluded to prevent double-counting
    # Also note that MW_nrgs['*_Storage'] units are actually MWh of capacity.  Not even compatable.
    MW_total     = MW_nrgs.sum() - MW_nrgs['Battery']
    
    start_knobs  = pd.Series(1,index=nrgs, dtype=float)
    max_add_nrgs = pd.Series(1,index=nrgs, dtype=float)
    supercharge  = inbox.at['SuperCharge', 'Initial']
    for nrg in nrgs:
        if nrg == 'Battery':
            # Nominal for Storage is always half of max.
            max_add_nrgs['Battery'] = 2
        elif MW_nrgs[nrg] == 0.: 
            max_add_nrgs[nrg] = 10.
        else:    
            max_add_nrgs[nrg] = tweaked_globals['Demand'] + ((tweaked_nrgs.at['Max_PCT', nrg]*MW_total)/MW_nrgs[nrg])
            
        knobs_nrgs[nrg] = min(knobs_nrgs[nrg], max_add_nrgs[nrg] - .00001)
        start_knobs[nrg] = knobs_nrgs[nrg].copy()
        
    # and retire some old plants
    MW_nrgs = fig_decadence(MW_nrgs, tweaked_nrgs)

    if (debug_one_case):
        knobs_nrgs   = one_case(year)
        max_add_nrgs = pd.Series(999.,index=nrgs, dtype=float)
        start_knobs  = knobs_nrgs
    else:
        hi_bound = max_add_nrgs.copy()
        lo_bound = pd.Series(0.,index=nrgs, dtype=float)
        # Gas and Storage are as needed.  If knob < 1, is same as knob = 1 - no new capacity built
        lo_bound['Gas']     = 1.0
        lo_bound['Battery'] = 1.0
        bnds     = Bounds(lo_bound, hi_bound, True)
        method = 'Nelder-Mead'
        fatol  = .0001
        xatol = .00001
        rerun = .01
        opt_done = False
        try_count = 0
        last_result = 0.
        while(not(opt_done)):
            minimizer_failure = False
            call_time = time.time()
            knobs = pd.Series(knobs_nrgs).values
            if(debug_minimizer):
                debug_matrix = pd.concat(debug_matrix,pd.Series([f'Start Knobs = {knobs}']))
                debug_matrix = pd.concat(debug_matrix,pd.Series([f'Max Knobs = {max_add_nrgs}']))
                debug_matrix = pd.concat(debug_matrix,pd.Series([bnds]))
                
            if(debug_step_minimizer and year == 27):
                row_debug = len(debug_matrix)
                debug_matrix.at[row_debug, 'Year'] = year * 100
                for nrg in nrgs:
                    debug_matrix.at[row_debug, 'Knob_' + nrg] = knobs_nrgs[nrg]
                    
                row_debug += 1
                debug_matrix.at[row_debug, 'Year'] = year * 100 + 1
                for nrg in nrgs:
                    debug_matrix.at[row_debug, 'Knob_' + nrg] = max_add_nrgs[nrg]
        
            results =   minimize(
                        solve_this, 
                        knobs, 
                        args=(                 
                            hourly_cap_pct_nrgs,
                            cap_pct_nrgs,         
                            MW_nrgs,            
                            battery_stored,
                            molten_stored,
                            target_hourly,
                            tweaked_globals,
                            tweaked_nrgs,
                            molten_max,
                            expensive,               
                            zero_nrgs,
                            supercharge,
                            year
                           ),
                        bounds=bnds,                  
                        method=method, 
                        options={'fatol'  : fatol,
                                 'xatol'  : xatol,
                                 'maxiter': 10000,
                                 'maxfev' : 10000,
                                 'disp'   : False
                                }
            )
            end_time = time.time() 
        
            if not(results.success):
                print(f'{region} - Minimizer Failure')
                print(results)
                output_close(output_matrix, inbox, region)
                results_dict = {
                    'fun': [results.fun],
                    'x': [results.x],
                    'nit': [results.nit],
                    'nfev': [results.nfev],
                    'success': [results.success],
                    'status': [results.status],
                    'message': [results.message]
                    } 
                error_matrix = pd.DataFrame(results_dict)
                save_matrix(f'Minimizer_Failure-{region}', error_matrix)
                if (debug_one_case or debug_step_minimizer or debug_final_run):
                    save_matrix(debug_filename, debug_matrix)
                raise RuntimeError('Minimizer Failure' )
            
            elif(debug_minimizer):
                debug_matrix = pd.concat([debug_matrix, f'fatol {fatol} xatol {xatol}']) 
                debug_matrix = pd.concat([debug_matrix, f'Knobs  {results.x}']) 
                debug_matrix = pd.concat([debug_matrix, f'Results {results.fun:,.3f} Time {end_time - call_time:,.2f} with {results.nfev} runs']) 
            knobs      = results.x
            knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)   
            if ((last_result > (results.fun * (1-rerun))) and \
                (last_result < (results.fun * (1+rerun)))):
                opt_done = True
            else:
                if(last_result > 0):
                     print(f'{region} - Extra try at minimizer')
                last_result = results.fun
                fatol       = fatol/10.
                xatol       = xatol/10.
                         
    return knobs_nrgs, max_add_nrgs, start_knobs

def one_case(year):
    # This array reverses decadence - so starting place is the same
    #  redemption = np.array([1.034483,	1.041666,	1.02564,	1.04167,	1.02564,	0.])
    #  Optimized Case
    knobs_nrgs = one_case_nrgs.loc[year-1]
    return knobs_nrgs
def init_data(hourly_cap_pct_nrgs, MW_nrgs, specs_nrgs):
    # Initialize data for the zeroth year
    global sample_years, sample_hours

    cap_pct_nrgs      = pd.Series(0,index=nrgs, dtype=float)
    supply_MWh_nrgs   = pd.Series(0,index=nrgs, dtype=float)
    hourly_target_MWh = pd.Series(np.zeros(sample_hours, dtype=float))
    zero_nrgs         = pd.Series(0,index=nrgs, dtype=float)

    # Initialize based on EIA data, normalized to 1 year
    for nrg in nrgs:
        cap_pct_nrgs[nrg]    = hourly_cap_pct_nrgs[nrg].sum()
        supply_MWh_nrgs[nrg] = cap_pct_nrgs[nrg] * MW_nrgs[nrg] / sample_years
        hourly_target_MWh   += hourly_cap_pct_nrgs[nrg] * MW_nrgs[nrg] / sample_years   
        if (supply_MWh_nrgs[nrg] == 0) & (nrg != 'Battery'):
            zero_nrgs[nrg] = 1
            
    return cap_pct_nrgs, supply_MWh_nrgs, hourly_target_MWh, zero_nrgs

# main is a once-through operation, so we try to do as much calc here as possible
def do_region(region):
    global sample_years, sample_hours

    start_time    = time.time()
    inbox         = get_inbox()
    years         = inbox.at['Years', 'Initial']
    specs_nrgs    = get_specs_nrgs()
 
    hourly_cap_pct_nrgs, MW_nrgs = get_eia_data(region) 
# Initialize based on EIA data, normalized to 1 year
    cap_pct_nrgs,      \
    supply_MWh_nrgs,   \
    hourly_target_MWh, \
    zero_nrgs         \
        = init_data(hourly_cap_pct_nrgs, MW_nrgs, specs_nrgs)
    
    

    output_matrix = init_output_matrix()

    # Figure cost of outage - 100 times the cost per hour of every nrg
    avg_cost_per_hour = 0.
    for nrg in nrgs:
        avg_cost_per_hour += supply_MWh_nrgs[nrg].sum() * specs_nrgs.at ['Variable', nrg] / (365.25*24)
        avg_cost_per_hour += MW_nrgs[nrg]               * specs_nrgs.at['Fixed', nrg]    / (365.25*24)

    expensive = avg_cost_per_hour * 100
    
    battery_stored = 0.
    molten_stored  = 0.
    molten_used    = 0.
    molten_max     = 0.
    outage_MWh     = 0.
    target_hourly  = hourly_target_MWh.copy()
        
    tweaked_globals, tweaked_nrgs = init_tweaks(specs_nrgs, inbox)

#Output Year Zero
    knobs_nrgs  = pd.Series(1., index=nrgs, dtype=float)
    demand_MWh_nrgs = supply_MWh_nrgs.copy()
    output_matrix = \
                add_output_year(
                    demand_MWh_nrgs = demand_MWh_nrgs,                          
                    MW_nrgs         = MW_nrgs,
                    supply_MWh_nrgs = supply_MWh_nrgs,
                    tweaked_globals = tweaked_globals,
                    tweaked_nrgs    = tweaked_nrgs,
                    expensive       = expensive,
                    outage_MWh      = outage_MWh,
                    molten_max      = molten_max,
                    molten_used     = molten_used,
                    output_matrix   = output_matrix,
                    year            = 0,
                    start_knobs     = knobs_nrgs,
                    knobs_nrgs      = knobs_nrgs,
                    max_add_nrgs    = knobs_nrgs,
                    target_hourly   = target_hourly)
        
    knobs_nrgs = init_knobs(tweaked_globals=tweaked_globals, tweaked_nrgs=tweaked_nrgs)                
    if (years > 0):                                
        for year in range(1, int(years)+1):
            print(f'Year {year} in {region}')
# Update prices                       
            tweaked_globals, tweaked_nrgs = \
                fig_tweaks (
                        tweaked_globals = tweaked_globals,
                        tweaked_nrgs    = tweaked_nrgs,
                        inbox           = inbox,
                        year            = year)

            target_hourly = (target_hourly * tweaked_globals['Demand'])

# Now optimize this year 
            after_optimize = False           
            knobs_nrgs, max_add_nrgs, start_knobs = \
                run_minimizer( \
                                hourly_cap_pct_nrgs = hourly_cap_pct_nrgs,
                                cap_pct_nrgs        = cap_pct_nrgs,                  
                                MW_nrgs             = MW_nrgs, 
                                battery_stored      = battery_stored, 
                                molten_stored       = molten_stored,
                                target_hourly       = target_hourly,
                                tweaked_globals     = tweaked_globals,
                                tweaked_nrgs        = tweaked_nrgs,
                                molten_max          = molten_max,
                                expensive           = expensive,               
                                zero_nrgs           = zero_nrgs,
                                knobs_nrgs          = knobs_nrgs,
                                inbox               = inbox,
                                region              = region,
                                output_matrix       = output_matrix,
                                year                = year)

            after_optimize = True
# Update data based on optimized knobs 
            demand_MWh_nrgs,\
            MW_nrgs,        \
            battery_stored, \
            molten_stored,  \
            molten_max,     \
            molten_used,    \
            adj_zeros,      \
            outage_MWh,     \
            supply_MWh_nrgs \
                = update_data( 
                        knobs_nrgs          = knobs_nrgs,       
                        hourly_cap_pct_nrgs = hourly_cap_pct_nrgs,
                        cap_pct_nrgs        = cap_pct_nrgs,  
                        MW_nrgs             = MW_nrgs,
                        tweaked_globals     = tweaked_globals,
                        tweaked_nrgs        = tweaked_nrgs,
                        battery_stored      = battery_stored,
                        molten_stored       = molten_stored,
                        molten_max          = molten_max,
                        target_hourly       = target_hourly,
                        zero_nrgs           = zero_nrgs,
                        after_optimize      = after_optimize,
                        supercharge         = inbox.at['SuperCharge', 'Initial'],
                        year                = year)

# Output     results of this year             
            output_matrix = \
                add_output_year(
                  demand_MWh_nrgs = demand_MWh_nrgs,         
                  MW_nrgs         = MW_nrgs,
                  supply_MWh_nrgs = supply_MWh_nrgs,
                  tweaked_globals = tweaked_globals,
                  tweaked_nrgs    = tweaked_nrgs,
                  expensive       = expensive,
                  outage_MWh      = outage_MWh,
                  molten_max      = molten_max,
                  molten_used     = molten_used,
                  output_matrix   = output_matrix,
                  year            = year,
                  start_knobs     = start_knobs,
                  knobs_nrgs      = knobs_nrgs,
                  max_add_nrgs    = max_add_nrgs,
                  target_hourly   = target_hourly)

    # End of years for loop
    output_close(output_matrix, inbox, region)
    if (debug_one_case or debug_step_minimizer or debug_final_run):
        save_matrix(debug_filename, debug_matrix)
    print(f'{region} Total Time = {(time.time() - start_time)/60:.2f} minutes')
    
# Copied from Stack Overflow:

class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

def main():
    inbox         = get_inbox()
    region        = inbox.at['Region', 'Text']
    
    print('Starting ' + ' ' + inbox.at['SubDir', 'Text'] + ' CO2-' \
          + str(int(inbox.at['CO2_Price','Initial'])) + '_' 
          + str(int(inbox.at['CO2_Price','Yearly'])))
           
    if (not kill_parallel) and (region == 'US'):
        regions = get_all_regions()
        region_process = pd.Series(0,index=regions,dtype=object)

        for region in regions:
    # Create new child process for each region
            region_process[region] = Process(target=do_region, args=(region,))
            region_process[region].start()
    
        for region in regions:
    # Now, wait for all of them to be done
            region_process[region].join()
            if(region_process[region].exception):
                error_matrix = pd.Series([f'Main Error in {region}'])
                error, traceback = region_process[region].exception
                error_matrix = pd.concat(error_matrix, pd.Series ([error]))
                error_matrix = pd.concat(error_matrix, pd.Series ([traceback]))
                save_matrix(f'Main_error-{region}', error_matrix)
                print(f'error {error} in {region}')
                print(traceback)
            else:
                print(region + ' Done')
                
    # kill_parallel True or not 'US'
    elif region == 'US':
        regions = get_all_regions()
        for region in regions:
            do_region(region)
                         
    else: 
        do_region(region)
                
if __name__ == '__main__':
    main()
# This is the main entry point for the optimization script.