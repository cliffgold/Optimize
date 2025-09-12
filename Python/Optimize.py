import pandas as pd
import numpy as np
import warnings
import numpy_financial as npf
from scipy.optimize import minimize, Bounds, differential_evolution
from math import log10, floor
import os 
import time
import warnings
import multiprocessing as mp
import traceback
from numba import jit, njit
 

warnings.filterwarnings('error',module=r'.*Optimize.*')

# Naming Rules:
# ending in _nrgs: vector in nrgs below
# Starting in hourly_: vector/matrix including sample years of hourly data
# Starting in yr: Scaled by sample_years

#Energy is in MWh
#Power is in MW.
#Cost is in M$
#CO2 is MTonne (MT)

#Globals

dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname + '/..')

# had to add to deletechars, they got inserted at the beginning of the first genfromtext entry.
delete_chars   = " !#$%&'()*+, -./:;<=>?@[\\]^{|}~﻿ï»¿"

# energy (nrg) representations
nrgs          = np.array(['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal', 'Battery'])
nrg_sources   = np.array(['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal'])
nrg2nrgx_lu   = {'Solar' : 0, 'Wind' : 1, 'Nuclear' : 2, 'Gas' : 3, 'Coal' : 4, 'Battery' : 5}
nrgx2nrg_lu   = {0 : 'Solar', 1 : 'Wind', 2 : 'Nuclear', 3 : 'Gas', 4 : 'Coal', 5 : 'Battery'}
nrgxs         = np.array([0, 1, 2, 3, 4, 5])
nrgx_sources  = np.array([0, 1, 2, 3, 4])
battery_nrgx  = 5

# Output Matrix Columns
# First group is total for all nrgs. 
output_header = pd.Series(['Year', 'CO2_M$_MT', 'Target_MWh', 'Outage_MWh', 
                           'Outage_M$_MWh', 'Iterations'])

param_order   = pd.Series(['MW', 'MWh', 'Capital_M$', 'Fixed_M$',
                           'Variable_M$', 'CO2_M$', 
                           'Start_Knob', 'Optimized_Knob', 'Max_Knob',
                           'Decadence', 'Avg_Cap_Factor'])

tweaked_globals_order = pd.Series(['CO2_M$_MT', 'Demand', 'Interest'])
tweaked_nrgs_order    = pd.Series(['Capital_M$_MW','Fixed_M$_MW', 'Variable_M$_MWh', 
                                   'Max_PCT', 'Lifetime_years', 'CO2_MT_MWh'])

# These are used all over the place.  get_eia_data fills them. Just lazy.
sample_years = 0
sample_hours = 0
first_year   = 0

#************ Debug Options ************
#Select from the following options to debug
# None                         Normal mode - No Debug output

# debug_one_case               Run one set of knobs/Year - no minimize()
# debug_step_minimizer         Save data on each run of solve_this
# debug_unexpected_change      Print out numbers that should not change in each year
# debug_final_hourly           Save every hour in fig_hourly final run

import debug

debug_option = 'None'  # Set to None, debug_one_case, debug_step_minimizer, debug_unexpected_change, or debug_final_hourly

debug_matrix, debug_filename, debug_enabled, one_case_nrgs = debug.setup(debug_option, nrgs)
debug_count = 0

# kill_parallel                Do not run parallel processes
kill_parallel = False

#******************** End of Globals ***************
#
# convert nrgs to numbers for use in optimized fig_hourly.
def nrg2nrgx(arr_in):
    if isinstance(arr_in, np.ndarray):
        arr = np.zeros(len(arr_in), dtype=int)  # ensure copy
        if arr_in.ndim == 1:
            for i in range(arr.shape[0]):
                arr[i] = nrg2nrgx_lu[arr_in[i]] 
        else:  # must be 2D - no higher d's used
            for i in range(arr.shape[0]):
                arr[i, 0] = nrg2nrgx_lu[arr_in[i, 0]] 

    elif isinstance(arr_in, pd.Series) or isinstance(arr_in, pd.DataFrame):
        arr = np.zeros((len(arr_in.columns), len(arr_in)))
        for nrg in arr_in.columns:
            arr[nrg2nrgx_lu[nrg]] = arr_in[nrg]
    else:
        raise RuntimeError(f'Array type {type(arr_in)} unknown')   
    return arr

def nrgx2nrg(arr_in):
    arr = pd.Series(0, index=nrgs, dtype=float)
    for nrgx in range(len(arr_in)):
        arr[nrgx2nrg_lu[nrgx]] = arr_in[nrgx]
    return arr

# Save debug matrix
def save_matrix(file_name, matrix, file_path='./Python/Mailbox/Outbox/'):
    if (matrix.size > 1):
        full_path = file_path  + file_name + '.csv'
        if os.path.exists(full_path):
            os.remove(full_path)
        matrix.to_csv(full_path)
        print('csv saved to', full_path)
      

# Get price, CO2 generated, etc for each nrg
def get_specs_nrgs():
    specs_nrgs = pd.read_csv('./CSV/Specs.csv',
                         header    = 0, 
                         skiprows  = 1,
                         sep       = ',', 
                         index_col = 0)
            
    return specs_nrgs

# Get parameters set in overlord by user
def get_inbox():
    inbox = pd.read_csv('./Python/Mailbox/Inbox.csv', 
                         header    = 0,
                         sep       = ',',                           
                         index_col = 0)  
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
#       Then, it is adjusted each year as plants are retired, new plants are built, and demand changes.

#   Note that nrg sources 'Hydro', 'Oil' and 'Other' are not used in the optimization.
#      They are assumed to grow with demand.
#   Also note that gas and battery are figured as fill-in at the end of each test run.
def get_eia_data(region):
    global sample_years, sample_hours, first_year
    eia_filename = f'{region}_master.csv'
    csv_path = f'./csv/Eia_Hourly/Latest/Hourly_Capacity_values/{eia_filename}'
    eia_cap_csv = pd.read_csv(csv_path,
                         header=0, 
                         skiprows=0)


    # Sometimes, Solar has small negative values, which are not useful.
    for nrg in nrg_sources:
        eia_cap_csv[nrg] = eia_cap_csv[nrg].where(eia_cap_csv[nrg] > 0, 0)  # Remove negative values
    
    # Max MWh Values are assumed to be MW (MWh/h)
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
    first_year          = int(eia_cap_csv['date'][len(eia_cap_csv) - 1000][0:4])

    for nrg in nrg_sources:
        hourly_cap_pct_nrgs[nrg] = eia_cap_csv[nrg]
        # This .max() is to pick the maximum value for all years
        # Usually, this is the last year
        MW_nrgs[nrg] = eia_max_csv[nrg].max()
        
    MW_nrgs['Battery']  = 0
    return hourly_cap_pct_nrgs, MW_nrgs

def init_output_matrix():
    output_header_loc = output_header    
    for nrg in nrgs:
        output_header_loc = pd.concat([output_header_loc,pd.Series(nrg)], axis=0, ignore_index=True)
        for param in param_order:
            output_header_loc = pd.concat([output_header_loc,pd.Series([nrg + '_' + param])], axis=0, ignore_index=True)                 
    output_matrix = pd.DataFrame(columns=output_header_loc, dtype=float)
    return output_matrix

# Initial values for year zero - Spec Numbers
def init_tweaks(specs_nrgs,inbox):
    tweaked_nrgs = pd.DataFrame(1,columns=nrgs, index=tweaked_nrgs_order, dtype=float) 
    for nrg in nrgs: 
        tweaked_nrgs.at['Capital_Total_M$_MW',nrg]   = specs_nrgs.at['Capital_Total_M$_MW',nrg]
        tweaked_nrgs.at['Capital_M$_MW', nrg]    = \
                -4 * npf.pmt(inbox.at['Interest','Initial']/4,
                specs_nrgs.at['Lifetime', nrg]*4,
                specs_nrgs.at['Capital_Total_M$_MW', nrg])

        tweaked_nrgs.at['Fixed_M$_MW', nrg]      = specs_nrgs.at['Fixed_M$_MW', nrg]   
        tweaked_nrgs.at['Variable_M$_MWh', nrg]  = specs_nrgs.at['Variable_M$_MWh', nrg]
        tweaked_nrgs.at['Fixed_M$_MWh', nrg]     = specs_nrgs.at['Fixed_M$_MW', nrg]
        tweaked_nrgs.at['CO2_MT_MWh', nrg]       = specs_nrgs.at['CO2_MT_MWh', nrg]
                     
        tweaked_nrgs.at['Lifetime', nrg]         = specs_nrgs.at['Lifetime', nrg]
        tweaked_nrgs.at['Max_PCT', nrg]          = specs_nrgs.at['Max_PCT', nrg]
        
    tweaked_globals = pd.Series(0,index=tweaked_globals_order, dtype=float)
    tweaked_globals['CO2_M$_MWh'] = 0.
    tweaked_globals['Demand']    = 1
    tweaked_globals['Interest']  = 0.

    return tweaked_globals, tweaked_nrgs

# Figure next year's info
def fig_tweaks(    
                tweaked_nrgs,
                tweaked_globals,
                inbox,
                year):

    if year == 1:
        loc_ = 'Initial'
        tweaked_globals['CO2_M$_MT']   = inbox.at['CO2_Price', loc_]
        tweaked_globals['Demand']      = inbox.at['Demand', loc_] 
        tweaked_globals['Interest']    = inbox.at['Interest', loc_]
    else:
        loc_ = 'Yearly'
        if tweaked_globals['CO2_M$_MT'] < inbox.at['CO2_Price', loc_]:
            tweaked_globals['CO2_M$_MT']   += inbox.at['CO2_Price', 'Initial']
        tweaked_globals['Demand']           *= inbox.at['Demand', loc_] 
        tweaked_globals['Interest']         *= inbox.at['Interest', loc_]  
    
    for nrg in nrgs: 
        tweaked_nrgs.at['Capital_Total_M$_MW', nrg]   *= inbox.at[nrg + '_Capital', loc_]
        tweaked_nrgs.at['Fixed_M$_MWh', nrg]          *= inbox.at[nrg + '_Fixed', loc_]
        tweaked_nrgs.at['Variable_M$_MWh', nrg]       *= inbox.at[nrg + '_Variable', loc_]        
        tweaked_nrgs.at['Lifetime', nrg]              *= inbox.at[nrg + '_Lifetime', loc_]
        tweaked_nrgs.at['Max_PCT', nrg]               *= inbox.at[nrg + '_Max_PCT', loc_]
        
        # Note that this figures a quarterly payoff, 4 payments per year  
        tweaked_nrgs.at['Capital_M$_MW', nrg]    = \
                (-4 * npf.pmt(tweaked_globals.at['Interest']/4, 
                tweaked_nrgs.at['Lifetime', nrg]*4,
                tweaked_nrgs.at['Capital_Total_M$_MW', nrg]))        
                         
    return tweaked_globals, tweaked_nrgs

# Figure loss due to lifetime of plant
def fig_decadence(MW_nrgs, tweaked_nrgs):
    for nrg in nrgs:
        MW_nrgs[nrg]         *= 1 - (1/tweaked_nrgs.at['Lifetime', nrg])
    return MW_nrgs

# jit gives a 77x speedup.  Worth getting rid of all Pandas, print, etc.
@jit(nopython=True)  
def fig_hourly (
        hourly_MWh_needed,
        costly_nrgxs,
        hourly_MWh_avail_nrgxs,   
        battery_max,
        battery_stored,
        battery_efficiency,
        battery_MW):
    
    global debug_matrix

    battery_used   = 0.
    battery_empty  = battery_max - battery_stored
    outage_MWh     = 0.
    MWh_used_nrgxs = np.zeros((len(nrgs)), dtype=np.float32)
    # apply max MWh to entire grid. best case - hopefully no outage
    for nrgx in nrgx_sources:
        hourly_MWh_needed   -= hourly_MWh_avail_nrgxs[nrgx]

    # Now, lets see if the Battery can help
    # Charge the Battery with excess, then use it to fill in the need
    # Battery Variables:
    #   battery_avail:  battery to be used for need this hour.
    #   MWh_avail:      Excess to be used to charge battery this hour.
    #   battery_empty:  Empty part of battery
    #   battery_stored: Full part of battery
    #   battery_used:   Total battery output
    #   battery_efficiciency: % of complete charge/discharge cycle.
    #   battery_MW      Output capability.
    for hour in range(len(hourly_MWh_needed)):
        MWh_needed = hourly_MWh_needed[hour]
        if (MWh_needed < 0) and (battery_empty  > 0):
            # Charge battery with excess
            battery_avail      = min(battery_empty/battery_efficiency, -MWh_needed, battery_MW) \
                                        * battery_efficiency
            battery_empty     -= battery_avail 
            battery_stored    += battery_avail 
            MWh_needed        += battery_avail 

        elif (MWh_needed > 0) and (battery_stored > 0):
            # If there is a need, use battery to fill it
            MWh_avail        = min(battery_stored, MWh_needed, battery_MW)
            battery_used    += MWh_avail 
            battery_stored  -= MWh_avail 
            MWh_needed      -= MWh_avail
        
        hourly_MWh_needed[hour] = MWh_needed

    MWh_used_nrgxs[battery_nrgx] = battery_used
                       
    # Now, take care of excess    
    hourly_excess = -np.where(hourly_MWh_needed < 0, hourly_MWh_needed, 0)
    outage_MWh    = np.sum(hourly_MWh_needed[hourly_MWh_needed > 0])

    # Remove excess most expensive first
    # Note that battery is NOT included in costly_nrgxs  
    for nrgx in costly_nrgxs:
        hourly_excess          = hourly_excess - hourly_MWh_avail_nrgxs[nrgx]
        MWh_used_nrgxs[nrgx]   = -hourly_excess[hourly_excess < 0].sum()
        hourly_excess          = np.where(hourly_excess > 0, hourly_excess, 0)

    return MWh_used_nrgxs,    \
           outage_MWh,       \
           battery_stored

    
# add another year to the output matrix
# Note that MWh converted to GWh for output only
def add_output_year(           
                  MW_nrgs,
                  MWh_nrgs,
                  tweaked_globals,
                  tweaked_nrgs,
                  expensive,
                  outage_MWh,
                  output_matrix,
                  year,
                  start_knobs,
                  knobs_nrgs,
                  max_add_nrgs,
                  hourly_target_MWh,
                  iterations,
                  hourly_cap_pct_nrgs):

    output_matrix.at[year, 'Year']                 = year + first_year
    output_matrix.at[year, 'CO2_M$_MT']            = tweaked_globals['CO2_M$_MT'] 
    output_matrix.at[year, 'Target_MWh']           = hourly_target_MWh.sum() / sample_years
    output_matrix.at[year, 'Outage_MWh']           = outage_MWh / sample_years
    output_matrix.at[year, 'Outage_M$_MWh']        = expensive
    output_matrix.at[year, 'Iterations']           = iterations
    
    for nrg in nrgs:
        
        output_matrix.at[year, nrg + '_MW']             = MW_nrgs[nrg]
        output_matrix.at[year, nrg + '_MWh']            = MWh_nrgs[nrg] / sample_years
        output_matrix.at[year, nrg + '_Capital_M$']     = MW_nrgs[nrg]  * tweaked_nrgs.at['Capital_M$_MW', nrg] 
        output_matrix.at[year, nrg + '_Fixed_M$']       = MW_nrgs[nrg]  * tweaked_nrgs.at['Fixed_M$_MW', nrg]
        output_matrix.at[year, nrg + '_Variable_M$']    = MWh_nrgs[nrg] * tweaked_nrgs.at['Variable_M$_MWh', nrg]
        output_matrix.at[year, nrg + '_CO2_M$']         = MWh_nrgs[nrg] \
                                                            * tweaked_nrgs.at['CO2_MT_MWh', nrg] \
                                                            * tweaked_globals.at['CO2_M$_MT'] 

        output_matrix.at[year, nrg + '_Start_Knob']     = start_knobs[nrg]
        output_matrix.at[year, nrg + '_Optimized_Knob'] = knobs_nrgs[nrg]
        output_matrix.at[year, nrg + '_Max_Knob']       = max_add_nrgs[nrg]
        output_matrix.at[year, nrg + '_Decadence']      = 1 - (1/tweaked_nrgs.at['Lifetime', nrg])
        if (nrg != 'Battery'):
            output_matrix.at[year, nrg + '_Avg_Cap_Factor']  = hourly_cap_pct_nrgs[nrg].sum() / sample_hours
        else:
            output_matrix.at[year, nrg + '_Avg_Cap_Factor']  = 0

    return output_matrix

 # Save Output file.  Also called if minimizer error
def output_close(output_matrix, inbox, region):   
    file_name = f'{inbox.at["SubDir", "Text"]}-{region}'
    # minimized returned a really really small number for outage.  Excel couldn't handle it.
    # So rounding it to make that number 0.  Careful if you use really small numbers here.
    output_matrix_t = output_matrix.round(8).transpose()
    save_matrix(file_name, output_matrix_t)

# Cost function used by minimizer
def fig_cost(     
                  MW_nrgs, 
                  MWh_nrgs,
                  tweaked_globals,
                  tweaked_nrgs,  
                  expensive,     
                  outage_MWh):    
    cost = 0.

    for nrg in nrgs:
        cost += MW_nrgs[nrg]  * (tweaked_nrgs.at['Capital_M$_MW', nrg] + tweaked_nrgs.at['Capital_M$_MW', nrg])
        cost += MWh_nrgs[nrg] * tweaked_nrgs.at['Variable_M$_MWh', nrg]
        cost += MWh_nrgs[nrg] * tweaked_nrgs.at['CO2_MT_MWh', nrg] * tweaked_globals['CO2_M$_MT']
        
    cost += outage_MWh * expensive

    return cost
    
# This updates the data based on knob settings.
#   Solar, Wind, Nuclear and Coal have nrg total multiplied by knob
#   Gas and (if necessary) Battery is used to fill up to target value
#   Any excess is used to recharge Battery. 

# Note that all values must be passed as copies, so that
#   next run of optimize starts from the beginning

def update_data(
               knobs_nrgs,       
               hourly_cap_pct_nrgs,
               MW_nrgs,
               tweaked_nrgs,
               tweaked_globals,
               specs_nrgs,
               battery_stored,    
               hourly_target_MWh):    
    
    hourly_MWh_needed      = np.copy(hourly_target_MWh)
    MW_total               = MW_nrgs.sum()
    MWh_nrgs               = pd.Series(0, index=nrgs, dtype=float)
    hourly_MWh_avail_nrgs  = pd.DataFrame(0, index=hourly_cap_pct_nrgs.index, columns=nrgs, dtype=float) 
    for nrg in nrgs:
        MW_nrgs[nrg]               += knobs_nrgs[nrg] * MW_total
        hourly_MWh_avail_nrgs[nrg]  = MW_nrgs[nrg] * hourly_cap_pct_nrgs[nrg]

    MWh_cost_nrgs = pd.Series(0, index=nrg_sources, dtype=float)

    for nrg in nrg_sources:
        cost_per_MWh = tweaked_nrgs.at['Variable_M$_MWh', nrg]
        cost_per_MWh += tweaked_nrgs.at['CO2_MT_MWh', nrg] * tweaked_globals['CO2_M$_MT']
        MWh_cost_nrgs[nrg] = cost_per_MWh

    costly_nrgs            = MWh_cost_nrgs.sort_values(ascending=False).index
    costly_nrgs            = np.array(costly_nrgs)
    costly_nrgxs           = nrg2nrgx(costly_nrgs)
    hourly_MWh_avail_nrgxs = nrg2nrgx(hourly_MWh_avail_nrgs)

    global debug_count
    debug_count += 1

    MWh_used_nrgxs,   \
    outage_MWh,       \
    battery_stored    \
        = fig_hourly (
                hourly_MWh_needed      = np.copy(hourly_MWh_needed),
                costly_nrgxs           = costly_nrgxs,
                hourly_MWh_avail_nrgxs = hourly_MWh_avail_nrgxs,                   
                battery_max            = MW_nrgs['Battery'] * specs_nrgs.at['Hours', 'Battery'],
                battery_stored         = battery_stored,
                battery_efficiency     = specs_nrgs.at['Efficiency', 'Battery'],
                battery_MW             = MW_nrgs['Battery'])
          
    MWh_nrgs = nrgx2nrg(MWh_used_nrgxs)
        
    return \
           MW_nrgs,        \
           battery_stored, \
           outage_MWh,     \
           MWh_nrgs
          

# Main function used by minimizer              
def solve_this(
               knobs,               # Guess from minimizer                  
               hourly_cap_pct_nrgs, # Percentage of capacity used. Does not change      
               MW_nrgs,             # Total capacity from last year
               battery_stored,      # Battery stored from last year 
               hourly_target_MWh,   # Target hourly demand for this year
               tweaked_globals,     # Global tweaks
               tweaked_nrgs,        # Tweaks for each nrg
               specs_nrgs,          # Specifications
               expensive,           # Cost of outage      
               year):               # Current year
               
    global debug_matrix
    knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)

# Must make a separate copy of these.  Otherwise, Python modifies the original.
# We need it to stay the same for the next minimize run 
         
    new_MW_nrgs             = MW_nrgs.copy()                 
    new_battery_stored      = battery_stored

    new_MW_nrgs, \
    new_battery_stored, \
    outage_MWh, \
    MWh_nrgs = update_data(
        knobs_nrgs           = knobs_nrgs,
        hourly_cap_pct_nrgs  = hourly_cap_pct_nrgs,
        MW_nrgs              = new_MW_nrgs,
        tweaked_nrgs         = tweaked_nrgs,
        tweaked_globals      = tweaked_globals,
        specs_nrgs           = specs_nrgs,
        battery_stored       = new_battery_stored,
        hourly_target_MWh    = hourly_target_MWh)
                                 
    cost = fig_cost(
               MW_nrgs         = new_MW_nrgs,
               MWh_nrgs        = MWh_nrgs,
               tweaked_globals = tweaked_globals,
               tweaked_nrgs    = tweaked_nrgs,
               expensive       = expensive,
               outage_MWh      = outage_MWh)

    if (debug_enabled and debug_option == 'debug_step_minimizer'):
        debug_matrix, abort = debug.debug_step_minimizer( \
            debug_matrix, year, outage_MWh, knobs_nrgs, MWh_nrgs, cost)
        if abort:
            save_matrix(debug_filename, debug_matrix, './Analysis/')
            raise RuntimeError('Step Debug Done')
    return cost #This is a return from solve_this

# Initialize for year 1 starting place
def init_knobs(tweaked_globals, tweaked_nrgs):
    knobs_nrgs = pd.Series(0,index=nrgs, dtype=float)
    for nrg in nrgs:
        knobs_nrgs[nrg] = tweaked_globals['Demand'] + (1/tweaked_nrgs.at['Lifetime', nrg])
    return knobs_nrgs

def run_minimizer(    
                  hourly_cap_pct_nrgs,                 
                  MW_nrgs,
                  battery_stored,
                  hourly_target_MWh,
                  tweaked_globals,
                  tweaked_nrgs,
                  specs_nrgs,
                  expensive,               
                  knobs_nrgs,
                  inbox,
                  region,
                  output_matrix,
                  iterations,
                  year):
    
    global debug_matrix
    MW_total = MW_nrgs.sum() - MW_nrgs['Battery']

    max_add_nrgs = pd.Series(0,index=nrgs, dtype=float)
    # The logic is:
    #   Max_PCT is the max rate that MW increased year over year
    #      as a percentage of the total MW for that year.
    #      This gets around the problem of starting from zero for battery.
    #   To get this number, we need to allow it to rebuild from decadence
    #      plus Max_PCT more.
    for nrg in nrgs:
        pct_of_total      = MW_nrgs[nrg]/MW_total
        decayed           = 1/tweaked_nrgs.at['Lifetime',nrg]
        rebuild_pct       = decayed*pct_of_total            
        max_add_nrgs[nrg] = tweaked_nrgs.at['Max_PCT', nrg] + rebuild_pct

        
    # and retire some old plants
    MW_nrgs = fig_decadence(MW_nrgs, tweaked_nrgs)

    if (debug_enabled and debug_option == 'debug_one_case'):
        knobs_nrgs = one_case_nrgs.iloc[year - 1]

    else:
        hi_bound = max_add_nrgs.copy()
        # Can't unbuild - just let it decay
        # Note that the knobs now mean how much to build.
        lo_bound    = pd.Series(0,index=nrgs, dtype=float)
        bnds        = Bounds(lo_bound, hi_bound, True)
        start_knobs = max_add_nrgs
        knobs       = start_knobs.values
        method      = 'Nelder-Mead'
        fatol       = .0001
        xatol       = .00001
        rerun       = .01
        opt_done    = False
        last_result = 0.

        while(not(opt_done)):
            results =   minimize(
                        solve_this, 
                        knobs, 
                        args=(                 
                            hourly_cap_pct_nrgs,
                            MW_nrgs,            
                            battery_stored,
                            hourly_target_MWh,
                            tweaked_globals,
                            tweaked_nrgs,
                            specs_nrgs,
                            expensive,               
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
                    'iters': [results.nit],
                    'nfev': [results.nfev],
                    'success': [results.success],
                    'status': [results.status],
                    'message': [results.message]
                    } 
                error_matrix = pd.DataFrame(results_dict)
                save_matrix(f'Minimizer_Failure-{region}', error_matrix)
                if(debug_matrix.len() > 1):
                    save_matrix(debug_filename, debug_matrix, './Analysis/')
                raise RuntimeError('Minimizer Failure' )
            
            elif(debug_enabled and debug_option == 'debug_minimizer'):
                debug_matrix = debug.debug_minimizer_add1(debug_matrix, results, fatol, xatol, end_time, region)

            knobs      = results.x
            iterations += results.nit
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
                         
    return knobs_nrgs, max_add_nrgs, start_knobs, iterations


def init_data(hourly_cap_pct_nrgs, MW_nrgs):
    # Initialize data for the zeroth year

    MWh_nrgs          = pd.Series(0,index=nrgs, dtype=float)
    hourly_target_MWh = np.zeros(sample_hours, dtype=float)

    # Initialize based on EIA data
    for nrg in nrg_sources:
        MWh_nrgs[nrg]        = hourly_cap_pct_nrgs[nrg].sum() * MW_nrgs[nrg]
        hourly_target_MWh   += hourly_cap_pct_nrgs[nrg] * MW_nrgs[nrg]
            
    return MWh_nrgs, hourly_target_MWh

# main is a once-through operation, so we try to do as much calc here as possible
def do_region(region):

    start_time    = time.time()
    inbox         = get_inbox()
    years         = inbox.at['Years', 'Initial']
    specs_nrgs    = get_specs_nrgs()
 
    hourly_cap_pct_nrgs, MW_nrgs = get_eia_data(region)

# Initialize based on EIA data
    MWh_nrgs,          \
    hourly_target_MWh \
        = init_data(hourly_cap_pct_nrgs, MW_nrgs)

    output_matrix = init_output_matrix()

    # Figure cost of outage - 100 times the cost per hour of every nrg
    avg_cost_per_hour = 0.
    for nrg in nrgs:
        avg_cost_per_hour += MWh_nrgs[nrg] * specs_nrgs.at ['Variable_M$_MWh', nrg] / (365.25*24)
        avg_cost_per_hour += MW_nrgs[nrg]  * specs_nrgs.at['Fixed_M$_MW', nrg]     / (365.25*24)

    expensive = avg_cost_per_hour * 1
    
    battery_stored = 0.
    outage_MWh     = 0.

        
    tweaked_globals, tweaked_nrgs = init_tweaks(specs_nrgs, inbox)

#Output Year Zero
    knobs_nrgs  = pd.Series(0., index=nrgs, dtype=float)
    output_matrix = \
                add_output_year(                          
                    MW_nrgs             = MW_nrgs,
                    MWh_nrgs            = MWh_nrgs,
                    tweaked_globals     = tweaked_globals,
                    tweaked_nrgs        = tweaked_nrgs,
                    expensive           = expensive,
                    outage_MWh          = outage_MWh,
                    output_matrix       = output_matrix,
                    year                = 0,
                    start_knobs         = pd.Series(0, index=nrgs, dtype=float),
                    knobs_nrgs          = pd.Series(0, index=nrgs, dtype=float),
                    max_add_nrgs        = pd.Series(1, index=nrgs, dtype=float),  # Avoid div-by-zero in Excel
                    hourly_target_MWh   = hourly_target_MWh,
                    iterations          = 0,
                    hourly_cap_pct_nrgs = hourly_cap_pct_nrgs)
        
    knobs_nrgs = init_knobs(tweaked_globals=tweaked_globals, tweaked_nrgs=tweaked_nrgs)                
    if (years > 0):
        total_iterations = 0                                
        for year in range(1, int(years)+1):
            iterations = 0
            print(f'Year {year} in {region}')
# Update prices                       
            tweaked_globals, tweaked_nrgs = \
                fig_tweaks (
                        tweaked_globals = tweaked_globals,
                        tweaked_nrgs    = tweaked_nrgs,
                        inbox           = inbox,
                        year            = year)

            hourly_target_MWh = (hourly_target_MWh * tweaked_globals['Demand'])

# Now optimize this year         
            knobs_nrgs, max_add_nrgs, start_knobs, iterations = \
                run_minimizer( \
                                hourly_cap_pct_nrgs = hourly_cap_pct_nrgs,              
                                MW_nrgs             = MW_nrgs, 
                                battery_stored      = battery_stored, 
                                hourly_target_MWh   = hourly_target_MWh,
                                tweaked_globals     = tweaked_globals,
                                tweaked_nrgs        = tweaked_nrgs,
                                specs_nrgs          = specs_nrgs,
                                expensive           = expensive,               
                                knobs_nrgs          = knobs_nrgs,
                                inbox               = inbox,
                                region              = region,
                                output_matrix       = output_matrix,
                                iterations          = iterations,
                                year                = year)
# Re-run with final optimal numbers 
            MW_nrgs,        \
            battery_stored, \
            outage_MWh,     \
            MWh_nrgs        \
            = update_data( 
                        knobs_nrgs          = knobs_nrgs,       
                        hourly_cap_pct_nrgs = hourly_cap_pct_nrgs, 
                        MW_nrgs             = MW_nrgs,
                        tweaked_nrgs        = tweaked_nrgs,
                        tweaked_globals     = tweaked_globals,
                        specs_nrgs          = specs_nrgs,
                        battery_stored      = battery_stored,
                        hourly_target_MWh   = hourly_target_MWh)

# Output     results of this year             
            output_matrix = \
                add_output_year(         
                  MW_nrgs             = MW_nrgs,
                  MWh_nrgs            = MWh_nrgs,
                  tweaked_globals     = tweaked_globals,
                  tweaked_nrgs        = tweaked_nrgs,
                  expensive           = expensive,
                  outage_MWh          = outage_MWh,
                  output_matrix       = output_matrix,
                  year                = year,
                  start_knobs         = start_knobs,
                  knobs_nrgs          = knobs_nrgs,
                  max_add_nrgs        = max_add_nrgs,
                  hourly_target_MWh   = hourly_target_MWh,
                  iterations          = iterations,
                  hourly_cap_pct_nrgs = hourly_cap_pct_nrgs)
            
            total_iterations += iterations
    # End of years for loop
    output_close(output_matrix, inbox, region)
    if (debug_enabled and len(debug_matrix) > 2):
        save_matrix(debug_filename, debug_matrix, './Analysis/')
    print(f'{region} Total Time = {(time.time() - start_time)/60:.2f} minutes Total iterations = {total_iterations:,}')
    
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
                error_matrix = pd.concat([error_matrix, pd.Series ([error])])
                error_matrix = pd.concat([error_matrix, pd.Series ([traceback])])
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