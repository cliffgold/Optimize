import pandas as pd
import numpy as np
import warnings
import numpy_financial as npf
from scipy.optimize import minimize, Bounds
import os 
import time
import warnings
import multiprocessing as mp
import traceback
from numba import jit

warnings.filterwarnings('error',module=r'.*Optimize.*')

# Naming Rules:
# ending in _nrgs: Pandas Series vector in nrgs below
# ending in _nrgxs: Numpy vector in nrgxs below
# Starting in hourly_: vector/matrix including sample years of hourly data
# Starting in yr: Scaled by sample_years

#Energy is in MWh
#Power is in MW.
#Cost is in M (Million $ US)
#CO2 is MTonne (MT)

#Globals

dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname + '/..')

# had to add to deletechars, they got inserted at the beginning of the first genfromtext entry.
delete_chars   = " !#$%&'()*+, -./:;<=>?@[\\]^{|}~﻿ï»¿"

# energy (nrg) representations
nrgs          = np.array(['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal', 'Battery'])
nrg_sources   = np.array(['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal'])

# Energy type constants for numpy array access
Solarx   = 0
Windx    = 1
Nuclearx = 2
Gasx     = 3
Coalx    = 4
Batteryx = 5

nrg2nrgx_lu   = {'Solar' : Solarx, 'Wind' : Windx, 'Nuclear' : Nuclearx, 'Gas' : Gasx, 'Coal' : Coalx, 'Battery' : Batteryx}
nrgx2nrg_lu   = {Solarx : 'Solar', Windx : 'Wind', Nuclearx : 'Nuclear',  Gasx : 'Gas', Coalx : 'Coal', Batteryx : 'Battery'}

nrgxs        = np.array([Solarx, Windx, Nuclearx, Gasx, Coalx, Batteryx])
nrgx_sources = np.array([Solarx, Windx, Nuclearx, Gasx, Coalx])

# Specs CSV row constants for numpy array access
Capital_Total_M_MW = 0  #Total overnight cost of construction
Fixed_M_MW         = 1
Variable_M_MWh     = 2
CO2_MT_MWh         = 3
Lifetime           = 4
Max_PCT            = 5
Efficiency         = 6  # Round-trip for battery
Hours              = 7  # hours of battery at rated MW (usually 4)
# Only used in tweakxs
Capital_M_MW       = 8  #1 year cost of financing


specx2spec_lu = {
    Capital_Total_M_MW  : 'Capital_Total_M$_MW',           
    Fixed_M_MW          : 'Fixed_M$_MW',              
    Variable_M_MWh      : 'Variable_M$_MWh',      
	CO2_MT_MWh          : 'CO2_MT_MWh',        
	Lifetime            : 'Lifetime',          
	Max_PCT             : 'Max_PCT',           
	Efficiency          : 'Efficiency',        
	Hours               : 'Hours'}

spec2specx_lu = {
    'Capital_Total_M$_MW': Capital_Total_M_MW,   
    'Fixed_M$_MW'        : Fixed_M_MW        ,       
    'Variable_M$_MWh'    : Variable_M_MWh    ,   
	'CO2_MT_MWh'         : CO2_MT_MWh        , 
	'Lifetime'           : Lifetime          ,
	'Max_PCT'            : Max_PCT           , 
	'Efficiency'         : Efficiency        , 
	'Hours' 			 : Hours            }

specxs = np.array([Capital_Total_M_MW,       
                  Fixed_M_MW,         
                  Variable_M_MWh,     
                  CO2_MT_MWh,        
                  Lifetime,          
                  Max_PCT,           
                  Efficiency,        
                  Hours])

tweakxs = np.array([Capital_Total_M_MW,       
                  Fixed_M_MW,         
                  Variable_M_MWh,     
                  CO2_MT_MWh,        
                  Lifetime,          
                  Max_PCT,           
                  Efficiency,        
                  Hours,
                  Capital_M_MW])

# tweaked_globalxs array constants for numpy array access
CO2_M_MT   = 0
Demand     = 1
Interest   = 2

# Output Matrix Columns
# First group is total for all nrgs.
# Leave as pandas, not accessed very often 
output_header = pd.Series(['Year', 'CO2_M$_MT', 'Target_MWh', 'Outage_MWh', 
                           'Outage_M$_MWh', 'Iterations'])

param_order   = pd.Series(['MW', 'MWh', 'Capital_M$', 'Fixed_M$',
                           'Variable_M$', 'CO2_M$', 'CO2_MT',
                           'Start_Knob', 'Optimized_Knob', 'PCT_Max_Add',
                           'Decadence'])

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

# Save debug matrix
def save_matrix(file_name, matrix, file_path='./Python/Mailbox/Outbox/'):
    if (matrix.size > 1):
        full_path = file_path  + file_name + '.csv'
        if os.path.exists(full_path):
            os.remove(full_path)
        matrix.to_csv(full_path)
        print('csv saved to', full_path)
      
    
# Get price, CO2 generated, etc for each nrg
def get_specxs_nrgxs():
    # Read with pandas, the convert to numpy
    # Hopefully, this method minimizes errors.

    specs_pd = pd.read_csv('./CSV/Specs.csv',
                             header    = 0, 
                             skiprows  = 1,
                             sep       = ',', 
                             index_col = 0)
    
    specxs_nrgxs = np.zeros((specxs.shape[0],nrgxs.shape[0]),dtype=float)

    for nrgx in nrgxs:
        nrg = nrgx2nrg_lu[nrgx]
        for specx in specxs:
            spec = specx2spec_lu[specx]
            specxs_nrgxs[specx,nrgx] = specs_pd.at[spec,nrg]

    return specxs_nrgxs

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
#   1. hourly_cap_pct_nrgxs: Hourly capacity percentages used historicly for each nrg.  
#       These percentages do not change for each year run.
#   2. MW_nrgxs: Maximum MW for each nrg.  
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
    eia_max =  eia_max_csv.max(axis=0)

    # Set up the hourly capacity percentages and max MW for each nrg
    # Also total demand for each hour
    MWh_nrgxs           = np.zeros((eia_cap_csv.index.shape[0],nrgxs.shape[0]), dtype=float)
    MW_nrgxs            = np.zeros(nrgxs.shape[0], dtype=float)
    sample_years        = len(eia_cap_csv)/(365.25 * 24) 
    sample_hours        = len(eia_cap_csv)
    first_year          = int(eia_cap_csv['date'][len(eia_cap_csv) - 1000][0:4])
    hourly_cap_pct_nrgxs = np.zeros((eia_cap_csv.index.shape[0],nrgxs.shape[0]), dtype=float)
    for nrgx in nrgx_sources:
        hourly_cap_pct_nrgxs[:,nrgx] = eia_cap_csv[nrgx2nrg_lu[nrgx]]
        # This .max() is to pick the maximum value for all years
        # Usually, this is the last year
        MW_nrgxs[nrgx] = eia_max[nrgx2nrg_lu[nrgx]]
        
    MW_nrgxs[Batteryx]  = 0
    return hourly_cap_pct_nrgxs, MW_nrgxs

def init_output_matrix():
    output_header_loc = output_header    
    for nrg in nrgs:
        output_header_loc = pd.concat([output_header_loc,pd.Series(nrg)], axis=0, ignore_index=True)
        for param in param_order:
            output_header_loc = pd.concat([output_header_loc,pd.Series([nrg + '_' + param])], axis=0, ignore_index=True)                 
    output_matrix = pd.DataFrame(columns=output_header_loc, dtype=float)
    return output_matrix

# Initial values for year zero - Spec Numbers
def init_tweakxs(specxs_nrgxs,inbox):
    # Create tweaked_nrgxs as numpy array (8 + 1 rows x 6 energy types)
    tweaked_nrgxs = np.ones((tweakxs.shape[0], nrgs.shape[0]), dtype=float)
    
    for nrgx in nrgxs:  # For each energy type
        for specx in specxs:
            tweaked_nrgxs[specx, nrgx] = specxs_nrgxs[specx, nrgx]

    # Local Calc on all nrgxs
        tweaked_nrgxs[Capital_M_MW, nrgx] = \
            -4 * npf.pmt(inbox.at['Interest','Initial']/4,
            specxs_nrgxs[Lifetime, nrgx]*4,
            specxs_nrgxs[Capital_Total_M_MW, nrgx])

        
    tweaked_globalxs = np.zeros(3, dtype=float)
    tweaked_globalxs[CO2_M_MT] = 0.
    tweaked_globalxs[Demand] = 1
    tweaked_globalxs[Interest] = 0.

    return tweaked_globalxs, tweaked_nrgxs

# Figure next year's info
def fig_tweakxs(    
                tweaked_nrgxs,
                tweaked_globalxs,
                inbox,
                year):

    if year == 1:
        loc_ = 'Initial'
        tweaked_globalxs[CO2_M_MT] = inbox.at['CO2_Price', loc_]
        tweaked_globalxs[Demand] = inbox.at['Demand', loc_] 
        tweaked_globalxs[Interest] = inbox.at['Interest', loc_]
    else:
        loc_ = 'Yearly'
        if tweaked_globalxs[CO2_M_MT] < inbox.at['CO2_Price', loc_]:
            tweaked_globalxs[CO2_M_MT] += inbox.at['CO2_Price', 'Initial']
        tweaked_globalxs[Demand] *= inbox.at['Demand', loc_] 
        tweaked_globalxs[Interest] *= inbox.at['Interest', loc_]  
    
    for nrgx in nrgxs:  # For each energy type
        nrg = nrgx2nrg_lu[nrgx]  # Get string name for inbox lookup
        tweaked_nrgxs[Capital_Total_M_MW, nrgx] *= inbox.at[nrg + '_Capital', loc_]
        tweaked_nrgxs[Fixed_M_MW, nrgx]         *= inbox.at[nrg + '_Fixed', loc_]
        tweaked_nrgxs[Variable_M_MWh, nrgx]     *= inbox.at[nrg + '_Variable', loc_]        
        tweaked_nrgxs[Lifetime, nrgx]           *= inbox.at[nrg + '_Lifetime', loc_]
        tweaked_nrgxs[Max_PCT, nrgx]            *= inbox.at[nrg + '_Max_PCT', loc_]
        
        # Note that this figures a quarterly payoff, 4 payments per year  
        tweaked_nrgxs[Capital_M_MW, nrgx] = \
                (-4 * npf.pmt(tweaked_globalxs[Interest]/4, 
                tweaked_nrgxs[Lifetime, nrgx]*4,
                tweaked_nrgxs[Capital_Total_M_MW, nrgx]))        
                         
    return tweaked_globalxs, tweaked_nrgxs

# Figure loss due to lifetime of plant
def fig_decadence(MW_nrgxs, tweaked_nrgxs):
    for nrgx in nrgxs:
        MW_nrgxs[nrgx] *= 1 - (1/tweaked_nrgxs[Lifetime, nrgx])
    return MW_nrgxs

# jit gives a 77x speedup.  Worth getting rid of all Pandas, print, etc.
@jit(nopython=True)  
def fig_hourly (
        hourly_MWh_required,
        costly_nrgxs,
        hourly_MWh_avail_nrgxs,   
        battery_max,
        battery_stored,
        battery_efficiency,
        battery_MW,
        Batteryx):

    battery_used   = 0.
    battery_empty  = battery_max - battery_stored
    outage_MWh     = 0.
    MWh_used_nrgxs = np.zeros((nrgxs).shape[0], dtype=np.float32)
    # apply max MWh to entire grid. best case - hopefully no outage
    for nrgx in nrgx_sources:
        MWh_used_nrgxs[nrgx]   = hourly_MWh_avail_nrgxs[:,nrgx].sum()
        hourly_MWh_required   -= hourly_MWh_avail_nrgxs[:,nrgx]

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
    for hour in range(hourly_MWh_required.shape[0]):
        MWh_needed = hourly_MWh_required[hour]
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
        
        hourly_MWh_required[hour] = MWh_needed

    MWh_used_nrgxs[Batteryx] = battery_used
                       
    # Now, take care of excess    
    hourly_excess = -np.where(hourly_MWh_required < 0, hourly_MWh_required, 0)
    outage_MWh    = np.sum(hourly_MWh_required[hourly_MWh_required > 0])

    # Remove excess most expensive first
    # Note that battery is NOT included in costly_nrgxs  
    for nrgx in costly_nrgxs:
        hourly_excess          = hourly_excess - hourly_MWh_avail_nrgxs[:,nrgx]
        MWh_used_nrgxs[nrgx]   = -hourly_excess[hourly_excess < 0].sum()
        hourly_excess          = np.where(hourly_excess > 0, hourly_excess, 0)

    return MWh_used_nrgxs,    \
           outage_MWh,       \
           battery_stored

    
# add another year to the output matrix
# Note that MWh converted to GWh for output only
def add_output_year(           
                  MW_nrgxs,
                  MWh_nrgxs,
                  tweaked_globalxs,
                  tweaked_nrgxs,
                  expensive,
                  outage_MWh,
                  output_matrix,
                  year,
                  first_start_knobs,
                  knobs_nrgxs,
                  max_add_nrgxs,
                  hourly_target_MWh,
                  iterations,
                  hourly_cap_pct_nrgxs):

    output_matrix.at[year, 'Year']                 = year + first_year
    output_matrix.at[year, 'CO2_M$_MT']            = tweaked_globalxs[CO2_M_MT] 
    output_matrix.at[year, 'Target_MWh']           = hourly_target_MWh.sum() / sample_years
    output_matrix.at[year, 'Outage_MWh']           = outage_MWh / sample_years
    output_matrix.at[year, 'Outage_M$_MWh']        = expensive
    output_matrix.at[year, 'Iterations']           = iterations
    
    for nrg in nrgs:
        nrgx = nrg2nrgx_lu[nrg]
        output_matrix.at[year, nrg + '_MW']             = MW_nrgxs[nrgx]
        output_matrix.at[year, nrg + '_MWh']            = MWh_nrgxs[nrgx] / sample_years
        output_matrix.at[year, nrg + '_Capital_M$']     = MW_nrgxs[nrgx]  * tweaked_nrgxs[Capital_M_MW, nrgx] 
        output_matrix.at[year, nrg + '_Fixed_M$']       = MW_nrgxs[nrgx]  * tweaked_nrgxs[Fixed_M_MW, nrgx]
        output_matrix.at[year, nrg + '_Variable_M$']    = MWh_nrgxs[nrgx] * tweaked_nrgxs[Variable_M_MWh, nrgx]
        output_matrix.at[year, nrg + '_CO2_M$']         = MWh_nrgxs[nrgx] \
                                                            * tweaked_nrgxs[CO2_MT_MWh, nrgx] \
                                                            * tweaked_globalxs[CO2_M_MT] 
        output_matrix.at[year, nrg + '_CO2_MT']         = MWh_nrgxs[nrgx]  * tweaked_nrgxs[CO2_MT_MWh, nrgx] 
        output_matrix.at[year, nrg + '_Start_Knob']     = first_start_knobs[nrgx]
        output_matrix.at[year, nrg + '_Optimized_Knob'] = knobs_nrgxs[nrgx]
        output_matrix.at[year, nrg + '_PCT_Max_Add']    = knobs_nrgxs[nrgx] / max_add_nrgxs[nrgx]
       

    return output_matrix

 # Save Output file.  Also called if minimizer error
def output_close(output_matrix, inbox, region):   
    file_name = f'{inbox.at["SubDir", "Text"]}-{region}'
    # minimized returned a really really small number for outage.  Excel couldn't handle it.
    # So rounding it to make that number 0.  Careful if you use really small numbers here.
    output_matrix_t = output_matrix.round(8).transpose()
    save_matrix(file_name, output_matrix_t)


# Sort by most expensive energy first.  We will drop excess here first
@jit(nopython=True) 
def sort_costliest(costly_nrgxs,
                   cost_per_MWh):
    
    for end_nrgx in range(costly_nrgxs.shape[0]-1,0,-1):
        for current_nrgx in range(end_nrgx):
            if cost_per_MWh[costly_nrgxs[current_nrgx]] < cost_per_MWh[costly_nrgxs[current_nrgx+1]]:
                temp                           = costly_nrgxs[current_nrgx + 1]
                costly_nrgxs[current_nrgx + 1] = costly_nrgxs[current_nrgx]
                costly_nrgxs[current_nrgx]     = temp

    return costly_nrgxs


# Cost function used by minimizer
@jit(nopython=True) 
def fig_cost(     
                  MW_nrgxs, 
                  MWh_nrgxs,
                  tweaked_globalxs,
                  tweaked_nrgxs,  
                  expensive,     
                  outage_MWh):    
    cost = 0.

    for nrgx in nrgxs:
        cost += MW_nrgxs[nrgx]  * (tweaked_nrgxs[Capital_M_MW, nrgx] + tweaked_nrgxs[Capital_M_MW, nrgx])
        cost += MWh_nrgxs[nrgx] * tweaked_nrgxs[Variable_M_MWh, nrgx]
        cost += MWh_nrgxs[nrgx] * tweaked_nrgxs[CO2_MT_MWh, nrgx] * tweaked_globalxs[CO2_M_MT]
        
    cost += outage_MWh * expensive

    return cost
    
# This updates the data based on knob settings.
#   Solar, Wind, Nuclear and Coal have nrg total multiplied by knob
#   Gas and (if necessary) Battery is used to fill up to target value
#   Any excess is used to recharge Battery. 

# Note that all values must be passed as copies, so that
#   next run of optimize starts from the beginning
@jit(nopython=True) 
def update_data(
               knobs_nrgxs,       
               hourly_cap_pct_nrgxs,
               MW_nrgxs,
               tweaked_nrgxs,
               tweaked_globalxs,
               specxs_nrgxs,
               battery_stored,    
               hourly_target_MWh):    
    
    hourly_MWh_needed      = np.copy(hourly_target_MWh)
    MW_total               = MW_nrgxs.sum()
    MWh_nrgxs              = np.zeros(nrgxs.shape[0], dtype=float)
    hourly_MWh_avail_nrgxs = np.zeros((sample_hours,nrgxs.shape[0]), dtype=float) 
    for nrgx in nrgxs:
        MW_nrgxs[nrgx]              += knobs_nrgxs[nrgx] * MW_total
        hourly_MWh_avail_nrgxs[:,nrgx] = MW_nrgxs[nrgx] * hourly_cap_pct_nrgxs[:,nrgx]

    cost_per_MWh = np.zeros(nrgx_sources.shape[0], dtype=float)

    for nrgx in nrgx_sources:
        cost_per_MWh[nrgx] = tweaked_nrgxs[Variable_M_MWh, nrgx]
        cost_per_MWh[nrgx] += tweaked_nrgxs[CO2_MT_MWh, nrgx] * tweaked_globalxs[CO2_M_MT]

    costly_nrgxs = sort_costliest(costly_nrgxs = nrgx_sources.copy(),
                                  cost_per_MWh = cost_per_MWh)
    
    
    MWh_nrgxs,   \
    outage_MWh,       \
    battery_stored    \
        = fig_hourly (
                hourly_MWh_required    = np.copy(hourly_MWh_needed),
                costly_nrgxs           = costly_nrgxs,
                hourly_MWh_avail_nrgxs = hourly_MWh_avail_nrgxs,                   
                battery_max            = MW_nrgxs[Batteryx] * specxs_nrgxs[Hours, Batteryx],
                battery_stored         = battery_stored,
                battery_efficiency     = specxs_nrgxs[Efficiency, Batteryx],
                battery_MW             = MW_nrgxs[Batteryx],
                Batteryx               = Batteryx)
                 
    return \
           MW_nrgxs,        \
           battery_stored, \
           outage_MWh,     \
           MWh_nrgxs
          

# Main function used by minimizer              
def solve_this(
               knobs_nrgxs,          # Guess from minimizer                  
               hourly_cap_pct_nrgxs, # Percentage of capacity used. Does not change      
               MW_nrgxs,             # Total capacity from last year
               battery_stored,       # Battery stored from last year 
               hourly_target_MWh,    # Target hourly demand for this year
               tweaked_globalxs,     # Global tweakxs
               tweaked_nrgxs,        # tweakxs for each nrg
               specxs_nrgxs,          # Specifications
               expensive,           # Cost of outage      
               year):               # Current year
               
    global debug_matrix

# Must make a separate copy of these.  Otherwise, Python modifies the original.
# We need it to stay the same for the next minimize run 
         
    new_MW_nrgxs            = MW_nrgxs.copy()                 
    new_battery_stored      = battery_stored

    new_MW_nrgxs, \
    new_battery_stored, \
    outage_MWh, \
    MWh_nrgxs = update_data(
        knobs_nrgxs          = knobs_nrgxs,
        hourly_cap_pct_nrgxs = hourly_cap_pct_nrgxs,
        MW_nrgxs             = new_MW_nrgxs,
        tweaked_nrgxs        = tweaked_nrgxs,
        tweaked_globalxs     = tweaked_globalxs,
        specxs_nrgxs         = specxs_nrgxs,
        battery_stored       = new_battery_stored,
        hourly_target_MWh    = hourly_target_MWh)
                                 
    cost = fig_cost(
               MW_nrgxs        = new_MW_nrgxs,
               MWh_nrgxs        = MWh_nrgxs,
               tweaked_globalxs = tweaked_globalxs,
               tweaked_nrgxs    = tweaked_nrgxs,
               expensive       = expensive,
               outage_MWh      = outage_MWh)

    if (debug_enabled and debug_option == 'debug_step_minimizer'):
        debug_matrix, abort = debug.debug_step_minimizer( \
            debug_matrix, year, outage_MWh, knobs_nrgxs, MWh_nrgxs, cost)
        if abort:
            save_matrix(debug_filename, debug_matrix, './Analysis/')
            raise RuntimeError('Step Debug Done')
    return cost #This is a return from solve_this

# Initialize for year 1 starting place
def init_knobs(tweaked_globalxs, tweaked_nrgxs):
    knobs_nrgxs = np.zeros(nrgxs.shape[0], dtype=float)
    for nrgx in nrgxs:
        knobs_nrgxs[nrgx] = tweaked_globalxs[Demand] + (1/tweaked_nrgxs[Lifetime, nrgx])
    return knobs_nrgxs

def run_minimizer(    
                  hourly_cap_pct_nrgxs,                 
                  MW_nrgxs,
                  battery_stored,
                  hourly_target_MWh,
                  tweaked_globalxs,
                  tweaked_nrgxs,
                  specxs_nrgxs,
                  expensive,               
                  knobs_nrgxs,
                  inbox,
                  region,
                  output_matrix,
                  iterations,
                  year):
    
    global debug_matrix
    MW_total = MW_nrgxs.sum() - MW_nrgxs[Batteryx]

    max_add_nrgxs = np.zeros(nrgxs.shape[0], dtype=float)

    # The logic is:
    #   Max_PCT is the max rate that MW increased year over year
    #      as a percentage of the total MW for that year.
    #      This gets around the problem of starting from zero for battery.
    #   To get this number, we need to allow it to rebuild from decadence
    #      plus Max_PCT more.
    for nrgx in nrgxs:
        pct_of_total        = MW_nrgxs[nrgx]/MW_total
        decayed             = 1/tweaked_nrgxs[Lifetime,nrgx]
        rebuild_pct         = decayed*pct_of_total            
        max_add_nrgxs[nrgx] = tweaked_nrgxs[Max_PCT, nrgx] + rebuild_pct

        
    # and retire some old plants
    MW_nrgxs= fig_decadence(MW_nrgxs, tweaked_nrgxs)

    if (debug_enabled and debug_option == 'debug_one_case'):
        knobs_nrgxs = one_case_nrgs.iloc[year - 1]

    else:
        hi_bound = max_add_nrgxs
        # Can't unbuild - just let it decay
        # Note that the knobs now mean how much to build.
        lo_bound          = np.zeros(nrgxs.shape[0], dtype=float)
        bnds              = Bounds(lo_bound, hi_bound, True)
        start_knobs       = max_add_nrgxs.copy()
        method            = 'Nelder-Mead'
        fatol             = .0001
        xatol             = .00001
        rerun             = .01
        opt_done          = False
        last_result       = 0.
        first_start_knobs = start_knobs
        while(not(opt_done)):
            results =   minimize(
                        solve_this, 
                        start_knobs, 
                        args=(                 
                            hourly_cap_pct_nrgxs,
                            MW_nrgxs,            
                            battery_stored,
                            hourly_target_MWh,
                            tweaked_globalxs,
                            tweaked_nrgxs,
                            specxs_nrgxs,
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
                if(len(debug_matrix) > 1):
                    save_matrix(debug_filename, debug_matrix, './Analysis/')
                raise RuntimeError('Minimizer Failure' )
            
            elif(debug_enabled and debug_option == 'debug_minimizer'):
                debug_matrix = debug.debug_minimizer_add1(debug_matrix, results, fatol, xatol, end_time, region)


            iterations += results.nit  
            if ((last_result > (results.fun * (1-rerun))) and \
                (last_result < (results.fun * (1+rerun)))):
                knobs_nrgxs = np.array(results.x)
                opt_done = True
            else:
                start_knobs = np.array(results.x)                
                if(last_result > 0):
                     print(f'{region} - Extra try at minimizer')
                last_result = results.fun
                fatol       = fatol/10.
                xatol       = xatol/10.
                         
    return knobs_nrgxs, max_add_nrgxs, first_start_knobs, iterations


def init_data(hourly_cap_pct_nrgxs, MW_nrgxs):
    # Initialize data for the zeroth year

    MWh_nrgxs         = np.zeros(nrgxs.shape[0], dtype=float)
    hourly_target_MWh = np.zeros(sample_hours, dtype=float)

    # Initialize based on EIA data
    for nrgx in nrgx_sources:
        MWh_nrgxs[nrgx]        = hourly_cap_pct_nrgxs[:,nrgx].sum() * MW_nrgxs[nrgx]
        hourly_target_MWh     += hourly_cap_pct_nrgxs[:,nrgx] * MW_nrgxs[nrgx]
            
    return MWh_nrgxs, hourly_target_MWh

# main is a once-through operation, so we try to do as much calc here as possible
def do_region(region):

    start_time    = time.time()
    inbox         = get_inbox()
    years         = inbox.at['Years', 'Initial']
    specxs_nrgxs  = get_specxs_nrgxs()
 
    hourly_cap_pct_nrgxs, MW_nrgxs= get_eia_data(region)

# Initialize based on EIA data
    MWh_nrgxs,          \
    hourly_target_MWh \
        = init_data(hourly_cap_pct_nrgxs, MW_nrgxs)

    output_matrix = init_output_matrix()

    # Figure cost of outage - sum cost per hour of every nrg
    sum_cost_per_hour = 0.
    for nrgx in nrgxs:
        sum_cost_per_hour += MWh_nrgxs[nrgx] * specxs_nrgxs[Variable_M_MWh, nrgx] / (365.25*24)
        sum_cost_per_hour += MW_nrgxs[nrgx]  * specxs_nrgxs[Fixed_M_MW, nrgx]     / (365.25*24)

    expensive = sum_cost_per_hour
    
    battery_stored = 0.
    outage_MWh     = 0.

        
    tweaked_globalxs, tweaked_nrgxs = init_tweakxs(specxs_nrgxs, inbox)

#Output Year Zero
    knobs_nrgxs  = np.zeros(nrgxs.shape[0], dtype=float)
    output_matrix = \
        add_output_year( \
            MW_nrgxs            = MW_nrgxs,
            MWh_nrgxs           = MWh_nrgxs,
            tweaked_globalxs    = tweaked_globalxs,
            tweaked_nrgxs       = tweaked_nrgxs,
            expensive           = expensive,
            outage_MWh          = outage_MWh,
            output_matrix       = output_matrix,
            year                = 0,
            first_start_knobs   = np.zeros(nrgxs.shape[0], dtype=float),
            knobs_nrgxs         = np.zeros(nrgxs.shape[0], dtype=float),
            max_add_nrgxs       = np.ones(nrgxs.shape[0], dtype=float),  # Avoid div-by-zero in Excel
            hourly_target_MWh   = hourly_target_MWh,
            iterations          = 0,
            hourly_cap_pct_nrgxs = hourly_cap_pct_nrgxs)
        
    knobs_nrgxs = init_knobs(tweaked_globalxs=tweaked_globalxs, tweaked_nrgxs=tweaked_nrgxs)                
    if (years > 0):
        total_iterations = 0                                
        for year in range(1, int(years)+1):
            iterations = 0
            print(f'Year {year} in {region}')
# Update prices                       
            tweaked_globalxs, tweaked_nrgxs = \
                fig_tweakxs (
                        tweaked_globalxs = tweaked_globalxs,
                        tweaked_nrgxs    = tweaked_nrgxs,
                        inbox           = inbox,
                        year            = year)

            hourly_target_MWh = (hourly_target_MWh * tweaked_globalxs[Demand])

# Now optimize this year         
            knobs_nrgxs, max_add_nrgxs, first_start_knobs, iterations = \
                run_minimizer( \
                                hourly_cap_pct_nrgxs = hourly_cap_pct_nrgxs,              
                                MW_nrgxs            = MW_nrgxs, 
                                battery_stored      = battery_stored, 
                                hourly_target_MWh   = hourly_target_MWh,
                                tweaked_globalxs     = tweaked_globalxs,
                                tweaked_nrgxs        = tweaked_nrgxs,
                                specxs_nrgxs        = specxs_nrgxs,
                                expensive           = expensive,               
                                knobs_nrgxs          = knobs_nrgxs,
                                inbox               = inbox,
                                region              = region,
                                output_matrix       = output_matrix,
                                iterations          = iterations,
                                year                = year)
# Re-run with final optimal numbers 
            MW_nrgxs,        \
            battery_stored, \
            outage_MWh,     \
            MWh_nrgxs        \
            = update_data( 
                        knobs_nrgxs          = knobs_nrgxs,       
                        hourly_cap_pct_nrgxs = hourly_cap_pct_nrgxs, 
                        MW_nrgxs             = MW_nrgxs,
                        tweaked_nrgxs        = tweaked_nrgxs,
                        tweaked_globalxs     = tweaked_globalxs,
                        specxs_nrgxs         = specxs_nrgxs,
                        battery_stored       = battery_stored,
                        hourly_target_MWh    = hourly_target_MWh)

# Output     results of this year             
            output_matrix = \
                add_output_year(         
                  MW_nrgxs            = MW_nrgxs,
                  MWh_nrgxs           = MWh_nrgxs,
                  tweaked_globalxs    = tweaked_globalxs,
                  tweaked_nrgxs       = tweaked_nrgxs,
                  expensive           = expensive,
                  outage_MWh          = outage_MWh,
                  output_matrix       = output_matrix,
                  year                = year,
                  first_start_knobs   = first_start_knobs,
                  knobs_nrgxs         = knobs_nrgxs,
                  max_add_nrgxs       = max_add_nrgxs,
                  hourly_target_MWh   = hourly_target_MWh,
                  iterations          = iterations,
                  hourly_cap_pct_nrgxs = hourly_cap_pct_nrgxs)
            
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