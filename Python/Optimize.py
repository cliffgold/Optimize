import numpy as np
import pandas as pd
import numpy_financial as npf
from scipy.optimize import minimize, Bounds, differential_evolution
from math import log10, floor
import os 
import time
import warnings
import multiprocessing as mp
import traceback
from shared import df_int_to_float, nrgs 

warnings.filterwarnings('error',module=r'.*Optimize.*')

# Naming Rules:
# ending in _nrgs: vector in nrgs below
# Starting in hourly_: vector/matrix including sample years of hourly data
# Starting in yr: Scaled by sample_years

#Energy is in MWh
#Power is in MW.
#Cost is in M$
#CO2 is MTonne

#Globals

dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname + '/..')

# had to add to deletechars, they got inserted at the beginning of the first genfromtext entry.
delete_chars   = " !#$%&'()*+, -./:;<=>?@[\\]^{|}~﻿ï»¿"

# Output Matrix Columns
# First group is total nrgs.  Removed molten from earlier, so added in excess and curtailed, so old vs new line up mostly
output_header = pd.Series(['Year', 'CO2_Price', 'Outage', 'Total_MW', 'Total_MWh', 'Total_Target', 'MW_Cost', 'MWh_Cost', 'Outage_Cost','CO2_Cost', 'MW+MWh+Outage_Cost', 'Including_CO2_Cost','Demand','Excess MWh','Curtailed MWh'])
param_order   = pd.Series(['MW', 'gen_MWh', 'TBD', 'Cost', 'CO2_Cost', 'CO2_MTon', 'MW_Cost', 'MWh_Cost', 'Start_Knob', 'Knob', 'Max_Knob'])
tweaked_globals_order = pd.Series(['CO2_Price', 'Demand', 'Interest'])
tweaked_nrgs_order    = pd.Series(['Capital','Fixed', 'perMW', 'perMWh', 'Max_PCT', 'Lifetime', 'CO2_gen'])

#************ Debug Options ************
#Select from the following options to debug
# None                         Normal mode - No Debug output

# debug_one_case               Run one set of knobs/Year - no minimize()
# debug_step_minimizer         Save data on each run of solve_this
# debug_unexpected_change      Print out numbers that should not change in each year
# debug_final_hourly           Save every hour in fig_hourly final run

import debug

debug_option = "None"
debug_matrix, debug_filename, debug_enabled, one_case_nrgs = debug.setup(debug_option)

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
#       Then, it is adjusted each year as plants are retired, new plants are built, and demand changes.

#   Note that nrg sources "Hydro", "Oil" and "Other" are not used in the optimization.
#      They are assumed to grow with demand.
#   Also note that gas and battery are figured as fill-in at the end of each test run.
def get_eia_data(region):
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

    return hourly_cap_pct_nrgs, MW_nrgs, sample_years, sample_hours

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
    else:
        loc_ = 'Yearly'
        tweaked_globals['CO2_Price']   += inbox.at['CO2_Price', loc_]
        tweaked_globals['Demand']      *= inbox.at['Demand', loc_] 
        tweaked_globals['Interest']    *= inbox.at['Interest', loc_]  
    
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
    
# Battery fills any leftover need.  If not enough, outage (VERY expensive)
def fig_hourly (
        hourly_MWh_needed,    
        battery_max,
        battery_stored,
        gen_MWh_nrgs,
        after_optimize, 
        year):
    
    global debug_matrix
    
    battery_used = 0.
    outage_MWh   = 0.
    excess_MWh   = 0.
    hour         = 0.

    for hour_of_need in hourly_MWh_needed:
        hour = hour + 1
        #Already have too much NRG
        if(hour_of_need < 0):
            path              = 'Excess'
            # Can battery take all the remaining excess, with some left over?
            battery_charge = min(battery_max - battery_stored, -hour_of_need)
            battery_stored += battery_charge
            excess_MWh     += -hour_of_need - battery_charge

        # Enough battery to meet need
        elif (hour_of_need <= battery_stored):
            path             = 'Use_Battery'
            battery_stored  -= hour_of_need
            battery_used    += hour_of_need
            
        # Not enough to meet need
        else:
            path           = 'UhOh'
            outage_MWh    += hour_of_need - battery_stored      
            battery_used  += battery_stored
            battery_stored = 0.
            
        if(debug_option == 'debug_final_hourly' and after_optimize):
            row_debug_matrix = len(debug_matrix)
            
            debug_matrix.at[row_debug_matrix, 'Year']           = year
            debug_matrix.at[row_debug_matrix, 'Path']           = path
            debug_matrix.at[row_debug_matrix, 'Hour_of_Need']   = hour_of_need
            debug_matrix.at[row_debug_matrix, 'Battery_Max']    = battery_max
            debug_matrix.at[row_debug_matrix, 'Battery_Used']   = battery_used
            debug_matrix.at[row_debug_matrix, 'Battery_Stored'] = battery_stored
            debug_matrix.at[row_debug_matrix, 'Excess']         = excess_MWh


    gen_MWh_nrgs['Battery'] = battery_used
   
    return gen_MWh_nrgs,     \
           outage_MWh,       \
           battery_stored,   \
           excess_MWh

def fig_excess(year, gen_MWh_nrgs, excess_MWh):
    curtailed_MWh = 0
    total_curtailable = 0.

    for nrg in ['Solar', 'Wind', 'Gas']:
        total_curtailable += gen_MWh_nrgs[nrg]

    if(excess_MWh > 0) and (total_curtailable > 0):
        for nrg in ['Solar', 'Wind', 'Gas']:
            excess = excess_MWh * gen_MWh_nrgs[nrg] / total_curtailable
            # .5 is just a test number
            gen_MWh_nrgs[nrg]  -= excess * .5
            curtailed_MWh      += excess
            excess_MWh         -= excess

    # Note that by this time, excess_MWh should be zero.
    # 
    if (debug_enabled and debug_option == "debug_one_case"):
        debug_matrix = debug.debug_one_case_year(debug_matrix, year,
            excess_MWh, curtailed_MWh)
    return gen_MWh_nrgs, excess_MWh, curtailed_MWh
    
# add another year to the output matrix
def add_output_year(           
                  MW_nrgs,
                  gen_MWh_nrgs,
                  tweaked_globals,
                  tweaked_nrgs,
                  expensive,
                  outage_MWh,
                  output_matrix,
                  year,
                  start_knobs,
                  knobs_nrgs,
                  max_add_nrgs,
                  target_hourly,
                  sample_years,
                  excess_MWh,
                  curtailed_MWh
                  ):

    yr_outage_MWh    = outage_MWh    / sample_years
    yr_excess_MWh    = excess_MWh    / sample_years
    yr_curtailed_MWh = curtailed_MWh / sample_years 

    output_matrix.at[year, 'Year']          = year
    output_matrix.at[year, 'CO2_Price']     = tweaked_globals['CO2_Price']
    output_matrix.at[year, 'Outage']        = yr_outage_MWh
    output_matrix.at[year, 'Demand']        = tweaked_globals['Demand']
    output_matrix.at[year, 'Excess MWh']    = yr_excess_MWh
    output_matrix.at[year, 'Curtailed MWh'] = yr_curtailed_MWh
    MW_cost      = 0.
    yr_MWh_cost  = 0.
    yr_total_CO2 = 0.
    total_MW     = 0.
    yr_total_MWh = 0.

    yr_gen_MWh_nrgs    = pd.Series(0, index=nrgs, dtype=float)

    
    for nrg in nrgs:
        yr_gen_MWh_nrgs[nrg]    = gen_MWh_nrgs[nrg]    / sample_years       

        output_matrix.at[year, nrg + '_gen_MWh'] = \
                            yr_gen_MWh_nrgs[nrg]
        output_matrix.at[year, nrg + '_MWh_Cost']   = \
                            yr_gen_MWh_nrgs[nrg] * tweaked_nrgs.at['perMWh', nrg]
        output_matrix.at[year, nrg + '_Cost']       = \
                            (MW_nrgs[nrg]  * tweaked_nrgs.at['perMW', nrg]) \
                            + (yr_gen_MWh_nrgs[nrg] * tweaked_nrgs.at['perMWh', nrg])
        output_matrix.at[year, nrg + '_CO2_MTon']  = \
                            yr_gen_MWh_nrgs[nrg] * tweaked_nrgs.at['CO2_gen', nrg]
        output_matrix.at[year, nrg + '_CO2_Cost']  = \
                            yr_gen_MWh_nrgs[nrg] * tweaked_nrgs.at['CO2_gen', nrg] \
                            * tweaked_globals['CO2_Price']
        
    # these do not need to be scaled by sample_years
        output_matrix.at[year, nrg + '_MW']         = \
                            MW_nrgs[nrg]  
        output_matrix.at[year, nrg + '_MW_Cost']    =  \
                            MW_nrgs[nrg]  * tweaked_nrgs.at['perMW', nrg]
        output_matrix.at[year, nrg + '_Start_Knob'] = \
                            start_knobs[nrg]
        output_matrix.at[year, nrg + '_Knob']       = \
                            knobs_nrgs[nrg]
        output_matrix.at[year, nrg + '_Max_Knob']   = \
                            max_add_nrgs[nrg]    
    
# These are summed over the nrgs and scaled by sample_years as required
        MW_cost      += MW_nrgs[nrg]              * tweaked_nrgs.at['perMW', nrg]
        yr_MWh_cost  += yr_gen_MWh_nrgs[nrg]  * tweaked_nrgs.at['perMWh', nrg]
        yr_total_CO2 += yr_gen_MWh_nrgs[nrg]  * tweaked_nrgs.at['CO2_gen', nrg]
        # Storage is really not a producer, and its MW is really MWh of capacity
        if (nrg != 'Battery'):
            total_MW     += MW_nrgs[nrg]
            yr_total_MWh += yr_gen_MWh_nrgs[nrg]
# end of "for nrg in nrgs"

    output_matrix.at[year, 'MW_Cost']            = MW_cost
    output_matrix.at[year, 'MWh_Cost']           = yr_MWh_cost
    output_matrix.at[year, 'Outage_Cost']        = yr_outage_MWh * expensive
    output_matrix.at[year, 'CO2_Cost']           = yr_total_CO2  * tweaked_globals['CO2_Price'] 
    
    output_matrix.at[year, 'MW+MWh+Outage_Cost'] = output_matrix[['MW_Cost','MWh_Cost','Outage_Cost']].loc[year].sum()
    output_matrix.at[year, 'Including_CO2_Cost'] = output_matrix[['MW+MWh+Outage_Cost', 'CO2_Cost']].loc[year].sum()
    
    output_matrix.at[year, 'Total_MW']    = total_MW 
    output_matrix.at[year, 'Total_MWh']   = yr_total_MWh
    output_matrix.at[year, 'Total_Target']= target_hourly.sum() / sample_years

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
                  gen_MWh_nrgs,
                  tweaked_globals,
                  tweaked_nrgs,  
                  expensive,     
                  outage_MWh,    
                  adj_zeros):    
    cost = 0.
    total_MWh_nrgs = pd.Series(0, index=nrgs, dtype=float)
    for nrg in nrgs:
        total_MWh_nrgs[nrg] = gen_MWh_nrgs[nrg].sum()

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
               target_hourly, 
               zero_nrgs,
               after_optimize,
               year):    
    
    hourly_MWh_needed      = target_hourly.copy()
    MW_total               = MW_nrgs.sum()
    adj_zeros              = 0.
    hourly_gen_MWh_nrgs    = hourly_cap_pct_nrgs.copy() # * MW Below
    gen_MWh_nrgs           = pd.Series(0, index=nrgs, dtype=float)
    MW_avail_nrgs          = pd.Series(0, index=nrgs, dtype=float)

    for nrg in ['Solar','Wind','Nuclear','Gas', 'Coal']:
        if (zero_nrgs[nrg] == 0):
            MW_avail_nrgs[nrg] = MW_nrgs[nrg] * knobs_nrgs[nrg]
            hourly_gen_MWh_nrgs[nrg] *= MW_avail_nrgs[nrg]
        # Build more plants
            if (knobs_nrgs[nrg] > 1): 
                MW_nrgs[nrg] *= knobs_nrgs[nrg]

            gen_MWh_nrgs[nrg]  = cap_pct_nrgs[nrg] * MW_nrgs[nrg]
            hourly_MWh_needed -= hourly_gen_MWh_nrgs[nrg]

        else:
            adj_zeros += knobs_nrgs[nrg]

    if (knobs_nrgs['Battery'] > 1):      
        MW_nrgs['Battery'] += (tweaked_nrgs.at['Max_PCT', 'Battery'] * MW_total * (knobs_nrgs['Battery'] - 1))
        # New batteries come pre-charged
        battery_stored     += (tweaked_nrgs.at['Max_PCT', 'Battery'] * MW_total * (knobs_nrgs['Battery'] - 1))
        
    gen_MWh_nrgs,     \
    outage_MWh,       \
    battery_stored,   \
    excess_MWh        \
        =             \
        fig_hourly (
                hourly_MWh_needed   = hourly_MWh_needed,                   
                battery_max         = MW_nrgs['Battery'],
                battery_stored      = battery_stored,
                gen_MWh_nrgs        = gen_MWh_nrgs,
                after_optimize      = after_optimize,
                year                = year)  
          
    gen_MWh_nrgs, excess_MWh, curtailed_MWh = fig_excess(year, gen_MWh_nrgs, excess_MWh)
        
    return \
           MW_nrgs,        \
           battery_stored, \
           adj_zeros,      \
           outage_MWh,     \
           gen_MWh_nrgs,   \
           excess_MWh,     \
           curtailed_MWh
          

# Main function used by minimizer              
def solve_this(
               knobs,               # Guess from minimizer                  
               hourly_cap_pct_nrgs, # Percentage of capacity used. Does not change
               cap_pct_nrgs,        # sum of percentages per nrg          
               MW_nrgs,             # Total capacity from last year
               battery_stored,      # battery stored from last year 
               target_hourly,       # Target hourly demand for this year
               tweaked_globals,     # Global tweaks
               tweaked_nrgs,        # Tweaks for each nrg
               expensive,           # Cost of outage      
               zero_nrgs,           # Zeroed out nrgs
               year):               # Current year
               
    global debug_matrix
    knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)

# Must make a separate copy of these.  Otherwise, Python modifies the original.
# We need it to stay the same for the next minimize run 
         
    new_MW_nrgs             = MW_nrgs.copy()                 
    new_battery_stored      = battery_stored

    new_MW_nrgs,         \
    new_battery_stored,  \
    adj_zeros,           \
    outage_MWh,          \
    gen_MWh_nrgs,        \
    excess_MWh,          \
    curtailed_MWh        \
        = update_data(
                      knobs_nrgs          = knobs_nrgs,           
                      hourly_cap_pct_nrgs = hourly_cap_pct_nrgs,
                      cap_pct_nrgs        = cap_pct_nrgs,    
                      MW_nrgs             = new_MW_nrgs,
                      tweaked_globals     = tweaked_globals,
                      tweaked_nrgs        = tweaked_nrgs,
                      battery_stored      = new_battery_stored,
                      target_hourly       = target_hourly,
                      zero_nrgs           = zero_nrgs,
                      after_optimize      = False,
                      year                = year)
                                 
    cost = cost_function(
               MW_nrgs         = new_MW_nrgs,
               gen_MWh_nrgs    = gen_MWh_nrgs,
               tweaked_globals = tweaked_globals,
               tweaked_nrgs    = tweaked_nrgs,
               expensive       = expensive,
               outage_MWh      = outage_MWh,
               adj_zeros       = adj_zeros)

    if (debug_enabled and debug_option == 'debug_step_minimizer'):
        debug_matrix = debug.debug_step_minimizer( \
            debug_matrix, year, outage_MWh, gen_MWh_nrgs['Gas'], gen_MWh_nrgs['Coal'],
            cost, knobs_nrgs['Gas'], knobs_nrgs['Coal'])
        
    return cost #This is a return from solve_this

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
                  target_hourly,
                  tweaked_globals,
                  tweaked_nrgs,
                  expensive,               
                  zero_nrgs,
                  knobs_nrgs,
                  inbox,
                  region,
                  output_matrix,
                  nit_total,
                  year):
    
    global debug_matrix
    #This is total energy produced - Storage is excluded to prevent double-counting
    # Also note that MW_nrgs['*_Storage'] units are actually MWh of capacity.  Not even compatable.
    MW_total     = MW_nrgs.sum() - MW_nrgs['Battery']
    
    start_knobs  = pd.Series(1,index=nrgs, dtype=float)
    max_add_nrgs = pd.Series(1,index=nrgs, dtype=float)
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

    if (debug_enabled and debug_option == "debug_one_case"):
        knobs_nrgs = one_case_nrgs.iloc[year - 1]

    else:
        hi_bound = max_add_nrgs.copy()
        # Can't unbuild - just let it decay
        lo_bound = pd.Series(0,index=nrgs, dtype=float)
        bnds  = Bounds(lo_bound, hi_bound, True)
        rerun = .01
        tol   = 1e-3
        atol  = 1e-6
        opt_done = False
        last_result = 0.
        while(not(opt_done)):
            call_time = time.time()
            knobs = pd.Series(knobs_nrgs).values
            if(debug_enabled and debug_option == 'debug_minimizer'):
                debug_matrix = debug.debug_minimizer_add2(debug_matrix, knobs, max_add_nrgs, bnds)

            results =   differential_evolution(
                        func = solve_this, 
                        x0   = knobs, 
                        args=(                 
                            hourly_cap_pct_nrgs,
                            cap_pct_nrgs,         
                            MW_nrgs,            
                            battery_stored,
                            target_hourly,
                            tweaked_globals,
                            tweaked_nrgs,
                            expensive,               
                            zero_nrgs,
                            year
                           ),
                        bounds=bnds,                  
                        tol = tol,
                        atol = atol,
                        polish=True,
                        strategy='best1bin',
                        maxiter  = 10000,
                        disp    = False)
 
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
                if(debug_matrix.len() > 1):
                    save_matrix(debug_filename, debug_matrix, './Analysis/')
                raise RuntimeError('Minimizer Failure' )
            
            elif(debug_enabled and debug_option == 'debug_minimizer'):
                debug_matrix = debug.debug_minimizer_add1(debug_matrix, results, tol, atol, end_time, call_time, region)

            knobs      = results.x
            nit_total += results.nit
            knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)   
            if ((last_result > (results.fun * (1-rerun))) and \
                (last_result < (results.fun * (1+rerun)))):
                opt_done = True
            else:
                if(last_result > 0):
                     print(f'{region} - Extra try at minimizer')
                last_result = results.fun
                tol       = tol/10.
                atol       = atol/10.
                         
    return knobs_nrgs, max_add_nrgs, start_knobs, nit_total


def init_data(hourly_cap_pct_nrgs, MW_nrgs, sample_hours):
    # Initialize data for the zeroth year

    cap_pct_nrgs      = pd.Series(0,index=nrgs, dtype=float)
    gen_MWh_nrgs      = pd.Series(0,index=nrgs, dtype=float)
    hourly_target_MWh = pd.Series(np.zeros(sample_hours, dtype=float))
    zero_nrgs         = pd.Series(0,index=nrgs, dtype=float)

    # Initialize based on EIA data
    for nrg in nrgs:
        cap_pct_nrgs[nrg]    = hourly_cap_pct_nrgs[nrg].sum()
        gen_MWh_nrgs[nrg] = cap_pct_nrgs[nrg] * MW_nrgs[nrg]
        hourly_target_MWh   += hourly_cap_pct_nrgs[nrg] * MW_nrgs[nrg]
        if (gen_MWh_nrgs[nrg] == 0) & (nrg != 'Battery'):
            zero_nrgs[nrg] = 1
            
    return cap_pct_nrgs, gen_MWh_nrgs, hourly_target_MWh, zero_nrgs

# main is a once-through operation, so we try to do as much calc here as possible
def do_region(region):

    start_time    = time.time()
    inbox         = get_inbox()
    years         = inbox.at['Years', 'Initial']
    specs_nrgs    = get_specs_nrgs()
 
    hourly_cap_pct_nrgs, MW_nrgs, sample_years, sample_hours = get_eia_data(region)

# Initialize based on EIA data
    cap_pct_nrgs,      \
    gen_MWh_nrgs,   \
    hourly_target_MWh, \
    zero_nrgs         \
        = init_data(hourly_cap_pct_nrgs, MW_nrgs, sample_hours)

    output_matrix = init_output_matrix()

    # Figure cost of outage - 100 times the cost per hour of every nrg
    avg_cost_per_hour = 0.
    for nrg in nrgs:
        avg_cost_per_hour += gen_MWh_nrgs[nrg].sum() * specs_nrgs.at ['Variable', nrg] / (365.25*24)
        avg_cost_per_hour += MW_nrgs[nrg]               * specs_nrgs.at['Fixed', nrg]    / (365.25*24)

    expensive = avg_cost_per_hour * 100
    
    battery_stored = 0.
    outage_MWh     = 0.
    target_hourly  = hourly_target_MWh.copy()
        
    tweaked_globals, tweaked_nrgs = init_tweaks(specs_nrgs, inbox)

#Output Year Zero
    knobs_nrgs  = pd.Series(1., index=nrgs, dtype=float)
    output_matrix = \
                add_output_year(                          
                    MW_nrgs         = MW_nrgs,
                    gen_MWh_nrgs    = gen_MWh_nrgs,
                    tweaked_globals = tweaked_globals,
                    tweaked_nrgs    = tweaked_nrgs,
                    expensive       = expensive,
                    outage_MWh      = outage_MWh,
                    output_matrix   = output_matrix,
                    year            = 0,
                    start_knobs     = knobs_nrgs,
                    knobs_nrgs      = knobs_nrgs,
                    max_add_nrgs    = knobs_nrgs,
                    target_hourly   = target_hourly,
                    sample_years    = sample_years,
                    excess_MWh      = 0,
                    curtailed_MWh   = 0)
        
    knobs_nrgs = init_knobs(tweaked_globals=tweaked_globals, tweaked_nrgs=tweaked_nrgs)                
    if (years > 0):
        nit_total = 0                                
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
            knobs_nrgs, max_add_nrgs, start_knobs, nit_total = \
                run_minimizer( \
                                hourly_cap_pct_nrgs = hourly_cap_pct_nrgs,
                                cap_pct_nrgs        = cap_pct_nrgs,                  
                                MW_nrgs             = MW_nrgs, 
                                battery_stored      = battery_stored, 
                                target_hourly       = target_hourly,
                                tweaked_globals     = tweaked_globals,
                                tweaked_nrgs        = tweaked_nrgs,
                                expensive           = expensive,               
                                zero_nrgs           = zero_nrgs,
                                knobs_nrgs          = knobs_nrgs,
                                inbox               = inbox,
                                region              = region,
                                output_matrix       = output_matrix,
                                nit_total           = nit_total,
                                year                = year)

            after_optimize = True
# Update data based on optimized knobs 
            MW_nrgs,        \
            battery_stored, \
            adj_zeros,      \
            outage_MWh,     \
            gen_MWh_nrgs,   \
            excess_MWh,     \
            curtailed_MWh   \
                = update_data( 
                        knobs_nrgs          = knobs_nrgs,       
                        hourly_cap_pct_nrgs = hourly_cap_pct_nrgs,
                        cap_pct_nrgs        = cap_pct_nrgs,  
                        MW_nrgs             = MW_nrgs,
                        tweaked_globals     = tweaked_globals,
                        tweaked_nrgs        = tweaked_nrgs,
                        battery_stored      = battery_stored,
                        target_hourly       = target_hourly,
                        zero_nrgs           = zero_nrgs,
                        after_optimize      = after_optimize,
                        year                = year)

# Output     results of this year             
            output_matrix = \
                add_output_year(         
                  MW_nrgs         = MW_nrgs,
                  gen_MWh_nrgs    = gen_MWh_nrgs,
                  tweaked_globals = tweaked_globals,
                  tweaked_nrgs    = tweaked_nrgs,
                  expensive       = expensive,
                  outage_MWh      = outage_MWh,
                  output_matrix   = output_matrix,
                  year            = year,
                  start_knobs     = start_knobs,
                  knobs_nrgs      = knobs_nrgs,
                  max_add_nrgs    = max_add_nrgs,
                  target_hourly   = target_hourly,
                  sample_years    = sample_years,
                  excess_MWh      = excess_MWh, 
                  curtailed_MWh   = curtailed_MWh)

    # End of years for loop
    output_close(output_matrix, inbox, region)
    if (debug_enabled and len(debug_matrix) > 2):
        save_matrix(debug_filename, debug_matrix, './Analysis/')
    print(f'{region} Total Time = {(time.time() - start_time)/60:.2f} minutes Total iterations = {nit_total}')
    
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
# This is the main entry point for the optimization script.