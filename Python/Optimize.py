﻿from tkinter import messagebox
import numpy as np
import pandas as pd
import numpy_financial as npf
from scipy.optimize import minimize, Bounds
from math import log10, floor
import os 
import time
import ctypes
import warnings
import multiprocessing as mp
import traceback

warnings.filterwarnings('error',module=r'.*Optimize_V.*')

# Naming Rules:
# ending in _nrgs: vector in nrgs below
# ending in _hourly: vector/matrix including 4 years of hourly values

#Energy is in MWh
#Power is in MW.
#Cost is in M$
#CO2 is MTonne

#Globals

dirname, filename = os.path.split(os.path.abspath(__file__)) 
os.chdir(dirname + '\\..')

# had to add to deletechars, they got inserted at the beginning of the first genfromtext entry.
sample_years   = 4
sample_points  = int(sample_years * 365.25 * 24)
delete_chars   = " !#$%&'()*+, -./:;<=>?@[\\]^{|}~﻿ï»¿"

nrgs           = np.array(['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal', 'Storage'])

others         = np.array(['Hydro', 'Oil', 'Other'])

# Output Matrix Columns
output_header = pd.Series(['Year', 'CO2_price', 'Outage', 'Total_MW', 'Total_MWh', 'Energy_Cost', 'CO2_Cost', 'Demand'])
param_order   = pd.Series(['MW','MWh','Cost', 'CO2_Cost', 'CO2_MTon', 'MW_Cost', 'MWh_Cost', 'Knob', 'Max_Knob', 'Lifetime'])


#************ Debug Options ************
# True = normal run - go through minimize()
normal_run = True

# True = do not run parallel processes
kill_parallel = False

# run one set of knobs/Year - no minimize()
#  Also set normal_run to False
debug_one_case = False

# True = print minimize results
debug_minimizer = False

# Print out on each step of the minimizer 
debug_step_minimizer = False

# Print out numbers that should not change in each year
debug_unexpected_change = False

# Printout cost info
debug_cost = False

# Printout cost info less verbose
debug_cost_sum = False


def msgbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def df_int_to_float(df):
    list = df.select_dtypes(include=['int']).columns.tolist()
    df[list] = df[list].astype(float) 
    return df

def round_significant(x,sig):
    x = round(x, -int(floor(log10(abs(x))))+sig-1)
    return x

# Get price, CO2 generated, etc for each nrg
def get_specs_nrgs():
    specs_nrgs = pd.read_csv('./csv/specs.csv',
                         header=0, 
                         skiprows=1, 
                         index_col=0)
   
    specs_nrgs = df_int_to_float(specs_nrgs)
          
    return specs_nrgs

# Get parameters set in overlord by user
def get_inbox():
    inbox = pd.read_csv('./Python/mailbox/inbox.csv', 
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
    
#Get hourly data
def get_eia_data(region):
    latest = open('./csv/Eia_Hourly/Latest.txt','r')
    csv_date = latest.read()
    latest.close()
    eia_filename = f'{region}_master_{csv_date}.csv'
    csv_path = f'./csv/Eia_Hourly/latest/{eia_filename}'

    hourly_nrgs = pd.read_csv(csv_path,
                         header=0, 
                         skiprows=0)
    storage = pd.DataFrame(0,index=hourly_nrgs.index, columns=['Storage'])
    others  = pd.DataFrame(0,index=hourly_nrgs.index, columns=['Others'])
    others['Others']  = hourly_nrgs['Hydro'] + hourly_nrgs['Oil'] + hourly_nrgs['Other']
    hourly_nrgs = pd.concat([hourly_nrgs,storage,others], axis=1)
    hourly_nrgs = df_int_to_float(hourly_nrgs)    
    return hourly_nrgs

def init_output_matrix():
    output_header_loc = output_header    
    for nrg in nrgs:
        for param in param_order:
            output_header_loc = pd.concat([output_header_loc,pd.Series([nrg + '_' + param])], axis=0, ignore_index=True)                 
    output_matrix = pd.DataFrame(columns=output_header_loc, dtype=float)
    return output_matrix

def init_prices_nrgs(specs_nrgs,inbox):
    prices_nrgs = pd.DataFrame(0,columns=nrgs, index=['Cap','Fixed', 'perMWh', 'perMW'], dtype=float) 
    for nrg in nrgs: 
        prices_nrgs[nrg].loc['Cap']     = specs_nrgs[nrg].loc['Capital']
        prices_nrgs[nrg].loc['Fixed']   = specs_nrgs[nrg].loc['Fixed']   
        prices_nrgs[nrg].loc['perMWh']  = specs_nrgs[nrg].loc['Variable']
        prices_nrgs[nrg].loc['perMW']   = specs_nrgs[nrg].loc['Fixed'] + \
                     (-4 * npf.pmt(inbox['Interest'].loc['Initial']/4, specs_nrgs[nrg].loc['Lifetime']*4,specs_nrgs[nrg].loc['Capital']))
    return prices_nrgs
# Figure any changes in pricing from overlord
def fig_next_prices(    
                specs_nrgs,
                prices_nrgs,
                CO2_price,
                demand,
                interest,
                inbox,
                year):

    if year == 1:
        
        CO2_price = inbox['CO2_Price'].loc['Initial']
        demand    = inbox['Demand'].loc['Initial'] 
        interest  = inbox['Interest'].loc['Initial'] 
        for nrg in nrgs: 
            prices_nrgs[nrg].loc['Cap']     = specs_nrgs[nrg].loc['Capital']  * inbox[nrg + '_Cap'].loc['Initial']
            prices_nrgs[nrg].loc['Fixed']   = specs_nrgs[nrg].loc['Fixed']    * inbox[nrg + '_Fixed'].loc['Initial']
            prices_nrgs[nrg].loc['perMWh']  = specs_nrgs[nrg].loc['Variable'] * inbox[nrg + '_Variable'].loc['Initial']
            prices_nrgs[nrg].loc['perMW']   = prices_nrgs[nrg].loc['Fixed'] + \
                             (-4 * npf.pmt(interest/4, specs_nrgs[nrg].loc['Lifetime']*4,prices_nrgs[nrg].loc['Cap']))
                             # Note that this figures a quarterly payoff, 4 payments per year  
    else: 
        CO2_price += inbox['CO2_Price'].loc['Yearly']
        for nrg in nrgs:  
            prices_nrgs[nrg].loc['Cap']     = prices_nrgs[nrg].loc['Cap']      * inbox[nrg + '_Cap'].loc['Yearly']
            prices_nrgs[nrg].loc['Fixed']   = prices_nrgs[nrg].loc['Fixed']    * inbox[nrg + '_Fixed'].loc['Yearly']
            prices_nrgs[nrg].loc['perMWh']  = prices_nrgs[nrg].loc['perMWh']   * inbox[nrg + '_Variable'].loc['Yearly']
            prices_nrgs[nrg].loc['perMW']   = prices_nrgs[nrg].loc['Fixed'] + \
                             (-4 * npf.pmt(interest/4, specs_nrgs[nrg].loc['Lifetime']*4, prices_nrgs[nrg].loc['Cap']))
            
    return prices_nrgs, CO2_price,  interest, demand

# Figure loss due to lifetime of plant
def fig_decadence(hourly_nrgs, MWh_nrgs, MW_nrgs, specs_nrgs):
    for nrg in nrgs:
        hourly_nrgs[nrg]   *= 1 - (1/specs_nrgs[nrg].loc['Lifetime'])
        MW_nrgs[nrg]       *= 1 - (1/specs_nrgs[nrg].loc['Lifetime'])
        MWh_nrgs[nrg]      *= 1 - (1/specs_nrgs[nrg].loc['Lifetime'])
    return hourly_nrgs, MWh_nrgs, MW_nrgs
    
# Gas fills any leftover need.  If not enough, storage.  If not enough, outage (VERY expensive)
def fig_gas_and_storage(needed_hourly, gas_max, storage_max, stored):
    gas_used = 0
    storage_used = 0
    outage_MWh = 0
    # High Compute Time
    for hour_of_need in needed_hourly:
        if(hour_of_need > 0):
            # if enough gas for everybody
            if (hour_of_need + (storage_max - stored) < gas_max):
                gas_used += hour_of_need + (storage_max - stored)
                stored    = storage_max
            # Enough gas to avoid outage + a little left over for stored
            elif (hour_of_need < gas_max):
                gas_used += gas_max
                stored   += gas_max - hour_of_need
            # Enough gas + storage to meet need
            elif (hour_of_need < gas_max + stored):
                gas_used     += gas_max
                stored       -= hour_of_need - gas_max
                storage_used += hour_of_need - gas_max
            # Not enough to meet need
            else:
                outage_MWh   += hour_of_need - gas_max - stored                    
                gas_used     += gas_max
                storage_used = stored
                stored       = 0
    return gas_used, storage_used, outage_MWh, stored

# add another year to the output matrix
def add_output_year(
                  MWh_nrgs,            
                  MW_nrgs,
                  specs_nrgs,
                  prices_nrgs,
                  expensive,
                  outage_MWh,
                  CO2_price,
                  output_matrix,
                  year,
                  knobs_nrgs,
                  max_add_nrgs,
                  demand):
            
    output_matrix.at[year, 'Year']      = year
    output_matrix.at[year, 'CO2_price'] = CO2_price
    output_matrix.at[year, 'Outage']    = outage_MWh
    output_matrix.at[year, 'Demand']    = demand
    
    energy_cost = 0
    total_CO2   = 0
    total_MW    = 0
    total_MWh   = 0
    
    for nrg in nrgs:
        output_matrix.at[year, nrg + '_MW']        = MW_nrgs[nrg]  
        output_matrix.at[year, nrg + '_MWh']       = MWh_nrgs[nrg]  
        output_matrix.at[year, nrg + '_MW_Cost']   = MW_nrgs[nrg]  * prices_nrgs[nrg].loc['perMW']
        output_matrix.at[year, nrg + '_MWh_Cost']  = MWh_nrgs[nrg] * prices_nrgs[nrg].loc['perMWh']
        output_matrix.at[year, nrg + '_Cost']      = MW_nrgs[nrg]  * prices_nrgs[nrg].loc['perMW']
        output_matrix.at[year, nrg + '_Cost']     += MWh_nrgs[nrg] * prices_nrgs[nrg].loc['perMWh']
        output_matrix.at[year, nrg + '_CO2_MTon']  = MWh_nrgs[nrg] * specs_nrgs[nrg].loc['CO2_gen']
        output_matrix.at[year, nrg + '_CO2_Cost']  = MWh_nrgs[nrg] * specs_nrgs[nrg].loc['CO2_gen'] * CO2_price
        output_matrix.at[year, nrg + '_Knob']      = knobs_nrgs[nrg]
        output_matrix.at[year, nrg + '_Max_Knob']  = max_add_nrgs[nrg]
        output_matrix.at[year, nrg + '_Lifetime']  = specs_nrgs[nrg].loc['Lifetime']
        

        energy_cost += MW_nrgs[nrg]  * prices_nrgs[nrg].loc['perMW']
        energy_cost += MWh_nrgs[nrg] * prices_nrgs[nrg].loc['perMWh']
        total_CO2   += MWh_nrgs[nrg] * specs_nrgs[nrg].loc['CO2_gen']
        # Storage is really not a producer, and its MW is really MWh of capacity
        if (nrg != 'Storage'):
            total_MW    += MW_nrgs[nrg]
            total_MWh   += MWh_nrgs[nrg]
    
    CO2_cost     = total_CO2 * CO2_price
    energy_cost += outage_MWh * expensive
    output_matrix.at[year, 'Energy_Cost'] = energy_cost
    output_matrix.at[year, 'CO2_Cost']    = CO2_cost 
    output_matrix.at[year, 'Total_MW']    = total_MW 
    output_matrix.at[year, 'Total_MWh']   = total_MWh
    return output_matrix
    
# Cost function used by minimizer
def cost_function(
                  MWh_nrgs,
                  specs_nrgs,
                  prices_nrgs,
                  MW_nrgs,
                  expensive,
                  outage_MWh,
                  CO2_price,
                  adj_zeros):
    cost = 0
    for nrg in nrgs:
        old_cost = cost
        cost += MW_nrgs[nrg]  * prices_nrgs[nrg].loc['perMW']
        cost += MWh_nrgs[nrg] * prices_nrgs[nrg].loc['perMWh']
        cost += MWh_nrgs[nrg] * specs_nrgs[nrg].loc['CO2_gen'] * CO2_price
        if debug_cost:
            print(f'{nrg}: ({MW_nrgs[nrg]:,.0f} MW * {prices_nrgs[nrg].loc["perMW"]:,.6f} M$/MW)')
            print(f'      + {MWh_nrgs[nrg]:,.0f} MWh * {prices_nrgs[nrg].loc["perMWh"]:,.6f} M$/MWh')
            print(f'      + {MWh_nrgs[nrg]:,.0f} MWh  * {specs_nrgs[nrg].loc["CO2_gen"]:,.5f} MTon/MWh * {CO2_price:,.6f} M$ per ton')
            print(f' sum = {cost - old_cost:,.0f}') 
        elif debug_cost_sum:
            print(f' {nrg},  {cost - old_cost:.0f}') 
    cost += outage_MWh * expensive
    cost += adj_zeros  * expensive
    if debug_cost:
        print(f'outage + zeros, {cost - old_cost:,.0f}')
        print(f'final: {cost:,.0f}')
    if debug_cost_sum:
        print(f'outage + zeros, {cost - old_cost:.0f}')
        print(f'final: {cost:.0f}')
         
    return cost
    
# This updates the data based on knob settings.
#   Solar, Wind, Nuclear and Coal have nrg total multiplied by knob
#   Gas and (if necessary) storage is used to fill up to target value
#   Any left is used to recharge storage

def update_data(
               knobs_nrgs,    
               hourly_nrgs,   
               MWh_nrgs,      
               MW_nrgs,
               specs_nrgs,
               stored,        
               target_hourly, 
               zero_nrgs):    
    
    needed_hourly = target_hourly.copy()
    MW_total      = MW_nrgs.sum()
    adj_zeros     = 0
    for nrg in ['Solar','Wind','Nuclear','Coal']:
        if (zero_nrgs[nrg] == 0):
            # High Compute Time
            hourly_nrgs[nrg] *= knobs_nrgs[nrg]
            MWh_nrgs[nrg]    *= knobs_nrgs[nrg]
            if (knobs_nrgs[nrg] > 1):
                MW_nrgs[nrg] *= knobs_nrgs[nrg]
            # High Compute Time
            needed_hourly    -= hourly_nrgs[nrg]
        else:
            adj_zeros += knobs_nrgs[nrg]
# Storage MW max_add = 2, so max_build is specs Max_PCT  That is why we divide by 2        
    MW_nrgs['Storage'] += (specs_nrgs['Storage'].loc['Max_PCT'] * MW_total * knobs_nrgs['Storage']) / 2
        
    if (knobs_nrgs['Gas'] > 1):
        MW_nrgs['Gas']     = MW_nrgs['Gas'] * knobs_nrgs['Gas']
    
    gas_used, storage_used, outage_MWh, stored = \
        fig_gas_and_storage(
                needed_hourly = needed_hourly,
                gas_max       = MW_nrgs['Gas'],
                storage_max   = MW_nrgs['Storage'],
                stored        = stored)

    MWh_nrgs['Gas']     = gas_used / sample_years
    MWh_nrgs['Storage'] = storage_used / sample_years
            
    return hourly_nrgs, \
           MWh_nrgs,    \
           MW_nrgs,     \
           stored,      \
           adj_zeros,   \
           outage_MWh

# Main function used by minimizer              
def solve_this(
               knobs,                   
               hourly_nrgs,  
               MWh_nrgs,     
               MW_nrgs,      
               stored,       
               target_hourly,
               specs_nrgs,
               prices_nrgs,
               expensive,    
               CO2_price,  
               zero_nrgs):   
               
    knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)

# Must make a separate copy of these.  Otherwise, Python modifies the original.
# We need it to stay the same for the next minimize run
    new_hourly_nrgs = hourly_nrgs.copy()            
    new_MWh_nrgs= MWh_nrgs.copy()             
    new_MW_nrgs = MW_nrgs.copy()                 
    new_stored      = stored 
    
    if (debug_unexpected_change):
        # These should start the same for all calls to solve_this in one year
        print(new_hourly_nrgs.at[1234,'Solar'],  \
              new_MWh_nrgs['Wind'],   \
              new_MW_nrgs['Nuclear'], \
              new_stored              )
    
    new_hourly_nrgs,  \
    new_MWh_nrgs,     \
    new_MW_nrgs,      \
    new_stored,       \
    adj_zeros,        \
    outage_MWh        \
        = update_data(
                      knobs_nrgs    = knobs_nrgs,
                      hourly_nrgs   = new_hourly_nrgs,
                      MWh_nrgs      = new_MWh_nrgs,    
                      MW_nrgs       = new_MW_nrgs,
                      specs_nrgs    = specs_nrgs,
                      stored        = new_stored,
                      target_hourly = target_hourly,
                      zero_nrgs     = zero_nrgs)
                                 
    cost = cost_function(
               MWh_nrgs    = new_MWh_nrgs,
               MW_nrgs     = new_MW_nrgs,
               specs_nrgs  = specs_nrgs,
               prices_nrgs = prices_nrgs,
               expensive   = expensive,
               outage_MWh  = outage_MWh,
               CO2_price   = CO2_price,
               adj_zeros   = adj_zeros)

    if (debug_step_minimizer):
        print(f'Knobs = {np.round(pd.Series(knobs_nrgs).values,4)} outage_MWh = {outage_MWh:,.0f} cost = {cost:,.0f}')

    return (cost)

# Initialize for year 1 starting place
def init_knobs(demand, specs_nrgs):
    knobs_nrgs = pd.Series(1,index=nrgs, dtype=float)
    for nrg in nrgs:
        knobs_nrgs[nrg] = demand + (1/specs_nrgs[nrg].loc['Lifetime'])
    return knobs_nrgs


def run_minimizer(    
                  hourly_nrgs,         
                  MWh_nrgs,          
                  MW_nrgs,               
                  stored,              
                  target_hourly,
                  specs_nrgs,
                  prices_nrgs,
                  expensive,               
                  CO2_price,
                  zero_nrgs,
                  demand,
                  knobs_nrgs,
                  inbox,
                  region):
    
    #This is total energy produced - Storage is excluded to prevent double-counting
    # Also note that MW_nrgs['Storage'] units are actually MWh of capacity.  Not even compatable.
    MW_total     = MW_nrgs.sum() - MW_nrgs['Storage']
    
    max_add_nrgs = pd.Series(1,index=nrgs, dtype=float)
    for nrg in nrgs:
        if nrg == 'Storage':
            # Nominal for Storage is always half of max.
            max_add_nrgs['Storage'] = 2
        elif MW_nrgs[nrg] == 0.: 
            max_add_nrgs[nrg] = 10.
        else:    
            max_add_nrgs[nrg] = demand + ((specs_nrgs[nrg].loc['Max_PCT']*MW_total)/MW_nrgs[nrg])
        knobs_nrgs[nrg] = min(knobs_nrgs[nrg], max_add_nrgs[nrg] - .00001)
    # and retire some old plants
    hourly_nrgs, MWh_nrgs, MW_nrgs = \
                fig_decadence(hourly_nrgs, MWh_nrgs, MW_nrgs, specs_nrgs)  
    
    hi_bound = max_add_nrgs.copy()
    lo_bound = np.array([0.]  * len(nrgs))
    bnds     = Bounds(lo_bound, hi_bound, True)
    method = 'Nelder-Mead'
    fatol = .0001
    xatol = .00001
    opt_done = False
    last_result = 0
    while(not(opt_done)):
        call_time = time.time()
        knobs = pd.Series(knobs_nrgs).values
        if(debug_minimizer):
            print(f'Start Knobs = {knobs}')

        results =   minimize(
                    solve_this, 
                    knobs, 
                    args=(               
                        hourly_nrgs,         
                        MWh_nrgs,          
                        MW_nrgs,               
                        stored,              
                        target_hourly,
                        specs_nrgs,
                        prices_nrgs,
                        expensive,               
                        CO2_price,
                        zero_nrgs
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
            print('***************** Minimizer Failed ********************')
            msgbox('Minimizer Failure', f'{results.message}, {inbox["Title"].loc["Initial"]}, {region}' , 1)
            print(results)
            raise RuntimeError('Minimizer Failure' )
        elif(debug_minimizer):
            print (f'fatol {fatol} xatol {xatol}')
            print(f'Knobs  {results.x}')
            print(f'Results {results.fun:,.3f} Time {end_time - call_time:,.2f} with {results.nfev} runs')
            
        knobs      = results.x
        knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)   
        if ((last_result > (results.fun * .99)) and \
            (last_result < (results.fun * 1.01))):
            opt_done = True
        else:
            if(last_result > 0):
                print('Extra try at minimizer')
            last_result = results.fun
            fatol       = round_significant(fatol/10.,4)
            xatol       = round_significant(xatol/10.,4)
                     
    return knobs_nrgs, max_add_nrgs

def one_case(year):
    # This array reverses decadence - so starting place is the same
    #  redemption = np.array([1.034483,	1.041666,	1.02564,	1.04167,	1.02564,	0.])
    #  Optimized Case
    # test_cases = np.array([[1.7906705,  1.15446494, 1.03271335, 1.12593594, 0.91245342, 0.23845428]])
    test_cases =   np.array([[1.7906705,  1.15446494, 1.03271335, 1.13593594, 0.90245342, 0.23845428]])
    knobs = test_cases[year-1]
    knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)
    print(knobs)
    return knobs_nrgs
    
# main is a once-through operation, so we try to do as much calc here as possible
def do_region(region):
    start_time    = time.time()
    inbox         = get_inbox()
    years         = inbox['Years'].loc['Initial']
    specs_nrgs    = get_specs_nrgs()
    hourly_nrgs   = get_eia_data(region) 
    # If there is old data there, remove it
    outbox_path = './python/mailbox/outbox'    
    file_path = f'{outbox_path}/{inbox["Title"].loc["Initial"]}-{region}.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
    
    demand       = inbox['Demand'].loc['Initial']
    MW_nrgs      = pd.Series(0,index=nrgs, dtype=float)
    MWh_nrgs     = pd.Series(0,index=nrgs, dtype=float)
    total_hourly = pd.Series(np.zeros(sample_points, dtype=float))
    zero_nrgs    = pd.Series(0,index=nrgs, dtype=float)
    for nrg in nrgs:
        MW_nrgs[nrg]    = hourly_nrgs[nrg].max()
        MWh_nrgs[nrg]   = hourly_nrgs[nrg].sum() / sample_years
        total_hourly   += hourly_nrgs[nrg]   
        if (MWh_nrgs[nrg] == 0) & (nrg != 'Storage'):
            zero_nrgs[nrg] = 1
    
    MW_total      = MW_nrgs.sum()                       
    output_matrix = init_output_matrix()
    
    average_cost  = (MWh_nrgs * specs_nrgs[nrg].loc['Variable']).sum()
    average_cost += (MW_nrgs  * specs_nrgs[nrg].loc['Fixed']).sum()
    expensive     = average_cost
    stored        = 0
    outage_MWh    = 0
    target_hourly = total_hourly.copy()
    CO2_price     = 0
    
    prices_nrgs = init_prices_nrgs(specs_nrgs, inbox)
    interest    = inbox['Interest'].loc['Initial']
#Output Year Zero
    knobs_nrgs = pd.Series(1., index=nrgs, dtype=float)
    output_matrix = \
                add_output_year(
                    MWh_nrgs      = MWh_nrgs,                  
                    MW_nrgs       = MW_nrgs,
                    specs_nrgs    = specs_nrgs,
                    prices_nrgs   = prices_nrgs,
                    expensive     = expensive,
                    outage_MWh    = outage_MWh,
                    CO2_price     = CO2_price,
                    output_matrix = output_matrix,
                    year          = 0,
                    knobs_nrgs    = knobs_nrgs,
                    max_add_nrgs  = knobs_nrgs,
                    demand        = demand)
        
    knobs_nrgs = init_knobs(demand=demand, specs_nrgs=specs_nrgs)    
    
    for year in range(1, int(years)+1):
        print(f'Year {year} in {region}')
        # Figure new target (update_data used by solve_this, so not included there)
        target_hourly = (target_hourly * demand) + (hourly_nrgs['Others'] * (demand-1))

# Update prices                       
        prices_nrgs, CO2_price, interest, demand = \
            fig_next_prices (
                    specs_nrgs  = specs_nrgs, 
                    prices_nrgs = prices_nrgs,
                    CO2_price   = CO2_price,  
                    demand      = demand,     
                    interest    = interest,   
                    inbox       = inbox,
                    year        = year)
        
# Now optimize this year            
        # Normal run
        if normal_run:
            knobs_nrgs, max_add_nrgs = run_minimizer( \
                                hourly_nrgs    = hourly_nrgs,         
                                MWh_nrgs       = MWh_nrgs,         
                                MW_nrgs        = MW_nrgs,               
                                stored         = stored,              
                                target_hourly  = target_hourly,
                                specs_nrgs     = specs_nrgs,
                                prices_nrgs    = prices_nrgs,
                                expensive      = expensive,               
                                CO2_price      = CO2_price,
                                zero_nrgs      = zero_nrgs,
                                demand         = demand,
                                knobs_nrgs     = knobs_nrgs,
                                inbox          = inbox,
                                region         = region)
                                
        elif (debug_one_case):
            knobs_nrgs   = one_case(year)
            max_add_nrgs = pd.Series(999.,index=nrgs, dtype=float)
             

# Update data based on optimized knobs            
        hourly_nrgs,    \
        MWh_nrgs,       \
        MW_nrgs,        \
        stored,         \
        adj_zeros,      \
        outage_MWh      \
            = update_data(
                    knobs_nrgs    = knobs_nrgs,
                    hourly_nrgs   = hourly_nrgs,
                    MWh_nrgs      = MWh_nrgs,    
                    MW_nrgs       = MW_nrgs,
                    specs_nrgs    = specs_nrgs,
                    stored        = stored,
                    target_hourly = target_hourly,
                    zero_nrgs     = zero_nrgs)

# Output results of this year             
        output_matrix = \
            add_output_year(
              MWh_nrgs        = MWh_nrgs, 
              MW_nrgs         = MW_nrgs,
              specs_nrgs      = specs_nrgs,
              prices_nrgs     = prices_nrgs,
              expensive       = expensive,
              outage_MWh      = outage_MWh,
              CO2_price       = CO2_price,
              output_matrix   = output_matrix,
              year            = year,
              knobs_nrgs      = knobs_nrgs,
              max_add_nrgs    = max_add_nrgs,
              demand          = demand)
    # End of years for loop    
    outbox_path = './python/mailbox/outbox'
    if not os.path.exists(outbox_path):
        os.makedirs(outbox_path)
        
    file_path = f'{outbox_path}/{inbox["Title"].loc["Initial"]}-{region}.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
    output_matrix_t = output_matrix.transpose()
    output_matrix_t.to_csv(file_path)
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
    region        = inbox['Region'].loc['Initial']
    print('Starting ', inbox['Title'].loc['Initial'])
    
        
    if (not kill_parallel) and (region == 'All'):
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
                error, traceback = region_process[region].exception
                print (region, inbox['Title'].loc['Initial'], 'Error')
                msgbox("Process Error", f"{region} {inbox['Title'].loc['Initial']} Error", 1)
                print(traceback)
            else:
                print(region, ' Done')
    # kill_parallel True or not 'All'
    elif region == 'All':
        regions = get_all_regions()
        for region in regions:
            do_region(region)
    else:
        do_region(region)
        
if __name__ == '__main__':
    main()



