# from ast import List
import os
import requests
import numpy as np
import pandas as pd
import datetime as dt
import shutil as sh
from typing import List

# Global read-only constants
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname + '\\..')

# Length of API sub-query
length_query = 5000

def init_master_df(first_hour_dt, total_num_records):
    ''' Initialize the master dataframe with all the datetimes '''
    
    return pd.DataFrame({'date': [first_hour_dt + dt.timedelta(hours=i) for i in range(total_num_records)]})

def URL_constructor(
        password: str,
        region: str,
        api_energy_code: str,
        first_hour: str, # Format = YYYY-MM-DDTHH UTC
        last_hour: str, # Format = YYYY-MM-DDTHH UTC
        offset: int
        ) -> str:
    ''' Construct API URL to submit request '''
    
    return 'https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?' + \
        f'api_key={password}&frequency=hourly&data[0]=value&' + \
        f'facets[respondent][]={region}&' + \
        f'facets[fueltype][]={api_energy_code}&' + \
        f'start={first_hour}&' + \
        f'end={last_hour}&sort[0][column]=period&sort[0][direction]=asc&' + \
        f'offset={offset}&' + \
        f'length={length_query}'
        
def get_max_watthour(df, energy_source):
    ''' Find the maximum watthours for each year ''' 
    
    # Get the first energy date
    start_dt = df['date'].iloc[0]
    end_dt = df['date'].iloc[-1]
    
    # Difference in year between start and end datetime of EIA data
    no_years = int((end_dt - start_dt).total_seconds() / (365 * 24 * 3600))
    
    # Initialize list of max watthours
    max_watthour_lst = []
    
    # Temp variable to set date boundaries
    previous_year_dt = start_dt
    for i in range(no_years):
        next_year_dt = start_dt + dt.timedelta(years = i + 1)
        
        mask = (df['date'] >= previous_year_dt) & (df['date'] < next_year_dt)
        
        max_watthour_lst.append(df[energy_source][mask].max())
    
    return max_watthour_lst

def get_energy_df_from_api(
        first_hour_dt: dt.datetime,
        last_hour_dt: dt.datetime,
        total_num_records: int,
        region: str,
        api_energy_code: str,
        energy_source: str,
        password: str,
        datetimes: List[dt.datetime]) -> pd.DataFrame:
    '''
    This function fetches the energy data from the EIA API

    Parameters
    ----------
    first_hour_dt: dt.datetime
        First datetime.
    last_hour_dt: dt.datetime
        Last datetime.
    total_num_records : int
        Total number of entries.
    api_energy_code : str
        Code used in the EIA API relative to energy source.
    energy_source : str
        Energy source.
    password : str
        Password to access API of EIA.
    datetimes: List[dt.datetime]
        List of datetimes from first to last hour

    Returns
    -------
    None.

    '''
    
    # Time the process
    start = dt.datetime.today()
    
    df = pd.DataFrame({
        'date': datetimes,
        energy_source: np.zeros(len(datetimes), dtype=np.int64)
        })
    current = 0
    
    while current < total_num_records:
        
        # Construct the API URL
        url = URL_constructor(
            password,
            region,
            api_energy_code,
            first_hour_dt.isoformat(timespec='hours'),
            last_hour_dt.isoformat(timespec='hours'),
            current
            )
        
        # Get the JSON file from API URL
        response = requests.get(url)
        json_file = response.json()
        
        # Normalize JSON file to a specific record path in the JSON structure
        tmp_df = pd.json_normalize(
            json_file,
            record_path=['response', 'data']
            )
        
        # Rename columns and drop useless ones
        tmp_df = tmp_df.rename(columns={'period': 'date', 'value': energy_source})
        tmp_df = tmp_df[['date', energy_source]]
        
        # Type casting of megawatts: from str -> int64
        tmp_df[energy_source] = tmp_df[energy_source].astype('int64')
        
        # Fill megawatt entry to corresponding datetime element
        for dt_i, megaw_i in zip(tmp_df.date, tmp_df[energy_source]):
            mask = df.date == dt_i
            df.loc[mask, energy_source] = megaw_i
        
        # Update number of datetimes downloaded so far
        current += length_query
        
    # Store max watthours
    max_watthours = get_max_watthour(df, energy_source)
    
    # Time the process
    delta_time = (dt.datetime.today() - start).total_seconds() / 60
    
    # Update on downloaded dataset
    print(energy_source + f' -- finished download in {delta_time:.2f} minute(s)')
    
    return df, max_watthours

def clean_energy(master_df):
    """
    Cleans missing or invalid energy values in the dataset.
    Missing values are replaced with values from the same hour on the previous or next day (24-hour shift).
    
    Parameters:
    master_df (pd.DataFrame): DataFrame containing energy data, including a 'date' column.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with missing values filled.
    """
    
    print("Cleaning Values\n")
    
    # Iterate over each column, skipping 'date'
    for nrg in master_df.columns:
        if nrg != 'date':  
            filling = False  # Tracks whether we are in a filling sequence
            
            # Iterate over each row
            for n in range(master_df.shape[0]):
                value = master_df.at[n, nrg]
                
                # Check if the value is missing or not a valid number
                if pd.isna(value) or not isinstance(value, (int, float, np.integer)):
                    if not filling:
                        print(f'Starting Fill at row {n}, column: {nrg}')
                        filling = True
                    
                    # Ensure we don't go out of bounds
                    if n < 24:
                        if n + 24 < master_df.shape[0]:  # Check if forward fill is possible
                            master_df.at[n, nrg] = master_df.at[n + 24, nrg]
                        else:
                            master_df.at[n, nrg] = np.nan  # Assign NaN if no valid fill
                    else:
                        master_df.at[n, nrg] = master_df.at[n - 24, nrg]
                
                else:
                    filling = False  # Reset filling flag when encountering a valid value

    return master_df


def main(region_dict: dict[str]) -> None:
    
    # Start and end date and time of first and last entry
    first_hour_dt = dt.datetime(2020, 1, 1) # by default set to 12AM (midnight)
    last_hour_dt = dt.datetime(dt.date.today().year, 1, 1)
    
    # Total number of entries = hours
    total_num_records = int((last_hour_dt - first_hour_dt).total_seconds() / 3600) # No. of records = total no. of hours
    
    file_list = []
    os.chdir("./csv/Eia_Hourly")
    master_df_folder_path = 'Latest/'
    
    # Fetch EIA password
    with open('../../../passwords/EIA.txt','r') as f:
        password = f.read()
    
    # Check if Latest exists in cwd
    if os.path.exists('Latest'):
        # If it exists, delete all files and subdirectories in it
        sh.rmtree('Latest')
    
    # Make a new 'Latest' directory and if the parent directories are missing, create them as well
    os.makedirs('Latest') 
    
    # Iterate over US States
    for region in region_dict.keys():
        
        # Initialize pandas.DataFrame with all the dates from first hour to last one
        master_df = init_master_df(first_hour_dt, total_num_records)
        print(f'Starting region {region}\n')
        
        # File name of region-dependent master
        master_df_file = f"{region}_master.csv"
        
        # Get the dictionary corresponding to the State
        energy_source_dictionary = region_dict[region]
        
        # Iterate over the API code, e.g. SUN, NUC, etc.
        for api_energy_code in energy_source_dictionary.keys():
            
            # Get the name of the energy source
            energy_source = energy_source_dictionary[api_energy_code]['Title']
            
            # Check if the current state has the current energy source, or data are available
            if energy_source_dictionary[api_energy_code]['In_EIA']:
                
                # Get the megawatts produced hourly
                energy_df, max_watthour = get_energy_df_from_api(
                    first_hour_dt,
                    last_hour_dt,
                    total_num_records,
                    region,
                    api_energy_code,
                    energy_source,
                    password,
                    master_df.date)
                
                # Merge master df with the data of the next energy source
                if master_df.empty:
                    master_df = energy_df
                else:
                    master_df = pd.concat([master_df, energy_df[energy_source]], axis = 1)
            else:
                print(region, energy_source, "-- zero filled")
                zeros_df = pd.DataFrame(0, index=np.arange(total_num_records), columns=[energy_source])
                master_df = pd.concat([master_df, zeros_df], axis = 1)
        
        # Format casting from ISO 8601 to 'YYMMDDTHH'
        master_df['date'] = pd.to_datetime(master_df['date'], format="ISO8601")
        master_df['date'] = master_df['date'].dt.strftime("%Y%m%dT%H")
        
        # Clean up the DataFrame in case of missing data
        master_df_clean = clean_energy(master_df)
        
        # Normalize master df
        master_df_norm = pd.DataFrame([])
        for col in master_df_clean.columns:
            if col != 'date':
                max_val = master_df_clean[col].max()
                
                # If max_val is not null
                if max_val:
                    master_df_norm[col] = master_df_clean[col] / max_val
        
        # Dump data to a CSV file
        master_df_norm.to_csv(master_df_folder_path + master_df_file, index=False, sep=',')        
    
        file_list.append(master_df_folder_path + master_df_file)
    
# call the main method on each region with their associated list of available energy
region_dict = {

    # 'CAL':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },  
 
    # 'CAR':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':False},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },
 
    # 'CENT':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },
 
    # 'FLA':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':False},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },

    # 'MIDA':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },

    # 'MIDW':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':False}
    #     },
    
    # 'NE':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },
    
    # 'NW':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },
    
    # 'NY':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },
    
    # 'SE':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },
    
    # 'SW':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },
    
    # 'TEN':{
    #     'SUN':{'Title' :'Solar','In_EIA':True},
    #     'NUC':{'Title' :'Nuclear','In_EIA':True},
    #     'WND':{'Title' :'Wind','In_EIA':True},
    #     'COL':{'Title' :'Coal','In_EIA':True},
    #     'WAT':{'Title' :'Hydro','In_EIA':True},
    #     'NG': {'Title' :'Gas','In_EIA':True},
    #     'OTH':{'Title' :'Other','In_EIA':True},
    #     'OIL':{'Title' :'Oil','In_EIA':True}
    #     },
    
    'TEX':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':False}
        }
    }

main(region_dict=region_dict)

# eia_file_list = ['https://www.eia.gov/opendata/qb.php?category=3390106&sdid=EBA.FLA-ALL.NG.SUN.HL',
#                         'https://www.eia.gov/opendata/qb.php?category=3390106&sdid=EBA.FLA-ALL.NG.NUC.HL',
#                         'https://www.eia.gov/opendata/qb.php?category=3390106&sdid=EBA.FLA-ALL.NG.WND.HL',
#                         'https://www.eia.gov/opendata/qb.php?category=3390106&sdid=EBA.FLA-ALL.NG.COL.HL',
#                         'https://www.eia.gov/opendata/qb.php?category=3390106&sdid=EBA.FLA-ALL.NG.WAT.HL',
#                         'https://www.eia.gov/opendata/qb.php?category=3390106&sdid=EBA.FLA-ALL.NG.NG.HL',
#                         'https://www.eia.gov/opendata/qb.php?category=3390106&sdid=EBA.FLA-ALL.NG.OTH.HL',
#                         'https://www.eia.gov/opendata/qb.php?category=3390106&sdid=EBA.FLA-ALL.NG.OIL.HL']

# Changes for Optimize (saved in old/EIA_downloader_Rory.py)
    # Switched to create dates for master_df first, and just add to it.  Extra dates are ignored, skipped dates are filled with nan's
    # Clean at the end
    # No need for this to calculate sum column.
    # Not using dates anymore for version control.  Git is used for that.