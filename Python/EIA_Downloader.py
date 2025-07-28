# from ast import List
import os
import requests
import numpy as np
import pandas as pd
import datetime as dt
import shutil as sh
from os.path import join
from typing import List, Tuple
import stat

# Global read-only constants
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname + '\\..')

# Length of API sub-query
length_query = 5000

# Start and end date and time of first and last entry
first_hour_dt = dt.datetime(2020, 1, 1) # by default set to 12AM (midnight)
last_hour_dt = dt.datetime(dt.date.today().year, 1, 1)
# SHORTIE last_hour_dt = dt.datetime(2022, 1, 1)

def init_master_df(first_hour_dt, total_num_records):
    ''' Initialize the master dataframe with all the datetimes '''
    
    return pd.DataFrame({'date': [first_hour_dt + dt.timedelta(hours=i) for i in range(total_num_records)]})

def init_master_max_MWh(first_hour_dt, last_hour_dt):
    ''' Initialize the pandas.DataFrame of maximum megawatthours '''
    
    # Number of years of collected data
    no_years = int((last_hour_dt - first_hour_dt).total_seconds() / (365 * 24 * 3600))

    return pd.DataFrame(
        {
            'years': [int(first_hour_dt.year) + i for i in range(no_years)]
        }
    )

def normalize_per_year(
    df: pd.DataFrame,
    year: int,
    energy_source: str,
    max_val: int) -> pd.DataFrame:
    ''' Normalize the MWh data for each year to the maximum value (not inplace)'''
    
    # Type casting
    df[energy_source] = df[energy_source].astype('float64')
    
    # Define year boundaries
    lower_bound = dt.datetime(year, 1, 1)
    upper_bound = dt.datetime(year + 1, 1, 1)
    
    mask = (df['date'] >= lower_bound) & (df['date'] < upper_bound)
    df.loc[mask,energy_source] = df.loc[mask, energy_source] / max_val
    return df
    
def URL_constructor(
        password: str,
        region: str,
        api_energy_code: str,
        first_hour: str, # Format = YYYY-MM-DDTHH UTC
        last_hour: str, # Format = YYYY-MM-DDTHH UTC
        offset: int
        ) -> str:
    ''' Construct API URL to submit request '''
    
    return 's://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?' + \
        f'api_key={password}&frequency=hourly&data[0]=value&' + \
        f'facets[respondent][]={region}&' + \
        f'facets[fueltype][]={api_energy_code}&' + \
        f'start={first_hour}&' + \
        f'end={last_hour}&sort[0][column]=period&sort[0][direction]=asc&' + \
        f'offset={offset}&' + \
        f'length={length_query}'
        
def get_max_megawatthour(df, energy_source):
    ''' Return a DataFrame with the maximum megawatthours for each year '''

    # Number of years for which we collect the data
    no_years = int((last_hour_dt - first_hour_dt).total_seconds() / (365 * 24 * 3600))

    # List of maximum values of MWh for each year of a specific energy source
    max_megawatthour_lst = []

    # Temporary variable to set boundaries of each year
    current_year_dt = first_hour_dt
    for i in range(no_years):
        next_year_dt = dt.datetime(first_hour_dt.year + i + 1, 1, 1)

        # Get the maximum value
        mask = (df['date'] >= current_year_dt) & (df['date'] < next_year_dt)
        max_val = df[energy_source][mask].max()
        
        max_megawatthour_lst.append(max_val)  # Convert watthour to megawatthour

    return max_megawatthour_lst

def get_energy_df_from_api(
        first_hour_dt: dt.datetime,
        last_hour_dt: dt.datetime,
        total_num_records: int,
        region: str,
        api_energy_code: str,
        energy_source: str,
        password: str,
        datetimes: List[dt.datetime]) -> Tuple[pd.DataFrame, List]:
    '''
    This function fetches the energy data from the EIA API

    Parameters
    ----------
    first_hour_dt: dt.datetime
        First datetime.http
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
    max_watthours_lst = get_max_megawatthour(df, energy_source)
    
    # Time the process
    delta_time = (dt.datetime.today() - start).total_seconds() / 60
    
    # Update on downloaded dataset
    print(energy_source + f' -- finished download in {delta_time:.2f} minute(s)')
    
    return (df, max_watthours_lst)

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
        if nrg == 'date':
            continue
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
                if n < 24 and n + 24 < master_df.shape[0]:  # Check if forward fill is possible
                    master_df.at[n, nrg] = master_df.at[n + 24, nrg]
                elif n >= 24:
                    master_df.at[n, nrg] = master_df.at[n - 24, nrg]
                else:
                    master_df.at[n, nrg] = np.nan  # Assign NaN if no valid fill

            else:
                filling = False  # Reset filling flag when encountering a valid value

    return master_df


def main(region_dict: dict[str, dict[str, list[str]]]) -> None:
    
    # Total number of entries = hours
    total_num_records = int((last_hour_dt - first_hour_dt).total_seconds() / 3600) # No. of records = total no. of hours
    
    file_list_MWh = []
    file_list_max_MWh = []
    os.chdir("./csv/Eia_Hourly")
    master_df_folder_path = 'Latest/Hourly_Capacity_values'
    max_MWh_df_folder_path = 'Latest/max_MWh_values_yearly'
    
    # Fetch EIA password
    password = _get_eia_password('../../../passwords/EIA.txt')
    
    _reset_latest_folders(master_df_folder_path, max_MWh_df_folder_path)
    
    # Iterate over US States
    for region in region_dict:
        
        # Initialize pandas.DataFrame with all the dates from first hour to last one
        master_df = init_master_df(first_hour_dt, total_num_records)
        master_max_MWh_df = init_master_max_MWh(first_hour_dt, last_hour_dt)
        print(f'Starting region {region}\n')
        
        # File name of region-dependent master
        master_df_file = f"{region}_master.csv"
        master_max_MWh_file = f"{region}_max_vals.csv"
        
        # Get the dictionary corresponding to the State
        energy_source_dictionary = region_dict[region]
        
        # Iterate over the API code, e.g. SUN, NUC, etc.
        for api_energy_code in energy_source_dictionary.keys():
            
            # Get the name of the energy source
            energy_source = energy_source_dictionary[api_energy_code]['Title']
            
            # Check if the current state has the current energy source, or data are available
            if energy_source_dictionary[api_energy_code]['In_EIA']:
                
                # Get the megawatts produced hourly
                energy_df, max_MWh_df = get_energy_df_from_api(
                    first_hour_dt,
                    last_hour_dt,
                    total_num_records,
                    region,
                    api_energy_code,
                    energy_source,
                    password,
                    master_df.date.to_list())
                
                # Merge master df with the data of the next energy source
                if master_df.empty:
                    master_df = energy_df
                    master_max_MWh_df[energy_source] = np.zeros(master_max_MWh_df.shape[0])
                else:
                    master_df = pd.concat([master_df, energy_df[energy_source]], axis = 1)
                    master_max_MWh_df[energy_source] = max_MWh_df
            else:
                print(region, energy_source, "-- zero filled")
                zeros_df = pd.DataFrame(0, index=np.arange(total_num_records), columns=[energy_source])
                master_df = pd.concat([master_df, zeros_df], axis = 1)
                master_max_MWh_df[energy_source] = np.zeros(master_max_MWh_df.shape[0]) 

        
        # Clean up the DataFrame in case of missing data
        master_df_clean = clean_energy(master_df)
        
        # Normalize master df
        master_df_norm = _normalize_master_df_yearly(master_df_clean, master_max_MWh_df)
        
        # Format casting from datetime ISO 8601 to 'YYMMDDTHH'        
        master_df_norm['date'] = pd.to_datetime(master_df_norm['date'], format="ISO8601")
        master_df_norm['date'] = master_df_norm['date'].dt.strftime("%Y%m%dT%H")

        # Dump data to a CSV file
        master_df_norm.to_csv(join(master_df_folder_path, master_df_file), index=False, sep=',')
        master_max_MWh_df.to_csv(join(max_MWh_df_folder_path, master_max_MWh_file), index=False, sep=',')
    
        file_list_MWh.append(join(master_df_folder_path, master_df_file))
        file_list_max_MWh.append(join(max_MWh_df_folder_path, master_max_MWh_file))
    
def _get_eia_password(password_path: str) -> str:
    with open(password_path, 'r') as f:
        return f.read()

def force_remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"Error deleting {path}: {e}")

def _reset_latest_folders(master_df_folder_path: str, max_MWh_df_folder_path: str) -> None:
    if os.path.exists('Latest'):
        sh.rmtree('Latest', onerror=force_remove_readonly)
    os.makedirs(master_df_folder_path)
    os.makedirs(max_MWh_df_folder_path)
    
def _normalize_master_df_yearly(master_df_clean: pd.DataFrame, master_max_MWh_df: pd.DataFrame) -> pd.DataFrame:
    master_df_norm = master_df_clean.copy()
    for enrgy_src in master_df_clean.columns:
        if enrgy_src != 'date':
            for year, max_val in zip(master_max_MWh_df['years'], master_max_MWh_df[enrgy_src]):
                if max_val:
                    master_df_norm = normalize_per_year(master_df_norm, year, enrgy_src, max_val)
    return master_df_norm

# call the main method on each region with their associated list of available energy
region_dict = {

    'CAL':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },  
 
    'CAR':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':False},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
 
    'CENT':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
 
    'FLA':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':False},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },

    'MIDA':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },

    'MIDW':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':False}
        },
    
    'NE':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'NW':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'NY':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'SE':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'SW':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'TEN':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG': {'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
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