# from ast import List
import os
import requests
import numpy as np
import pandas as pd
import datetime as dt
import math
import shutil as sh

# Global read-only constants
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname + '\\..')

# Start and end date and time of first and last entry
first_hour_dt = dt.datetime(2020, 1, 1) # by default set to 12AM (midnight)
last_hour_dt = dt.datetime(dt.date.today().year - 1, 1, 1) # not need to specify time zone to UTC, it is by default dt.datetime(2020, 1, 2).isoformat(timespec='hours')
total_num_records = int((last_hour_dt - first_hour_dt).total_seconds() / 3600) # No. of records = total no. of hours

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
        'length=5000'

def get_energy_df_from_api(
        region: str,
        api_energy_code: str,
        energy_source: str,
        password: str) -> pd.DataFrame:
    '''
    This function fetches the energy data from the EIA API

    Parameters
    ----------
    region : str
        USA region.
    api_energy_code : str
        Code used in the EIA API relative to energy source.
    energy_source : str
        Energy source.
    password : str
        Password to access API of EIA.

    Returns
    -------
    None.

    '''
    
    df = pd.DataFrame()
    while df.shape[0] < total_num_records:
        
        # Relevant for API request, it only spills out 5000 entries each call
        records_left = min(total_num_records - df.shape[0], 5000)
        
        # Construct the API URL
        url = URL_constructor(
            password,
            region,
            api_energy_code,
            first_hour_dt.isoformat(timespec='hours'),
            last_hour_dt.isoformat(timespec='hours'),
            df.shape[0]
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
        tmp_df = tmp_df.rename(columns={'period': 'Date', 'value': energy_source}).drop(columns=['value-units'])
        tmp_df = tmp_df[['Date', energy_source]]
            
        # Concatenate the DataFrame adding the new entries
        df = pd.concat([df, tmp_df], ignore_index=True)

    # Format casting from ISO 8601 to 'YYMMDDTHH'
    df['Date'] = pd.to_datetime(df['Date'], format="ISO8601")
    df['Date'] = df['Date'].dt.strftime("%Y%m%dT%H")
    
    # Type casting of megawatts: from str -> int64
    df[energy_source] = df[energy_source].astype('int64')
    
    # Update on downloaded dataset
    print(energy_source + ' -- finished download')
    
    return df

def clean_energy(master_df):
    """
    Cleans missing or invalid energy values in the dataset.
    Missing values are replaced with values from the same hour on the previous or next day (24-hour shift).
    
    Parameters:
    master_df (pd.DataFrame): DataFrame containing energy data, including a 'Date' column.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with missing values filled.
    """
    
    print("Cleaning Values\n")
    
    # Iterate over each column, skipping 'Date'
    for nrg in master_df.columns:
        if nrg != 'Date':  
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
        
        # Initialize empty pandas.DataFrame
        master_df = pd.DataFrame()
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
                energy_df = get_energy_df_from_api(
                    region,
                    api_energy_code,
                    energy_source,
                    password)
                
                # Merge master df with the data of the next energy source
                if master_df.empty:
                    master_df = energy_df
                else:
                    master_df = master_df.merge(energy_df, how='left', on='Date')
            else:
                print(region, energy_source, "-- zero filled")
                zeros_df = pd.DataFrame(0, index=np.arange(total_num_records+1), columns=['Date', energy_source])
                master_df = pd.concat([master_df, zeros_df[energy_source]], axis = 1)
                # master_df = master_df.merge(zeros_df, how='left', on='Date')
        
        # Clean up the DataFrame in case of missing data
        master_df = clean_energy(master_df)
        
        # Dump data to a CSV file
        master_df.to_csv(master_df_folder_path + master_df_file, index=False, sep=',')        
    
        file_list.append(master_df_folder_path + master_df_file)
    
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