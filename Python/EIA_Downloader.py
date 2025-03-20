# from ast import List
from concurrent.futures.process import EXTRA_QUEUED_CALLS
from hmac import new
import os
from tkinter import END
import requests
import numpy as np
import pandas as pd
import datetime as dt
import math
import shutil as sh
from pytz import utc

# Global read-only constants
dirname, filename = os.path.split(os.path.abspath(__file__)) 
os.chdir(dirname + '\\..')

first_hour = dt.datetime(2020, 1, 1, 0).strftime("%Y%m%dT%H")
last_hour_dt = dt.datetime(2023, 12, 31, 23, tzinfo=utc)
total_num_records = int(365.25*4*24)  # 4 years


def init_master_df():
    master_df = pd.DataFrame([first_hour],columns=['Date'])

    for n in range(1,total_num_records):
        next_date = dt.datetime.fromisoformat(master_df.at[n-1, 'Date']) + dt.timedelta(hours=1)
        master_df.at[n, 'Date'] = next_date.strftime("%Y%m%dT%H")
    return master_df
    
def get_energy_df_from_api(region, api_energy_code, energy_source, password):
    df = pd.DataFrame()
    first = True
    early_hours = 0
    while len(df) < total_num_records:
        records_left = min(total_num_records - len(df),5000)
        url = f'https://api.eia.gov/v2/seriesid/EBA.{region}-ALL.NG.{api_energy_code}.HL?' \
            + f'api_key={password}&offset={len(df)+early_hours}&length={records_left}'
        
        print(url)
        END
        response = requests.get(url)
        json_str = response.json()
        data = json_str['response']['data']
        temp_df = pd.json_normalize(data)
        temp_df = temp_df.rename(columns={'period':'Date', 'value':energy_source})
        temp_df = temp_df[['Date', energy_source]]
        if (first):
            first_eia_dt  = dt.datetime.fromisoformat(temp_df.at[0, 'Date'])
            early_time    = first_eia_dt - last_hour_dt
            early_hours   = early_time.days * 24 + int((early_time.seconds + 1)/3600)
            if early_hours > 0:
                temp_df.drop(labels=range(0,early_hours),axis=0,inplace=True)
                temp_df.reset_index(drop=True, inplace=True)
            first = False
        df = pd.concat([df, temp_df], ignore_index=True)
#        print(f"Length of {energy_source} DF = {len(df)}")
    for n in range(0,total_num_records):
        date_utc = dt.datetime.fromisoformat(df.at[n, 'Date']).astimezone(utc)
        df.at[n, 'Date'] = date_utc.strftime("%Y%m%dT%H")
    print(energy_source + ' -- finished download')
    return df

def clean_energy(master_df):
    print("Cleaning Values")
    nrgs = master_df.columns       
    for nrg in nrgs:
        if(nrg != 'Date'):
            filling = False
            for n in range(0, len(master_df)):
# Fill in blanks with yesterday's or tomorrow's data
                if (not(isinstance(master_df.at[n, nrg], (float,int,np.int64)))) or math.isnan(master_df.at[n, nrg]):
                    if not filling:
                        print(f'Starting Fill at line {n} nrg {nrg}') 
                        filling = True
                    if (n < 25):
                        master_df.at[n, nrg] = master_df.at[n+24, nrg]
                    else:
                        master_df.at[n, nrg] = master_df.at[n-24, nrg]   
                else:
                    filling = False
                    
    return master_df
    

def main(region_dict: dict[str]):
    file_list = []
    os.chdir("./csv/Eia_Hourly")
    master_df_folder_path = 'Latest/'
    pass_file = open('../../../passwords/EIA.txt','r')
    password = pass_file.read()
    pass_file.close()
    if os.path.exists('Latest'):
        sh.rmtree('Latest')
    os.makedirs('Latest') 
    blank_master_df = init_master_df()
    for region in list(region_dict.keys()):
        master_df = blank_master_df.copy()
        print(f'Starting region {region}')
        master_df_file = f"{region}_master.csv"
        energy_source_dictionary = region_dict.get(region)
        for api_energy_code in energy_source_dictionary.keys():
            energy_source = energy_source_dictionary[api_energy_code]['Title']
            if energy_source_dictionary[api_energy_code]['In_EIA']:
                energy_df = get_energy_df_from_api(region, api_energy_code, energy_source, password)
                master_df = master_df.merge(energy_df, how='left', on='Date')
            else:
                print(region, energy_source, "-- zero filled")
                zeros_dict = {energy_source: [0] * total_num_records}
                zeros_df = pd.DataFrame(zeros_dict)
                master_df = pd.concat([master_df,zeros_df], axis = 1)
        
        master_df = clean_energy(master_df)
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