# from ast import List
from hmac import new
import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
from datetime import timedelta
import numbers
import shutil as sh

# Global read-only constants
today_date = date.today().isoformat()       # in case starting slightly before midnight.
total_num_records = int(365.25*4*24) # Always get most recent 4 years

def date2float(date):
    date_int = float(date[0:4] + date[5:7] + date[8:10])
    date_frac = float(date[11:13])/24   
    return date_int + date_frac

def get_energy_df_from_api(new_region, api_energy_code, energy_source, region):
    df = pd.DataFrame()
    pass_file = open('../../../passwords/EIA.txt','r')
    password = pass_file.read()
    pass_file.close()

    while len(df) < total_num_records:
        records_left = min(total_num_records - len(df),5000)
        url = f'https://api.eia.gov/v2/seriesid/EBA.{new_region}-ALL.NG.{api_energy_code}.HL?' \
                + f'api_key={password}&offset={len(df)}&length={records_left}'
        
        response = requests.get(url)
        json_str = response.json()
        data = json_str['response']['data']
        temp_df = pd.json_normalize(data)
        temp_df = temp_df.rename(columns={'period':'Date', 'value':f"{energy_source}"})
        temp_df = temp_df[['Date', f"{energy_source}"]]
      
        df = pd.concat([df, temp_df])
        print(f"Length of {energy_source} DF = {len(df)}")
    
    df = df.reset_index(drop=True)

    df.sort_values(by='Date', ascending = True, inplace = True)
    print(energy_source + ' -- finished download')
    return df

def clean_energy(energy_df):
    print('Cleaning Dates')
    nrg = energy_df.columns[1]
    n = 1
    while n < len(energy_df):
        this_time     = datetime.fromisoformat(energy_df['Date'].iat[n])
        last_time     = datetime.fromisoformat(energy_df['Date'].iat[n-1])
        should_time   = last_time + timedelta(hours = 1)
        if this_time != should_time:
            new_time  = should_time.strftime('%Y-%m-%dT%H%z')
            # Above format is same as EIA, but with 2 extra characters.  Make it the same.
            new_time  = new_time[0:len(new_time) - 2]
            insert_time = pd.DataFrame([[new_time, 'TBD']], columns=['Date', nrg])
            print(f'Inserting {new_time}')
            energy_df = pd.concat([energy_df.iloc[1:n,0:2],insert_time, energy_df.iloc[n:len(energy_df), 0:2]])
        else:
            n = n + 1

    print("Cleaning Values")
    for n in range(0, len(energy_df)):
# Fill in blanks with yesterday's or tomorrow's data
        if(not(isinstance(energy_df.loc[:,nrg].iat[n], numbers.Number))):
            if (n < 25):
                energy_df.loc[:,nrg].iat[n] = energy_df.loc[:,nrg].iat[n+24]
            else:
                energy_df.loc[:,nrg].iat[n] = energy_df.loc[:,nrg].iat[n-24]    
            print(f'filling {energy_df.loc[:,"Date"].iat[n]} with {energy_df.loc[:,nrg].iat[n-24]}')   
    return energy_df
    

def main(region_dict: dict[str]):
    file_list = []
    os.chdir(os.environ['Optimize_Dir'])
    os.chdir("./csv/Eia_Hourly")
    master_df_folder_path = f"{today_date}\\"
    if os.path.exists(master_df_folder_path) == True:
        sh.rmtree(master_df_folder_path)
        
    os.makedirs(master_df_folder_path)        
    for region in list(region_dict.keys()):
        print(f'Starting region {region}')
        master_df_file = f"{region}_master_{today_date}.csv"
        master_df = pd.DataFrame(columns=["Sum"]) # changed curly braces to square
        energy_source_dictionary = region_dict.get(region)
        first = True
        for api_energy_code in energy_source_dictionary.keys():
            energy_source = energy_source_dictionary[api_energy_code]['Title']
            if energy_source_dictionary[api_energy_code]['In_EIA']:
                energy_df = get_energy_df_from_api(region, api_energy_code, energy_source, region)
                energy_df = clean_energy(energy_df)
                if first:
                    master_df = pd.concat([energy_df[f'Date'], master_df, energy_df[f'{energy_source}']], axis=1)
                    first = False
                # master_df = pd.concat([master_df, energy_df], axis=1)
                else:
                    master_df = master_df.merge(energy_df, how='left', on='Date')
            else:
                print(region, energy_source, "-- zero filled")
                zeros_dict = {energy_source: [0] * total_num_records}
                zeros_df = pd.DataFrame(zeros_dict)
                master_df = pd.concat([master_df,zeros_df], axis = 1)
        
        # print(master_df.head(1))
        master_df = master_df.loc[:,~master_df.columns.duplicated()]
        cols = list(master_df.columns)
        # removing date from columns for sum calculation not removing from actual DF
        cols.remove('Date')
        master_df["Sum"] = master_df[cols].sum(axis=1)
        master_df["Date"] = master_df["Date"].apply(date2float)
        master_df.to_csv(master_df_folder_path + master_df_file, index=False, sep=',')        

        file_list.append(master_df_folder_path + master_df_file)
    latest = open("Latest.txt","w")
    latest.write(today_date)
    latest.close()
    
# call the main method on each region with their associated list of available energy
region_dict = {

    'CAL':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },  
 
    'CAR':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':False},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
 
    'CENT':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
 
    'FLA':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':False},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },

    'MIDA':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },

    'MIDW':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':False}
        },
    
    'NE':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'NW':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'NY':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'SE':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'SW':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'TEN':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
        'OTH':{'Title' :'Other','In_EIA':True},
        'OIL':{'Title' :'Oil','In_EIA':True}
        },
    
    'TEX':{
        'SUN':{'Title' :'Solar','In_EIA':True},
        'NUC':{'Title' :'Nuclear','In_EIA':True},
        'WND':{'Title' :'Wind','In_EIA':True},
        'COL':{'Title' :'Coal','In_EIA':True},
        'WAT':{'Title' :'Hydro','In_EIA':True},
        'NG':{'Title' :'Gas','In_EIA':True},
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

# Changes in V1 (saved in old/EIA_downloader_Rory.py)
#   Removed emailing (too hard for me to figure out)
#   Added missing nrg columns as 0's (missing columns cause bug in final results)
#   Removed writing csv for each individual energy type (not required)