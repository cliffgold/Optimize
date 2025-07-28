import os
import requests
import numpy as np
import pandas as pd
import datetime as dt
import shutil as sh
from os.path import join
from typing import List, Tuple
import stat

def URL_constructor(
        password: str,
        region: str,
        api_energy_code: str,
        first_hour_dt: str, # Format = YYYY-MM-DDTHH UTC
        last_hour_dt: str, # Format = YYYY-MM-DDTHH UTC
        offset: int
        ) -> str:
    ''' Construct API URL to submit request '''
    
    return 'https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?' + \
        f'api_key={password}&frequency=hourly&data[0]=value&' + \
        f'facets[respondent][]={region}&' + \
        f'facets[fueltype][]={api_energy_code}&' + \
        f'start={first_hour_dt}&' + \
        f'end={last_hour_dt}&sort[0][column]=period&sort[0][direction]=asc&' + \
        f'offset={offset}'

def _get_eia_password(password_path: str) -> str:
    with open(password_path, 'r') as f:
        return f.read()

def main():
    password        = _get_eia_password('../passwords/EIA.txt')
    region          = 'TEX'
    api_energy_code = 'OIL'
    first_hour_dt   = dt.datetime(2024, 1, 1)
    last_hour_dt    = dt.datetime(2024, 2, 1)
    offset          = 0
    url =  URL_constructor(
        password,
        region,
        api_energy_code,
        first_hour_dt.isoformat(timespec='hours'), 
        last_hour_dt.isoformat(timespec='hours'),
        offset)
    # Total number of entries = hours
    # Get the JSON file from API URL
    response = requests.get(url)
    json_file = response.json()
# Normalize JSON file to a specific record path in the JSON structure
    tmp_df = pd.json_normalize(
        json_file,
        record_path=['response', 'data']
        )
        
    print("Done")

main()