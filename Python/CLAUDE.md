# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an energy optimization system that simulates power grid scenarios using Python. The main optimization engine (`Optimize.py`) performs complex calculations for different energy sources (Solar, Wind, Nuclear, Gas, Coal, Storage) across multiple US regions and years.

## Key Architecture

### Core Components

- **Optimize.py**: Main optimization engine with scipy-based minimization
- **debug.py**: Debug utilities for different debugging modes
- **shared.py**, **plot_optimize.py**: Supporting utilities
- **EIA_Downloader.py**, **Battery_Downloader.py**: Data acquisition scripts

### Data Flow Architecture

1. **Configuration**: System reads parameters from `Mailbox/Inbox.csv` 
2. **Energy Specs**: Reads energy source specifications from `csv/Specs.csv`
3. **Regional Data**: Loads EIA hourly capacity data from `csv/Eia_Hourly/Latest/` 
4. **Optimization**: Uses scipy.optimize.minimize with scipy-optimized @jit functions
5. **Output**: Saves results to `Mailbox/Outbox/` as CSV files

### Key Data Structures

- **nrgs**: Energy types array `['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal', 'Storage']`
- **hourly_cap_pct_nrgs**: Historical hourly capacity percentages per energy type
- **MW_nrgs**: Maximum MW capacity for each energy type  
- **tweaked_globals/tweaked_nrgs**: Economic parameters (CO2 price, interest, demand multipliers)

## Running the System

### Main Execution
```bash
python Optimize.py
```

The system automatically:
- Reads configuration from `Mailbox/Inbox.csv`
- Processes either single region or all US regions (based on config)
- Runs multi-year optimization scenarios
- Outputs results to `Mailbox/Outbox/`

### Required Dependencies
The code requires these Python packages (install with pip):
```
pandas numpy scipy numba numpy-financial multiprocessing traceback warnings os time math
```

### Debug Modes
Set `debug_option` in Optimize.py to one of:
- `'None'`: Normal operation
- `'debug_one_case'`: Single test case (requires Analysis/One_Case_Nrgs.xlsx)  
- `'debug_step_minimizer'`: Step-by-step optimization debugging
- `'debug_final_hourly'`: Detailed hourly analysis

## Configuration

### Input Parameters (Mailbox/Inbox.csv)
- **Region**: Target region code or 'US' for all regions
- **Years**: Number of years to simulate 
- **CO2_Price**: Carbon pricing (Initial/Yearly)
- **Interest**: Financial interest rates
- **Demand**: Demand growth multipliers
- **[Energy]_[Parameter]**: Per-energy-source economic parameters

### Energy Specifications (csv/Specs.csv)
- **Capital**: Capital costs per MW
- **Fixed**: Fixed O&M costs
- **Variable**: Variable O&M costs  
- **CO2_gen**: CO2 emissions per MWh
- **Lifetime**: Plant lifetime in years
- **Max_PCT**: Maximum percentage of total grid capacity

## Key Algorithms

### Optimization Process
1. **fig_decadence()**: Models plant retirement due to aging
2. **run_minimizer()**: Uses scipy.optimize.minimize with Nelder-Mead method
3. **solve_this()**: Cost function optimized by minimizer
4. **fig_hourly()**: JIT-optimized hourly power dispatch simulation
5. **update_data()**: Updates grid state based on optimization knobs

### Performance Optimizations
- **@jit decorator**: 77x speedup for hourly calculations
- **Multiprocessing**: Parallel region processing when Region='US'
- **kill_parallel flag**: Disables parallel processing for debugging

## File Organization

- **Python/**: Main Python code
- **csv/**: Input data (EIA regional data, specifications)
- **Mailbox/Inbox.csv**: Input configuration
- **Mailbox/Outbox/**: Output results
- **Analysis/**: Debug output and analysis files

## Working with the Code

When modifying the optimization logic, understand that:
- Energy units are MWh, power units are MW, costs are M$, CO2 is MTonne
- The system uses "knobs" (multipliers) to represent capacity additions
- Storage is handled specially as grid-scale battery systems
- Historical EIA data provides realistic capacity factor patterns

The codebase follows a functional programming style with minimal classes, focusing on data transformations through pandas DataFrames and numpy arrays.