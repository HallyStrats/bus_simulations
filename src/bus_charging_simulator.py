import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm
from datetime import datetime, timedelta
import os

# --- New global variables will be handled via argparse for better flexibility ---

def parse_time_to_minutes(time_str):
    """Converts a HH:MM time string to minutes from midnight."""
    try:
        dt_obj = datetime.strptime(time_str, '%H:%M')
        return dt_obj.hour * 60 + dt_obj.minute
    except ValueError:
        raise ValueError(f"Incorrect time format for '{time_str}'. Please use HH:MM (24-hour format).")

def minutes_to_time_str(minutes, start_date):
    """
    MODIFIED: Converts minutes from simulation start to a YYYY-MM-DD HH:MM time string.
    """
    if pd.isna(minutes) or not np.isfinite(minutes):
        return ""
    # Convert the start_date (date object) to a datetime object to perform calculations
    start_datetime = datetime.combine(start_date, datetime.min.time())
    current_datetime = start_datetime + timedelta(minutes=int(minutes))
    return current_datetime.strftime('%Y-%m-%d %H:%M')

def simulate_unmanaged_charging(bus_arrivals_df, charge_profile, simulation_duration_minutes):
    """
    Simulates continuous charging from the moment a bus arrives.
    MODIFIED: Accepts simulation_duration_minutes as a parameter.
    """
    print("Starting UNMANAGED grid load simulation...")
    charge_profile.fillna(0, inplace=True)
    charge_profile['soc_rounded'] = charge_profile['soc'].round(0)
    soc_to_offset_map = charge_profile.reset_index().groupby('soc_rounded')['index'].first().to_dict()
    profile_socs = np.array(sorted(soc_to_offset_map.keys()))

    simulation_start_minute = 0
    simulation_end_minute = simulation_start_minute + simulation_duration_minutes
    
    time_index_minutes = range(simulation_start_minute, simulation_end_minute)
    grid_df = pd.DataFrame({'total_grid_load': 0.0, 'buses_on_charge': 0}, index=time_index_minutes)

    for _, bus in bus_arrivals_df.iterrows():
        print(f"Processing bus ID {bus['bus_id']} arriving at minute {int(round(bus['arrival_minute']))} with SOC {bus['soc_percent']:.2f}%")
        arrival_minute = int(round(bus['arrival_minute']))
        arrival_soc = bus['soc_percent']
        
        lookup_soc = arrival_soc
        closest_profile_soc = profile_socs[np.abs(profile_socs - lookup_soc).argmin()]
        start_offset = soc_to_offset_map.get(closest_profile_soc, 0)
        grid_load_values = charge_profile['grid_load'].iloc[start_offset:].values

        for minute_offset, load in enumerate(grid_load_values):
            current_minute = arrival_minute + minute_offset
            if current_minute in grid_df.index:
                grid_df.loc[current_minute, 'total_grid_load'] += load
                grid_df.loc[current_minute, 'buses_on_charge'] += 1
            elif current_minute >= simulation_end_minute:
                break
    return grid_df

def simulate_off_peak_charging(bus_arrivals_df, charge_profile, simulation_duration_minutes, num_buses):
    """
    MODIFIED: Simulates charging only during off-peak windows with three key changes:
    1. A cap on concurrent charging buses (90% of the fleet).
    2. The queue of waiting buses is CLEARED when an off-peak window ends.
    3. The energy deficit from cleared buses is calculated and reapplied to the next batch of buses by reducing their initial SOC.
    """
    print("Starting OFF-PEAK grid load simulation (Greedy Strategy)...")
    
    BATTERY_CAPACITY_KWH = 230.0 # Define battery capacity to convert energy to SOC.
    max_concurrent_charging = int(np.floor(0.6 * num_buses))
    print(f"Constraint: A maximum of {max_concurrent_charging} buses can charge at any moment.")

    def is_off_peak(minute):
        """Returns True if the minute is OUTSIDE of the defined peak windows."""
        time_of_day = minute % 1440
        is_morning_peak = (360 <= time_of_day < 540)  # 06:00 to 09:00
        is_evening_peak = (1020 <= time_of_day < 1320) # 17:00 to 22:00
        return not (is_morning_peak or is_evening_peak)

    charge_profile.fillna(0, inplace=True)
    charge_profile['soc_rounded'] = charge_profile['soc'].round(0)
    soc_to_offset_map = charge_profile.reset_index().groupby('soc_rounded')['index'].first().to_dict()
    profile_socs = np.array(sorted(soc_to_offset_map.keys()))
    
    simulation_start_minute = 0
    simulation_end_minute = simulation_start_minute + simulation_duration_minutes
    
    time_index_minutes = range(simulation_start_minute, simulation_end_minute)
    grid_df = pd.DataFrame({'total_grid_load': 0.0, 'buses_on_charge': 0}, index=time_index_minutes)
    
    active_buses = []
    unaccounted_energy_kWh = 0.0 # --- NEW: Track energy deficit across windows ---
    was_off_peak = is_off_peak(-1)

    for minute in time_index_minutes:
        now_is_off_peak = is_off_peak(minute)
        
        # --- NEW: Logic to handle transitions between windows ---
        
        # At the START of an off-peak window, distribute any previous energy deficit.
        if now_is_off_peak and not was_off_peak:
            if unaccounted_energy_kWh > 0 and active_buses:
                print(f"\nNew window started. Distributing {unaccounted_energy_kWh:.2f} kWh deficit among {len(active_buses)} new buses.")
                energy_share_per_bus = unaccounted_energy_kWh / len(active_buses)
                soc_reduction_percent = (energy_share_per_bus / BATTERY_CAPACITY_KWH) * 100
                
                for bus in active_buses:
                    original_soc = bus['soc_percent']
                    bus['soc_percent'] = max(0, original_soc - soc_reduction_percent)
                    # Recalculate the charging offset based on the new, lower SOC
                    closest_soc = profile_socs[np.abs(profile_socs - bus['soc_percent']).argmin()]
                    bus['charge_offset'] = soc_to_offset_map.get(closest_soc, 0)
                    print(f"  - Bus {bus['id']}: SOC adjusted from {original_soc:.2f}% to {bus['soc_percent']:.2f}%. New charge offset: {bus['charge_offset']}")

                unaccounted_energy_kWh = 0.0 # Deficit has been accounted for.

        # At the END of an off-peak window, calculate the energy deficit and clear the queue.
        if not now_is_off_peak and was_off_peak:
            print(f"\nOff-peak window ending at minute {minute}. Calculating energy deficit...")
            newly_unaccounted_kWh = 0
            for bus in active_buses:
                if bus['status'] != 'finished':
                    remaining_profile = charge_profile['grid_load'].iloc[bus['charge_offset']:]
                    energy_needed_kWh = remaining_profile.sum() / 60.0
                    newly_unaccounted_kWh += energy_needed_kWh
            
            if newly_unaccounted_kWh > 0:
                unaccounted_energy_kWh += newly_unaccounted_kWh
                print(f"  - Deficit from this window: {newly_unaccounted_kWh:.2f} kWh. Total cumulative deficit: {unaccounted_energy_kWh:.2f} kWh.")
            
            active_buses.clear()
            print("  - Bus queue cleared.")

        # Add buses that arrived this minute to the active list.
        arrived_this_minute = bus_arrivals_df[bus_arrivals_df['arrival_minute'].round() == minute]
        for _, bus_data in arrived_this_minute.iterrows():
            initial_soc = bus_data['soc_percent']
            closest_soc = profile_socs[np.abs(profile_socs - initial_soc).argmin()]
            start_offset = soc_to_offset_map.get(closest_soc, 0)
            active_buses.append({
                'id': bus_data['bus_id'], 'status': 'waiting', 
                'charge_offset': start_offset, 'soc_percent': initial_soc
            })

        current_minute_load = 0
        buses_charging_this_minute = 0

        if now_is_off_peak:
            active_buses.sort(key=lambda b: b['status'] != 'charging')
            for bus in active_buses:
                if bus['status'] == 'finished' or bus['charge_offset'] >= len(charge_profile):
                    bus['status'] = 'finished'; continue
                
                if bus['status'] == 'waiting':
                    if buses_charging_this_minute < max_concurrent_charging:
                        bus['status'] = 'charging'
                    else:
                        continue
                
                if bus['status'] == 'charging':
                    load = charge_profile['grid_load'].iloc[bus['charge_offset']]
                    current_minute_load += load
                    buses_charging_this_minute += 1
                    bus['charge_offset'] += 1
        else:
            for bus in active_buses:
                if bus['status'] == 'charging': bus['status'] = 'waiting'
        
        grid_df.loc[minute, 'total_grid_load'] = current_minute_load
        grid_df.loc[minute, 'buses_on_charge'] = buses_charging_this_minute
        was_off_peak = now_is_off_peak
        active_buses = [b for b in active_buses if b['status'] != 'finished']
        
    return grid_df

def simulate_balanced_off_peak_charging(bus_arrivals_df, charge_profile, simulation_duration_minutes):
    """
    Simulates charging during off-peak windows, balancing the load to a target power level.
    MODIFIED: Accepts simulation_duration_minutes as a parameter.
    """
    print("Starting BALANCED OFF-PEAK grid load simulation...")

    def is_off_peak(minute):
        """Returns True if the minute is OUTSIDE of the defined peak windows."""
        time_of_day = minute % 1440
        is_morning_peak = (360 <= time_of_day < 540)  # 06:00 to 09:00
        is_evening_peak = (1020 <= time_of_day < 1260) # 17:00 to 21:00
        return not (is_morning_peak or is_evening_peak)

    charge_profile.fillna(0, inplace=True)
    charge_profile['soc_rounded'] = charge_profile['soc'].round(0)
    soc_to_offset_map = charge_profile.reset_index().groupby('soc_rounded')['index'].first().to_dict()
    profile_socs = np.array(sorted(soc_to_offset_map.keys()))

    simulation_start_minute = 0
    simulation_end_minute = simulation_start_minute + simulation_duration_minutes
    time_index_minutes = range(simulation_start_minute, simulation_end_minute)

    total_energy_needed_kWh = 0
    for _, bus in bus_arrivals_df.iterrows():
        arrival_soc = bus['soc_percent']
        lookup_soc = arrival_soc
        closest_profile_soc = profile_socs[np.abs(profile_socs - lookup_soc).argmin()]
        start_offset = soc_to_offset_map.get(closest_profile_soc, 0)
        energy_for_bus = charge_profile['grid_load'].iloc[start_offset:].sum() / 60.0
        total_energy_needed_kWh += energy_for_bus

    total_off_peak_minutes = sum(1 for minute in time_index_minutes if is_off_peak(minute))
    available_hours = total_off_peak_minutes / 60.0

    if available_hours > 0:
        target_power_level_kW = total_energy_needed_kWh / available_hours
    else:
        target_power_level_kW = float('inf')
    
    print(f"\nTotal energy demand: {total_energy_needed_kWh:.2f} kWh")
    print(f"Available off-peak charging time: {available_hours:.2f} hours")
    print(f"Calculated target balanced power level: {target_power_level_kW:.2f} kW")

    grid_df = pd.DataFrame({'total_grid_load': 0.0, 'buses_on_charge': 0}, index=time_index_minutes)
    active_buses = []

    for minute in time_index_minutes:
        arrived_this_minute = bus_arrivals_df[bus_arrivals_df['arrival_minute'].round() == minute]
        for _, bus in arrived_this_minute.iterrows():
            arrival_soc = bus['soc_percent']
            lookup_soc = arrival_soc
            closest_soc = profile_socs[np.abs(profile_socs - lookup_soc).argmin()]
            start_offset = soc_to_offset_map.get(closest_soc, 0)
            active_buses.append({'id': bus['bus_id'], 'status': 'waiting', 'charge_offset': start_offset})

        is_charging_window = is_off_peak(minute)
        current_minute_load = 0
        current_buses_charging = 0

        if not is_charging_window:
            for bus in active_buses:
                if bus['status'] == 'charging':
                    bus['status'] = 'paused'
        
        buses_to_process = sorted(active_buses, key=lambda b: b['status'] != 'charging')

        for bus in buses_to_process:
            if bus['status'] == 'finished':
                continue

            if bus['status'] == 'charging':
                if bus['charge_offset'] < len(charge_profile):
                    load = charge_profile['grid_load'].iloc[bus['charge_offset']]
                    current_minute_load += load
                    current_buses_charging += 1
                    bus['charge_offset'] += 1
                    if bus['charge_offset'] >= len(charge_profile):
                        bus['status'] = 'finished'
                else:
                    bus['status'] = 'finished'

            elif is_charging_window and bus['status'] in ['waiting', 'paused']:
                if bus['charge_offset'] < len(charge_profile):
                    next_bus_load = charge_profile['grid_load'].iloc[bus['charge_offset']]
                    if current_minute_load + next_bus_load <= target_power_level_kW:
                        bus['status'] = 'charging'
                        current_minute_load += next_bus_load
                        current_buses_charging += 1
                        bus['charge_offset'] += 1
                        if bus['charge_offset'] >= len(charge_profile):
                            bus['status'] = 'finished'
                else:
                    bus['status'] = 'finished'

        grid_df.loc[minute, 'total_grid_load'] = current_minute_load
        grid_df.loc[minute, 'buses_on_charge'] = current_buses_charging
        
    return grid_df

def simulate_capped_off_peak_charging(bus_arrivals_df, charge_profile, simulation_duration_minutes, num_buses):
    """
    Simulates charging during off-peak windows with a specific power cap ONLY during the evening/overnight period.
    Daytime charging remains greedy (unlimited).
    """
    print("Starting CAPPED OFF-PEAK grid load simulation...")

    # --- 1. Define the evening power cap based on the number of buses ---
    if num_buses <= 25:
        evening_power_cap_kW = 500.0
    elif num_buses <= 30:
        evening_power_cap_kW = 600.0
    elif num_buses <= 40: # Covers 31-40
        evening_power_cap_kW = 800.0
    elif num_buses <= 50: # Covers 41-50
        evening_power_cap_kW = 900.0
    else: # For any number of buses over 50, there is no cap
        evening_power_cap_kW = float('inf')
    
    print(f"Applying evening/overnight power cap of {evening_power_cap_kW} kW for {num_buses} buses.")

    charge_profile.fillna(0, inplace=True)
    charge_profile['soc_rounded'] = charge_profile['soc'].round(0)
    soc_to_offset_map = charge_profile.reset_index().groupby('soc_rounded')['index'].first().to_dict()
    profile_socs = np.array(sorted(soc_to_offset_map.keys()))
    
    simulation_start_minute = 0
    simulation_end_minute = simulation_start_minute + simulation_duration_minutes
    
    time_index_minutes = range(simulation_start_minute, simulation_end_minute)
    grid_df = pd.DataFrame({'total_grid_load': 0.0, 'buses_on_charge': 0}, index=time_index_minutes)
    
    active_buses = []

    for minute in time_index_minutes:
        # Add newly arrived buses to the active pool
        arrived_this_minute = bus_arrivals_df[bus_arrivals_df['arrival_minute'].round() == minute]
        for _, bus in arrived_this_minute.iterrows():
            closest_soc = profile_socs[np.abs(profile_socs - bus['soc_percent']).argmin()]
            start_offset = soc_to_offset_map.get(closest_soc, 0)
            active_buses.append({
                'id': bus['bus_id'], 'status': 'waiting', 'charge_offset': start_offset
            })
            
        # --- 2. Determine the charging window type and rules for the current minute ---
        time_of_day = minute % 1440
        is_morning_peak = (360 <= time_of_day < 540)   # 06:00 to 09:00
        is_evening_peak = (1020 <= time_of_day < 1320)  # 17:00 to 22:00
        is_charging_window = not (is_morning_peak or is_evening_peak)
        
        # The night window is when the cap applies
        is_night_window = (time_of_day >= 1320) or (time_of_day < 360)

        current_minute_load = 0
        current_buses_charging = 0

        # In a peak window, pause any charging buses. No new buses can start.
        if not is_charging_window:
            for bus in active_buses:
                if bus['status'] == 'charging':
                    bus['status'] = 'paused'
        # In an off-peak (charging) window, manage the load
        else:
            # Prioritize buses that are already charging to ensure they continue
            active_buses.sort(key=lambda b: b['status'] != 'charging')
            
            for bus in active_buses:
                if bus['status'] == 'finished' or bus['charge_offset'] >= len(charge_profile):
                    bus['status'] = 'finished'
                    continue
                
                # Rule A: If a bus is already charging, let it continue.
                if bus['status'] == 'charging':
                    load = charge_profile['grid_load'].iloc[bus['charge_offset']]
                    current_minute_load += load
                    current_buses_charging += 1
                    bus['charge_offset'] += 1
                
                # Rule B: If a bus is waiting/paused, decide if it can START charging.
                elif bus['status'] in ['waiting', 'paused']:
                    can_start_charging = False
                    next_bus_load = charge_profile['grid_load'].iloc[bus['charge_offset']]
                    
                    # --- 3. Apply window-specific logic ---
                    if is_night_window:
                        # During the night, only start if under the power cap
                        if current_minute_load + next_bus_load <= evening_power_cap_kW:
                            can_start_charging = True
                    else:
                        # During the daytime off-peak, it's greedy: always start
                        can_start_charging = True
                    
                    if can_start_charging:
                        bus['status'] = 'charging'
                        current_minute_load += next_bus_load
                        current_buses_charging += 1
                        bus['charge_offset'] += 1

        grid_df.loc[minute, 'total_grid_load'] = current_minute_load
        grid_df.loc[minute, 'buses_on_charge'] = current_buses_charging
        
    return grid_df

def run_simulation(args):
    """
    MODIFIED: Main function to run the multi-day simulation.
    """
    # --- Date parsing and validation ---
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError("Incorrect date format. Please use YYYY-MM-DD.")
    
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date.")

    num_days = (end_date - start_date).days + 1
    print(f"Simulation configured to run for {num_days} day(s) from {start_date} to {end_date}.")

    # --- Load Charging Profile ---
    profile_path = f"../charging_profiles/230kWh_battery_{args.charge_rate}_charger_linear.csv"
    try:
        charge_profile = pd.read_csv(profile_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Charging profile not found. Please check the path: '{profile_path}'")

    if not (0 <= args.soc_lower <= args.soc_upper <= 100):
        raise ValueError("SOC bounds must be between 0 and 100, with lower <= upper.")

    # --- Multi-Day Bus Arrival Generation ---
    all_buses_list = []
    for day_index in range(num_days):
        print(f"\nGenerating bus arrivals for day {day_index + 1} of {num_days}...")
        minute_offset = day_index * 1440  # 1440 minutes in a day

        num_buses_day = int(round(args.no_of_buses * 0.9))
        num_buses_eve = int(round(args.no_of_buses * 0.9))
        
        day_start_min = parse_time_to_minutes(args.arrival_start_day)
        day_end_min = parse_time_to_minutes(args.arrival_end_day)
        arrivals_day = skewnorm.rvs(a=args.arrival_skew_day, loc=args.arrival_loc_day * 60, scale=args.arrival_scale_day * 60, size=num_buses_day)
        clipped_arrivals_day = np.clip(arrivals_day, day_start_min, day_end_min)
        
        soc_day = norm.rvs(loc=args.soc_mean_day, scale=args.soc_std_day, size=num_buses_day)
        clipped_soc_day = np.clip(soc_day, args.soc_lower, args.soc_upper)

        eve_start_min = parse_time_to_minutes(args.arrival_start_eve)
        eve_end_min = parse_time_to_minutes(args.arrival_end_eve)
        arrivals_eve = skewnorm.rvs(a=args.arrival_skew_eve, loc=args.arrival_loc_eve * 60, scale=args.arrival_scale_eve * 60, size=num_buses_eve)
        clipped_arrivals_eve = np.clip(arrivals_eve, eve_start_min, eve_end_min)

        soc_eve = norm.rvs(loc=args.soc_mean_eve, scale=args.soc_std_eve, size=num_buses_eve)
        clipped_soc_eve = np.clip(soc_eve, args.soc_lower, args.soc_upper)
        
        # Add the daily minute offset to arrival times
        combined_arrivals = np.concatenate((clipped_arrivals_day, clipped_arrivals_eve)) + minute_offset
        combined_soc = np.concatenate((clipped_soc_day, clipped_soc_eve))

        daily_df = pd.DataFrame({
            'arrival_minute': combined_arrivals,
            'soc_percent': combined_soc
        })
        all_buses_list.append(daily_df)

    # Combine all daily data into a single dataframe
    bus_arrivals_df = pd.concat(all_buses_list).sort_values(by='arrival_minute').reset_index(drop=True)
    bus_arrivals_df['bus_id'] = range(1, len(bus_arrivals_df) + 1)
    
    total_buses = len(bus_arrivals_df)
    print(f"Generated a total of {total_buses} bus arrivals across {num_days} day(s).")

    # --- Run Simulation for the Entire Period ---
    output_dir = "../output_data"
    plots_dir = "../plots"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculate total simulation duration, add 6 hours buffer to see trailing charges
    simulation_duration_minutes = num_days * 1440 + 6 * 60

    if args.strategy == 'unmanaged':
        grid_load_df = simulate_unmanaged_charging(bus_arrivals_df, charge_profile, simulation_duration_minutes)
    elif args.strategy == 'off_peak':
        # Pass the number of buses to the off-peak strategy function
        grid_load_df = simulate_off_peak_charging(bus_arrivals_df, charge_profile, simulation_duration_minutes, args.no_of_buses)
    elif args.strategy == 'balanced_off_peak':
        grid_load_df = simulate_balanced_off_peak_charging(bus_arrivals_df, charge_profile, simulation_duration_minutes)
    elif args.strategy == 'capped_off_peak':
        grid_load_df = simulate_capped_off_peak_charging(bus_arrivals_df, charge_profile, simulation_duration_minutes, args.no_of_buses)
    else:
        raise ValueError(f"Unknown strategy: '{args.strategy}'.")
    
    # --- Process and Save Results ---
    grid_load_df['time_str'] = [minutes_to_time_str(m, start_date) for m in grid_load_df.index]
    grid_load_df.fillna(0, inplace=True)
    grid_load_df['buses_on_charge'] = grid_load_df['buses_on_charge'].astype(int)
    
    # MODIFIED: Filename now includes dates for clarity
    base_name = f"{args.no_of_buses}_buses_{args.charge_rate}_{args.strategy}"
    date_range_str = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
    grid_csv_filename = os.path.join(output_dir, f"{base_name}_{date_range_str}_simulation.csv")
    grid_load_df.to_csv(grid_csv_filename)
    print(f"\nSuccessfully saved multi-day grid load data to '{grid_csv_filename}'")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.plot(grid_load_df.index, grid_load_df['total_grid_load'], label='Total Grid Load (kW)', color='darkblue')
    ax1.set_title(f'Simulated Grid Load ({total_buses} Buses, {args.charge_rate}, {args.strategy.title().replace("_", " ")}) - {num_days} Day(s)')
    ax1.set_xlabel('Date and Time')
    ax1.set_ylabel('Grid Load (kW)')
    
    ax2 = ax1.twinx()
    ax2.fill_between(grid_load_df.index, grid_load_df['buses_on_charge'], alpha=0.2, label='Buses on Charge', color='orange', step='post')
    ax2.set_ylabel('Number of Buses Charging', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # MODIFIED: Tick locations adjusted for multi-day plots
    # Major ticks every 24 hours (1440 minutes), minor every 6 hours (360 minutes)
    major_tick_interval = 1440
    minor_tick_interval = 360 if num_days < 5 else 720 # Adjust for longer simulations
    
    major_ticks = np.arange(0, simulation_duration_minutes, major_tick_interval)
    minor_ticks = np.arange(0, simulation_duration_minutes, minor_tick_interval)
    
    major_tick_labels = [minutes_to_time_str(m, start_date) for m in major_ticks]
    
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels(major_tick_labels, rotation=45, ha="right")
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(True, which='minor', axis='x', linestyle=':')
    ax1.grid(True, which='major', axis='x', linestyle='--')

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.tight_layout()
    grid_plot_filename = os.path.join(plots_dir, f"{base_name}_{date_range_str}_plot.png")
    fig.savefig(grid_plot_filename)
    print(f"Saved grid load plot to '{grid_plot_filename}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate bus fleet arrivals, SOC, and resulting charging grid load for a specified date range.")
    
    # --- NEW Arguments for date range ---
    parser.add_argument('--start_date', type=str, required=True, help='Start date of the simulation (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=True, help='End date of the simulation (YYYY-MM-DD).')

    parser.add_argument('--no_of_buses', type=int, required=True, help='Base number of buses ARRIVING PER DAY in the daytime window.')
    parser.add_argument('--charge_rate', type=str, required=True, help='Charger rate, used to select the profile file (e.g., "30kW").')
    parser.add_argument('--strategy', type=str, default='unmanaged', 
                        choices=['unmanaged', 'off_peak', 'balanced_off_peak', 'capped_off_peak'], 
                        help="Charging strategy to use.")

    # ... (rest of the arguments remain unchanged)
    parser.add_argument('--arrival_start_day', type=str, default='07:00', help='Start of the daytime arrival window (HH:MM).')
    parser.add_argument('--arrival_end_day', type=str, default='16:00', help='End of the daytime arrival window (HH:MM).')
    parser.add_argument('--arrival_skew_day', type=float, default=4.06, help='Skewness (alpha) of the daytime arrival distribution.')
    parser.add_argument('--arrival_loc_day', type=float, default=8.22, help='Location (mu) of the daytime arrival distribution (in hours).')
    parser.add_argument('--arrival_scale_day', type=float, default=3.08, help='Scale (sigma) of the daytime arrival distribution (in hours).')

    parser.add_argument('--arrival_start_eve', type=str, default='16:00', help='Start of the evening arrival window (HH:MM).')
    parser.add_argument('--arrival_end_eve', type=str, default='23:59', help='End of the evening arrival window (HH:MM).')
    parser.add_argument('--arrival_skew_eve', type=float, default=3.11, help='Skewness (alpha) of the evening arrival distribution.')
    parser.add_argument('--arrival_loc_eve', type=float, default=17.48, help='Location (mu) of the evening arrival distribution (in hours).')
    parser.add_argument('--arrival_scale_eve', type=float, default=2.81, help='Scale (sigma) of the evening arrival distribution (in hours).')
    parser.add_argument('--evening_arrival_fraction', type=float, default=0.84, help='Fraction of the daytime bus count that also arrives in the evening.')

    parser.add_argument('--soc_lower', type=float, default=1.0, help='Lower bound for SOC percentage.')
    parser.add_argument('--soc_upper', type=float, default=97.0, help='Upper bound for SOC percentage.')
    parser.add_argument('--soc_mean_day', type=float, default=53.82, help='Mean of the daytime SOC normal distribution.')
    parser.add_argument('--soc_std_day', type=float, default=16.77, help='Std. dev. of the daytime SOC normal distribution.')
    parser.add_argument('--soc_mean_eve', type=float, default=52.79, help='Mean of the evening SOC normal distribution.')
    parser.add_argument('--soc_std_eve', type=float, default=15.27, help='Std. dev. of the evening SOC normal distribution.')

    args = parser.parse_args()
    
    try:
        run_simulation(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}")