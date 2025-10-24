import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def minutes_to_time_str(minutes):
    """Converts minutes from midnight to a HH:MM time string."""
    if pd.isna(minutes) or not np.isfinite(minutes):
        return ""
    minutes_in_day = int(minutes) % 1440
    hours = minutes_in_day // 60
    mins = minutes_in_day % 60
    return f"{int(hours):02d}:{int(mins):02d}"

def plot_comparison_profiles():
    """
    Finds all multi-day simulation CSVs, groups them by charge rate and strategy,
    and creates one plot per group comparing the median load of different bus counts.
    All plots will share the same y-axis limits for consistent comparison.
    """
    data_dir = "../output_data"
    plots_dir = "../plots"
    os.makedirs(plots_dir, exist_ok=True)

    try:
        all_files = [f for f in os.listdir(data_dir) if f.endswith('_simulation.csv')]
        if not all_files:
            print(f"No simulation CSV files found in '{data_dir}'. Please run a multi-day simulation first.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{data_dir}' was not found.")
        return

    # --- 1. Group files by charge rate AND strategy ---
    grouped_files = defaultdict(list)
    for filename in all_files:
        match = re.search(r'(\d+)_buses_([\w]+)_([\w_]+)', filename)
        if match:
            charge_rate = match.group(2)
            strategy = match.group(3)
            grouped_files[(charge_rate, strategy)].append(filename)
        else:
            print(f"Warning: Could not parse parameters from '{filename}'. It will be skipped.")

    if not grouped_files:
        print("No valid simulation files found to plot.")
        return
        
    print(f"Found {len(grouped_files)} groups of simulations to plot...")

    # --- 2. Pre-process all files to calculate median profiles ---
    all_profiles = {}
    print("Pre-processing all files to calculate median profiles...")

    for filename in all_files:
        file_path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(file_path)
            df['DateTime'] = pd.to_datetime(df['time_str'])
            df['minute_of_day'] = df['DateTime'].dt.hour * 60 + df['DateTime'].dt.minute

            profile_df = df.groupby('minute_of_day')['total_grid_load'].agg(median='median').reset_index()
            
            # Ensure a full 1440-minute profile
            full_day_minutes = pd.DataFrame({'minute_of_day': range(1440)})
            profile_df = pd.merge(full_day_minutes, profile_df, on='minute_of_day', how='left')
            profile_df.interpolate(method='linear', inplace=True)
            profile_df.fillna(0, inplace=True)
            
            # Store the processed profile to avoid re-calculating
            all_profiles[filename] = profile_df

        except Exception as e:
            print(f"  Warning: Could not process {filename} during pre-scan. Error: {e}")

    # --- 3. Loop Through Each Group and Create a Plot ---
    for (charge_rate, strategy), files_in_group in grouped_files.items():
        
        strategy_title = strategy.replace("_", " ").title()
        print(f"\nProcessing Group (Original): {charge_rate} / {strategy_title}...")

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18.5, 9))
        
        # --- Add TOU tariff shading ---
        # Standard times
        standard_periods = [
            (480, 1020),  # 09:00 - 18:00
            (1200, 1320)  # 20:00 - 22:00
        ]
        for i, (start, end) in enumerate(standard_periods):
            ax.axvspan(start, end, color='yellow', alpha=0.15, lw=0, label='Standard' if i == 0 else "")

        # Peak times (red) - UPDATED to 6pm-8pm
        peak_periods = [
            (360, 480),   # 07:00 - 09:00
            (1020, 1200)  # 18:00 - 20:00
        ]
        for i, (start, end) in enumerate(peak_periods):
            ax.axvspan(start, end, color='red', alpha=0.15, lw=0, label='Peak' if i == 0 else "")

        # --- Plot each file's median line using pre-calculated profiles ---
        sorted_files = sorted(files_in_group, key=lambda f: int(re.match(r'(\d+)_buses', f).group(1)))
        
        print("  Calculating daily energy consumption for each profile:")
        for filename in sorted_files:
            if filename not in all_profiles:
                continue 

            bus_count_match = re.match(r'(\d+)_buses', filename)
            label = f"{bus_count_match.group(1)} Buses" if bus_count_match else filename
            
            profile_df = all_profiles[filename]
            
            # --- ENERGY CALCULATION (NEW) ---
            # Total daily energy
            total_energy_kwh = profile_df['median'].sum() / 60

            # Peak energy (7am-9am and 6pm-8pm)
            is_peak = (
                (profile_df['minute_of_day'] >= 420) & (profile_df['minute_of_day'] < 540) |  # 07:00-09:00
                (profile_df['minute_of_day'] >= 1080) & (profile_df['minute_of_day'] < 1200) # 18:00-20:00
            )
            peak_energy_kwh = profile_df.loc[is_peak, 'median'].sum() / 60
            
            # Percentage calculation
            peak_percentage = (peak_energy_kwh / total_energy_kwh * 100) if total_energy_kwh > 0 else 0

            # Print all calculated values
            print(
                f"    - {label}: Total: {total_energy_kwh:,.2f} kWh | "
                f"Peak: {peak_energy_kwh:,.2f} kWh ({peak_percentage:.1f}%)"
            )
            
            ax.plot(profile_df['minute_of_day'], profile_df['median'], label=label, lw=2.5, alpha=0.8)

        # --- 4. Final Plot Formatting for the Group ---
        #ax.set_title(f'24-Hour Median Grid Load Comparison ({charge_rate}, {strategy_title})', fontsize=18)
        ax.set_xlabel('Time of Day', fontsize=18)
        ax.set_ylabel('Median Grid Load (kW)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=15)

        ax.set_xlim(0, 1439)
        # Set fixed y-axis limit to 1000 (NEW)
        ax.set_ylim(0, 1200)

        tick_locations = np.arange(0, 1441, 120)
        tick_labels = [minutes_to_time_str(m) for m in tick_locations]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.75, linestyle='-')

        handles, labels = ax.get_legend_handles_labels()
        fleet_items = {l: h for h, l in zip(handles, labels) if "Buses" in l}
        tou_items = {l: h for h, l in zip(handles, labels) if l in ["Standard", "Peak"]}
        
        sorted_labels = sorted(fleet_items.keys(), key=lambda x: int(x.split()[0])) + sorted(tou_items.keys())
        sorted_handles = [fleet_items[l] for l in sorted_labels if l in fleet_items] + [tou_items[l] for l in sorted_labels if l in tou_items]

        ax.legend(sorted_handles, sorted_labels, title="Fleet Size / TOU", fontsize=15, title_fontsize=15)
        fig.tight_layout()
        
        output_filename = os.path.join(plots_dir, f"comparison_profile_{charge_rate}_{strategy}.png")
        try:
            fig.savefig(output_filename, dpi=300)
            print(f"  Success! Plot saved to '{output_filename}'")
        except Exception as e:
            print(f"  Error saving the plot: {e}")

        plt.close(fig)

def plot_comparison_profiles_smoothed():
    """
    Creates SMOOTHED plots by applying a rolling average to the median load data.
    """
    data_dir = "../output_data"
    plots_dir = "../plots"
    os.makedirs(plots_dir, exist_ok=True)

    try:
        all_files = [f for f in os.listdir(data_dir) if f.endswith('_simulation.csv')]
        if not all_files:
            return
    except FileNotFoundError:
        return

    # --- 1. Group files ---
    grouped_files = defaultdict(list)
    for filename in all_files:
        match = re.search(r'(\d+)_buses_([\w]+)_([\w_]+)', filename)
        if match:
            charge_rate, strategy = match.group(2), match.group(3)
            grouped_files[(charge_rate, strategy)].append(filename)

    # --- 2. Pre-process and smooth profiles ---
    all_profiles = {}
    print("\nPre-processing all files for SMOOTHED plots...")
    for filename in all_files:
        file_path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(file_path)
            df['DateTime'] = pd.to_datetime(df['time_str'])
            df['minute_of_day'] = df['DateTime'].dt.hour * 60 + df['DateTime'].dt.minute

            profile_df = df.groupby('minute_of_day')['total_grid_load'].agg(median='median').reset_index()
            
            full_day_minutes = pd.DataFrame({'minute_of_day': range(1440)})
            profile_df = pd.merge(full_day_minutes, profile_df, on='minute_of_day', how='left')
            profile_df.interpolate(method='linear', inplace=True)
            profile_df.fillna(0, inplace=True)
            
            # Apply a rolling average for smoothing
            smoothing_window = 30 
            profile_df['median_smoothed'] = profile_df['median'].rolling(
                window=smoothing_window, center=True, min_periods=1
            ).mean()
            
            all_profiles[filename] = profile_df
        except Exception as e:
            print(f"  Warning: Could not process {filename} during pre-scan for smoothed plot. Error: {e}")

    # --- 3. Loop Through Each Group and Create a SMOOTHED Plot ---
    for (charge_rate, strategy), files_in_group in grouped_files.items():
        
        strategy_title = strategy.replace("_", " ").title()
        print(f"\nProcessing Group (Smoothed): {charge_rate} / {strategy_title}...")

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18.5, 9))
        
        # --- Add TOU tariff shading ---
        standard_periods = [
            (480, 1020),  # 09:00 - 18:00
            (1200, 1320)  # 20:00 - 22:00
        ]
        for i, (start, end) in enumerate(standard_periods):
            ax.axvspan(start, end, color='yellow', alpha=0.15, lw=0, label='Standard' if i == 0 else "")

        peak_periods = [
            (360, 480),   # 07:00 - 09:00
            (1020, 1200)  # 18:00 - 20:00
        ]
        for i, (start, end) in enumerate(peak_periods):
            ax.axvspan(start, end, color='red', alpha=0.15, lw=0, label='Peak' if i == 0 else "")

        # --- Plot each file's SMOOTHED median line ---
        sorted_files = sorted(files_in_group, key=lambda f: int(re.match(r'(\d+)_buses', f).group(1)))
        
        # Print energy stats (calculated from original, non-smoothed data for accuracy)
        print("  Calculating daily energy consumption (from original, non-smoothed data):")
        for filename in sorted_files:
            if filename not in all_profiles:
                continue

            bus_count_match = re.match(r'(\d+)_buses', filename)
            label = f"{bus_count_match.group(1)} Buses" if bus_count_match else filename
            
            profile_df = all_profiles[filename]

            # --- ENERGY CALCULATION (NEW) ---
            # Total daily energy
            total_energy_kwh = profile_df['median'].sum() / 60

            # Peak energy (7am-9am and 6pm-8pm)
            is_peak = (
                (profile_df['minute_of_day'] >= 420) & (profile_df['minute_of_day'] < 540) |  # 07:00-09:00
                (profile_df['minute_of_day'] >= 1080) & (profile_df['minute_of_day'] < 1200) # 18:00-20:00
            )
            peak_energy_kwh = profile_df.loc[is_peak, 'median'].sum() / 60
            
            # Percentage calculation
            peak_percentage = (peak_energy_kwh / total_energy_kwh * 100) if total_energy_kwh > 0 else 0
            
            # Print all calculated values
            print(
                f"    - {label}: Total: {total_energy_kwh:,.2f} kWh | "
                f"Peak: {peak_energy_kwh:,.2f} kWh ({peak_percentage:.1f}%)"
            )
            
            # Plot the smoothed data for visualization
            ax.plot(profile_df['minute_of_day'], profile_df['median_smoothed'], label=label, lw=2.5, alpha=0.8)

        # --- 4. Final Plot Formatting for the Group ---
        #ax.set_title(f'24-Hour Smoothed Median Grid Load Comparison ({charge_rate}, {strategy_title})', fontsize=18)
        ax.set_xlabel('Time of Day', fontsize=18)
        ax.set_ylabel('Median Grid Load (kW)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=15)

        ax.set_xlim(0, 1439)
        # Set fixed y-axis limit to 1000 (NEW)
        ax.set_ylim(0, 1200)

        tick_locations = np.arange(0, 1441, 120)
        tick_labels = [minutes_to_time_str(m) for m in tick_locations]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.75, linestyle='-')

        handles, labels = ax.get_legend_handles_labels()
        fleet_items = {l: h for h, l in zip(handles, labels) if "Buses" in l}
        tou_items = {l: h for h, l in zip(handles, labels) if l in ["Standard", "Peak"]}
        
        sorted_labels = sorted(fleet_items.keys(), key=lambda x: int(x.split()[0])) + sorted(tou_items.keys())
        sorted_handles = [fleet_items[l] for l in sorted_labels if l in fleet_items] + [tou_items[l] for l in sorted_labels if l in tou_items]

        ax.legend(sorted_handles, sorted_labels, title="Fleet Size / TOU", fontsize=15, title_fontsize=15)
        fig.tight_layout()
        
        output_filename = os.path.join(plots_dir, f"comparison_profile_smoothed_{charge_rate}_{strategy}.png")
        try:
            fig.savefig(output_filename, dpi=300)
            print(f"  Success! Smoothed plot saved to '{output_filename}'")
        except Exception as e:
            print(f"  Error saving the smoothed plot: {e}")

        plt.close(fig)

if __name__ == '__main__':
    # Generate the original, discrete plots
    plot_comparison_profiles()
    
    # Generate the new, smoothed plots
    plot_comparison_profiles_smoothed()