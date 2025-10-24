#!/bin/bash

# This script runs the simulate_charging_multi_day.py script for a list of different
# fleet sizes and charging strategies using derived distribution parameters for a
# specified date range.

# --- Configuration ---
START_DATE="2025-01-01"
END_DATE="2025-01-31"

BUS_COUNTS=(30 40 50)
STRATEGIES=("unmanaged")
CHARGE_RATE="60kW"

# --- Arrival & SOC Parameters (Updated from data analysis) ---

# --- Daytime Window (07:00 - 15:59) ---
ARRIVAL_DAY_START="07:00"
ARRIVAL_DAY_END="15:59"
ARRIVAL_DAY_SKEW=4.18
ARRIVAL_DAY_LOC_HR=7.95
ARRIVAL_DAY_SCALE_HR=3.24
SOC_DAY_MEAN=54.38
SOC_DAY_STD=17.75

# --- Evening Window (16:00 - 23:59) ---
ARRIVAL_EVE_START="16:00"
ARRIVAL_EVE_END="23:59"
ARRIVAL_EVE_SKEW=3.03
ARRIVAL_EVE_LOC_HR=17.35   # In hours
ARRIVAL_EVE_SCALE_HR=2.98  # In hours
SOC_EVE_MEAN=55.35
SOC_EVE_STD=16.07

# Fraction of the daytime buses that also arrive during the evening
EVENING_ARRIVAL_FRACTION=0.84

# --- SOC Clipping Bounds ---
SOC_LOWER_BOUND=1.0
SOC_UPPER_BOUND=97.0

# --- Script Execution ---
PYTHON_CMD="python3"
# MODIFIED: Ensure this points to the correct python script that handles date ranges.
SCRIPT_NAME="bus_charging_simulator.py"

# --- Execution Loop ---
echo "Starting batch simulation from $START_DATE to $END_DATE..."

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Python script '$SCRIPT_NAME' not found."
    exit 1
fi

for count in "${BUS_COUNTS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        echo "--------------------------------------------------------"
        echo "Running simulation for ${count} buses with '${strategy}' strategy..."
        echo "--------------------------------------------------------"

        $PYTHON_CMD $SCRIPT_NAME \
            --start_date "$START_DATE" \
            --end_date "$END_DATE" \
            --no_of_buses "$count" \
            --charge_rate "$CHARGE_RATE" \
            --strategy "$strategy" \
            --arrival_start_day "$ARRIVAL_DAY_START" \
            --arrival_end_day "$ARRIVAL_DAY_END" \
            --arrival_skew_day "$ARRIVAL_DAY_SKEW" \
            --arrival_loc_day "$ARRIVAL_DAY_LOC_HR" \
            --arrival_scale_day "$ARRIVAL_DAY_SCALE_HR" \
            --arrival_start_eve "$ARRIVAL_EVE_START" \
            --arrival_end_eve "$ARRIVAL_EVE_END" \
            --arrival_skew_eve "$ARRIVAL_EVE_SKEW" \
            --arrival_loc_eve "$ARRIVAL_EVE_LOC_HR" \
            --arrival_scale_eve "$ARRIVAL_EVE_SCALE_HR" \
            --evening_arrival_fraction "$EVENING_ARRIVAL_FRACTION" \
            --soc_lower "$SOC_LOWER_BOUND" \
            --soc_upper "$SOC_UPPER_BOUND" \
            --soc_mean_day "$SOC_DAY_MEAN" \
            --soc_std_day "$SOC_DAY_STD" \
            --soc_mean_eve "$SOC_EVE_MEAN" \
            --soc_std_eve "$SOC_EVE_STD"

        if [ $? -ne 0 ]; then
            echo "Error: Simulation failed for ${count} buses with '${strategy}' strategy."
        else
            echo "Simulation for ${count} buses with '${strategy}' strategy completed."
        fi
        echo ""
    done
done

echo "Batch simulation finished."