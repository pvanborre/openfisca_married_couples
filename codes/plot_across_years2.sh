#!/bin/bash

# Initialize an associative array to store percentages for different scenarios
declare -A percentages

for i in $(seq 2018 2019); do
    result=$(python reform_towards_individualization.py -y $i)
    
    # Modify the regex pattern to match the new output format
    if [[ $result =~ Pourcentage\ de\ gagnants\ scenario\ ([0-9])\ ([1-9][0-9]?\.[0-9]+) ]]; then
        scenario="${BASH_REMATCH[1]}"
        percentage="${BASH_REMATCH[2]}"
        
        # Store the percentage in the associative array with the scenario as the key
        percentages["$scenario $i"]=$percentage
    else
        echo "Pattern not found in result: $result"
    fi
done
echo "le code s execute"
# Print the collected percentages for each scenario and year
for key in "${!percentages[@]}"; do
    echo "$key: ${percentages[$key]}"
done

# Pass the collected percentages to the Python script
python plot_across_years2.py "${percentages[@]}"
