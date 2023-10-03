#!/bin/bash


percentages=()

for i in $(seq 2018 2019); do
    result=$(python reform_towards_individualization.py -y $i)
    
    if [[ $result =~ Pourcentage\ de\ gagnants\ ([0-9]+) ]]; then
        percentage="${BASH_REMATCH[1]}"  
        percentages+=("$percentage")
    else
        echo "Pattern not found in result: $result"
    fi
done

for percentage in "${percentages[@]}"; do
    echo "$percentage"
done

python plot_across_years.py "${percentages[@]}"
