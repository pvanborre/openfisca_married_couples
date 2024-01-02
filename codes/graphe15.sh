#!/bin/bash

# Delete output_graphe_15.txt that will store outputs
if [ -e 'output_graphe_15.txt' ]; then
    rm 'output_graphe_15.txt'
    echo "output_graphe_15.txt has been removed."
else
    echo "output_graphe_15.txt does not exist in the current folder."
fi

touch "output_graphe_15.txt"
echo "output_graphe_15.txt has been created."

# Delete all csv files that store data 

main_folder_path="./excel"

for year_folder in {2002..2019}; do
    year_folder_path="$main_folder_path/$year_folder"

    if [ -d "$year_folder_path" ]; then
        find "$year_folder_path" -name "*.csv" -type f -delete
        echo "All CSV files deleted in $year_folder_path"
    else
        echo "$year_folder_path does not exist."
    fi
done

echo "Process completed."



for i in $(seq 2002 2019);
do
    python simulation_creation.py -y $i
done

for i in $(seq 2002 2019);
do
    python utils_paper.py -y $i
done

python graphe15_across_years.py