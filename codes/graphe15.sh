#!/bin/bash

################################################################################
# Ensures we are at the good location to launch codes 
expected_path="/app/codes"
current_path=$(pwd)

if [ "$current_path" != "$expected_path" ]; then
    echo "Error: Please run this script from the 'codes' folder."
    exit 1
fi
################################################################################



################################################################################
# Delete output_graphe_15.txt that will store text outputs
if [ -e 'output_graphe_15.txt' ]; then
    rm 'output_graphe_15.txt'
    echo "output_graphe_15.txt has been removed."
else
    echo "output_graphe_15.txt does not exist in the current folder."
fi

touch "output_graphe_15.txt"
echo "output_graphe_15.txt has been created."
################################################################################


################################################################################
# For the first use, creates the folder excel and its year subfolders
excel_folder="./excel"

# Create the excel folder if it does not exist
if [ ! -d "$excel_folder" ]; then
    mkdir "$excel_folder"
    echo "Created 'excel' folder."
fi

# Create subfolders 2002 to 2019 if they do not exist
for year_folder in {2002..2019}; do
    year_folder_path="$excel_folder/$year_folder"

    if [ ! -d "$year_folder_path" ]; then
        mkdir "$year_folder_path"
        echo "Created '$year_folder' subfolder."
    fi
done

echo "Folder and subfolders creation completed."
################################################################################


################################################################################
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
################################################################################


################################################################################
# For the first use, creates the subfolders of the outputs folder 
cd ../outputs

subfolders=("cdf_primary_secondary" "median_share_primary" "revenue_functions"  "winners_political_economy" "extensive_expectation_primary" "mtr_ratio_by_primary" "participation_elasticity" "welfare" "extensive_expectation_secondary"  "mtr_ratio_by_secondary"  "pdf_primary_secondary" "winners_over_time")

for subfolder in "${subfolders[@]}"; do
    subfolder_path="$subfolder"

    if [ ! -d "$subfolder_path" ]; then
        mkdir "$subfolder_path"
        echo "Created '$subfolder' subfolder."
    fi
done

echo "Folder and subfolders creation completed."

cd ../codes
################################################################################

################################################################################
# Delete all png files that store outputs 
outputs_folder_path="../outputs"

if [ -d "$outputs_folder_path" ]; then
    find "$outputs_folder_path" -type d -exec sh -c 'find "$0" -name "*.png" -type f -delete' {} \;
    echo "All PNG files deleted in subfolders within $outputs_folder_path"
else
    echo "$outputs_folder_path does not exist."
fi
echo "Process completed."
################################################################################



################################################################################
# Get data from the simulations
for i in $(seq 2002 2019);
do
    python simulation_creation.py -y $i
done
################################################################################


################################################################################
# Give all year-specific outputs
for i in $(seq 2002 2019);
do
    python utils_paper.py -y $i
done
################################################################################


################################################################################
# Plots results across time (percentage of winners)
python graphe15_across_years.py
################################################################################
