#!/bin/bash

# Check if output.txt exists in the current folder
if [ -e "output.txt" ]; then
    # Remove the file if it exists
    rm "output.txt"
    echo "output.txt has been removed."
else
    echo "output.txt does not exist in the current folder."
fi

# Use the touch command to create an empty output.txt file
touch "output.txt"

echo "output.txt has been created."

for i in $(seq 2018 2019);
do
    python reform_towards_individualization.py -y $i
done

python graphe15_across_years.py