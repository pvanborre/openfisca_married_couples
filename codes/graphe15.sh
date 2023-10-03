#!/bin/bash

if [ -e "output.txt" ]; then
    rm "output.txt"
    echo "output.txt has been removed."
else
    echo "output.txt does not exist in the current folder."
fi

touch "output.txt"
echo "output.txt has been created."

for i in $(seq 2005 2019);
do
    python reform_towards_individualization.py -y $i
done

python graphe15_across_years.py