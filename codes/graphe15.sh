#!/bin/bash

if [ -e 'output_graphe_15.txt' ]; then
    rm 'output_graphe_15.txt'
    echo "output_graphe_15.txt has been removed."
else
    echo "output_graphe_15.txt does not exist in the current folder."
fi

touch "output_graphe_15.txt"
echo "output_graphe_15.txt has been created."

for i in $(seq 2002 2019);
do
    python utils_paper.py -y $i
done

python graphe15_across_years.py