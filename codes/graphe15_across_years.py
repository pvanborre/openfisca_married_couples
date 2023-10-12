import matplotlib.pyplot as plt

with open('output_graphe_15.txt', 'r') as file:
    lines = file.readlines()

search_pattern = "Pourcentage de gagnants"
matching_lines = [line for line in lines if search_pattern in line]

nombre_scenarios = 3
percentages = [[] for _ in range(nombre_scenarios)]

for line in matching_lines:
    parts = line.split()
    annee = int(parts[-3])
    scenario = int(parts[-2])
    percentage = float(parts[-1])
    percentages[scenario].append([annee,percentage])

for i in range(nombre_scenarios):
    percentages[i].sort(key=lambda lst:lst[0]) # on trie sur les années

print("Percentages found:", percentages)


plt.figure()

for i in range(nombre_scenarios):
    plt.plot([inner[0] for inner in percentages[i]], [inner[1] for inner in percentages[i]], marker='o', linestyle='-')

plt.xlabel('Annees')
plt.ylabel('Pourcentage de gagnants')
plt.title('Pourcentage de gagnants à une réforme vers \n l\'individualisation de l\'impot au cours du temps')
plt.grid()

plt.show()
plt.savefig('../outputs/15/graphe_15.png')


