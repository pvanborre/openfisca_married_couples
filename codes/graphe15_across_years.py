import matplotlib.pyplot as plt

with open('output.txt', 'r') as file:
    lines = file.readlines()

search_pattern = "Pourcentage de gagnants"
matching_lines = [line for line in lines if search_pattern in line]

percentages = []
for line in matching_lines:
    parts = line.split()
    percentage = parts[-1]
    percentages.append(float(percentage))

print("Percentages found:", percentages)

start = 2018
end = 2020
years = list(range(start, end))  


plt.figure()
plt.plot(years, percentages[0:len(percentages):3], marker='o', linestyle='-')
plt.plot(years, percentages[1:len(percentages):3], marker='o', linestyle='-')
plt.plot(years, percentages[2:len(percentages):3], marker='o', linestyle='-')
plt.xlabel('Annees')
plt.ylabel('Pourcentage de gagnants')
plt.title('Pourcentage de gagnants à une réforme vers \n l\'individualisation de l\'impot au cours du temps')
plt.grid()

plt.show()
plt.savefig('../outputs/15/graphe_15.png')


