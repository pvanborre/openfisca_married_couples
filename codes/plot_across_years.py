import matplotlib.pyplot as plt

def plot_results(results):
    years = list(range(2005, 2020))  

    numeric_results = [float(result) for result in results]

    plt.figure()
    plt.plot(years, numeric_results, marker='o', linestyle='-')
    plt.xlabel('Annees')
    plt.ylabel('Pourcentage de gagnants')
    plt.title('Pourcentage de gagnants à une réforme vers \n l\'individualisation de l\'impot au cours du temps')
    plt.grid()

    plt.show()
    plt.savefig('../outputs/graphe_15_.png')

if __name__ == "__main__":
    import sys
    results = sys.argv[1:]  
    plot_results(results)
