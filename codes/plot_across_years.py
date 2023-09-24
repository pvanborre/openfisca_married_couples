import matplotlib.pyplot as plt

def plot_results(results):
    years = list(range(2005, 2020))  

    numeric_results = [float(result) for result in results]

    plt.figure()
    plt.plot(years, numeric_results, marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Result Value')
    plt.title('Results Over Time')
    plt.grid()

    plt.show()
    plt.savefig('../outputs/graphe_15_.png')

if __name__ == "__main__":
    import sys
    results = sys.argv[1:]  
    plot_results(results)
