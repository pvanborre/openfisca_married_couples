import matplotlib.pyplot as plt
import sys

def main():
    # Parse command-line arguments (percentages for different scenarios and years)
    percentages = sys.argv[1:]  # The first argument is the script name

    # Extract data for different scenarios and years
    scenario_data = {}
    for percentage in percentages:
        parts = percentage.split()
        if len(parts) == 3:
            scenario, year, value = parts
            scenario = int(scenario)
            year = int(year)
            value = float(value)
            
            if scenario not in scenario_data:
                scenario_data[scenario] = {"years": [], "values": []}
            
            scenario_data[scenario]["years"].append(year)
            scenario_data[scenario]["values"].append(value)

    # Plot data for different scenarios
    for scenario, data in scenario_data.items():
        plt.plot(data["years"], data["values"], label=f"Scenario {scenario}")

    # Add labels, legends, and other plot settings
    plt.xlabel("Year")
    plt.ylabel("Percentage")
    plt.legend()
    plt.title("Percentage Across Scenarios and Years")
    
    # Display or save the plot
    plt.show()
    plt.savefig('../outputs/graphe_15.png')

if __name__ == "__main__":
    main()
