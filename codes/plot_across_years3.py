import subprocess
import matplotlib.pyplot as plt
import re

# Initialize empty lists to store the arrays
data_arrays = []
start = 2018
# Loop through years from 2005 to 2019
for year in range(start, 2020):
    # Call code.py and capture its output
    cmd = ["python", "code.py", "-y", str(year)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Capture and parse the standard output (stdout)
    stdout_str = result.stdout.strip()
    
    # You can use a regular expression to extract the return value
    match = re.search(r'Return value pourcentage gagnants (\[.*\])', stdout_str)
    
    if match:
        array_str = match.group(1)
        array = eval(array_str)  # Safely evaluate the string as a list
        data_arrays.append(array)
    else:
        print(f"Failed to capture return value for year {year}")

print("mon data array", data_arrays)

# Plot the data
years = list(range(start, 2020))
for i in range(3):  # Assuming the arrays have 3 elements
    plt.plot(years, [arr[i] for arr in data_arrays], label=f'Element {i+1}')

plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.title('Data across Years')
plt.grid(True)
plt.show()
