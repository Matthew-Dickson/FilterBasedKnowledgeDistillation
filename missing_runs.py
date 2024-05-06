import os
import json

folder_path = r'data\results\feature-all' 
max_number = 971


import os
import re

def find_missing_numbers(directory, max_number):
    existing_numbers = set()
    
    pattern = r'_(\d+)\.json$'

    for filename in os.listdir(directory):
        match = re.search(pattern, filename)
        if match:
            number = int(match.group(1))
            existing_numbers.add(number)

    missing_numbers = set(range(1, max_number + 1)) - existing_numbers

    return list(missing_numbers)

if __name__ == "__main__":
    missing_numbers = find_missing_numbers(folder_path, max_number)
    
    if missing_numbers:
        print("Missing numbers:", missing_numbers)
    else:
        print("No missing numbers found within the specified range.")