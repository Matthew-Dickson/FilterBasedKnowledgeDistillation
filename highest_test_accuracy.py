import os
import json

folder_path = r'data\results\feature-last' 

highest_accuracy = 0.0
file_with_highest_accuracy = ""


if __name__ == '__main__':

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r") as file:
                data = json.load(file)
            
            test_accs = data["results"]["test_accuracy"]
            
            if test_accs > highest_accuracy:
                highest_accuracy = test_accs
                file_with_highest_accuracy = filename

    print("File with Highest Test Accuracy:", file_with_highest_accuracy)
    print("Highest Test Accuracy:", highest_accuracy)