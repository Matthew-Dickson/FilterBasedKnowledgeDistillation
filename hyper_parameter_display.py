import os
import json
import matplotlib.pyplot as plt
import numpy as np

folder_paths = [r'data\results\filter', r'data\results\filter-last', r'data\results\basic', r'data\results\traditional', r'data\results\feature-all', r'data\results\feature-last']

results = []

if __name__ == '__main__':
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))  # 3 rows and 2 columns

    for index, folder_path in enumerate(folder_paths):
        stats = {
            "name": folder_path.split("\\")[2],
            "test_accs": [],
            "max_acc": 0.0
        }
        test_accuracies_for_approach = []
        max_acc = 0.0
        approaches = ['Filter-last','Filter-last','Resnet32','Vanilla','Filter-all','Feature-last']

        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, "r") as file:
                    data = json.load(file)
                

                test_accs = data["results"]["test_accuracy"]
                test_accuracies_for_approach.append(test_accs)

                if test_accs > max_acc:
                    max_acc = test_accs
                    file_with_highest_accuracy = filename

        stats["test_accs"] = test_accuracies_for_approach
        stats["max_acc"] = max_acc
        results.append(stats)

    print("Stats", results)

    # Create a list of unique colors for each filter
    colors = ['b', 'g', 'c', 'm', 'y','orange']

    for i, (result, color) in enumerate(zip(results, colors)):
        name = result["name"]
        test_accs = result["test_accs"]
        max_acc = result["max_acc"]

        config_set = list(range(1, len(test_accs) + 1))
        config_set_max = config_set[test_accs.index(max_acc)]

        row, col = divmod(i, 2)  # Calculate the row and column for the current graph

        for j, acc in enumerate(test_accs):
            if j == test_accs.index(max_acc):
                axs[row, col].scatter(config_set[j], acc, color='red', marker='o', label=f'Max ({max_acc})')
            else:
                axs[row, col].scatter(config_set[j], acc, color=color, s=5)

        axs[row, col].set_title(f"{approaches[i]}")
        axs[row, col].set_xlabel("Configuration Set")
        axs[row, col].set_ylabel("Test Accuracy")

        # Set y-axis limits to start at 0 and go up in increments of 10
        axs[row, col].set_ylim(0, 80)

        # Set integer ticks on the x-axis only if there are 10 or fewer values
        if len(config_set) <= 10:
            axs[row, col].set_xticks(np.arange(min(config_set), max(config_set) + 1, 1))

    # Remove any unused subplots
    for i in range(len(folder_paths), 6):
        row, col = divmod(i, 2)
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.savefig(r'data\results\images\hyperparameter_results.png')
    plt.show()
