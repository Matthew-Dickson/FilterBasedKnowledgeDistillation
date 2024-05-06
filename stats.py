import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from functions.utilities.file_util import FileUtil
import os

stats_load_file_path = r'./data/results/runs/ce_history.json' 
stats_save_file_path = r'./data/results/stats/save_resnet32_stats.json' 
image_load_folder_path = r'./data/results/runs/' 
image_save_folder_path = r'data'
bar_load_folder_path = r'data\results\stats' 
bar_save_file_path = r'data\mean_test_accuracy_approaches_with_std.png'
hypothesis_test_load_folder_path = r'data\results\stats'
hypothesis_test_save_file_path = r'data\hypothesis_test.json'
alpha =  0.05
run_bar = True
run_stats = False
run_hypothesis_test = False
run_image = False

def get_mean_of_array(arr,axis=0):
    mean = np.mean(arr,axis)
    return mean

def get_std_of_array(arr,axis=0):
    mean = np.std(arr,axis)
    return mean
    
parser = argparse.ArgumentParser()
parser.add_argument('--stats-load-file-path', help='load data for stats', default=stats_load_file_path)
parser.add_argument('--stats-save-file-path', help='save stat file to path', default=stats_save_file_path)
parser.add_argument('--run-stats', help='Runs the mean and stds',type=bool, default=run_stats)
parser.add_argument('--image-load-folder-path', help='load data to produce images', default=image_load_folder_path)
parser.add_argument('--image-save-folder-path', help='save images to path', default=image_save_folder_path)
parser.add_argument('--run-image', help='Gets images',type=bool, default=run_image)
parser.add_argument('--bar-load-folder-path', help='load data to produce images', default=bar_load_folder_path)
parser.add_argument('--bar-save-file-path', help='save bar image to path', default=bar_save_file_path)
parser.add_argument('--run-bar', help='Runs the mean and stds',type=bool, default=run_bar)
parser.add_argument('--hypothesis-test-load-folder-path', help='load data to run hypothesis test', default=hypothesis_test_load_folder_path)
parser.add_argument('--hypothesis-test-save-file-path', help='save hypothesis test to path', default=hypothesis_test_save_file_path)
parser.add_argument('--run-hypothesis-test', help='Runs the hypothesis test',type=bool, default=run_hypothesis_test)
parser.add_argument('--alpha', help='alpha for hypothesis test',type=float, default=alpha)


args = parser.parse_args()

if __name__ == '__main__':
    file_helper = FileUtil()

    if(args.run_stats):
        stats={}
        train_accuracies =  []
        valid_accuracies = []
        test_accuracies = []
        train_losses = []
        valid_losses = []
        train_times = []
        valid_times = []
        convergence_iterations = []

        with open(args.stats_load_file_path, "r") as file:
            data = json.load(file)

        for item in data:

            if("run" in item):
                train_accuracies.append(np.ravel(data[item]["train_accs"]))
                valid_accuracies.append(np.ravel(data[item]["valid_accs"]))
                test_accuracies.append(np.ravel(data[item]["test_accuracy"]))
                train_losses.append(np.ravel(data[item]["train_losses"]))
                valid_losses.append(np.ravel(data[item]["valid_losses"]))
                train_times.append(np.ravel(data[item]["train_times"]))
                valid_times.append(np.ravel(data[item]["valid_times"]))
                convergence_iterations.append(np.ravel(data[item]["convergence_iteration"]))


        mean_train_accuracies =  get_mean_of_array(train_accuracies,1)
        mean_valid_accuracies =  get_mean_of_array(valid_accuracies,1)
        mean_test_accuracies = get_mean_of_array(test_accuracies)
        mean_valid_loss =  get_mean_of_array(valid_losses,1)
        mean_train_loss = get_mean_of_array(train_losses,1)
        mean_convergence_iterations = get_mean_of_array(convergence_iterations)
        mean_train_times =  get_mean_of_array(train_times,1)
        mean_valid_times =  get_mean_of_array(valid_times,1)
        avg_train_time =  get_mean_of_array(mean_train_times)

        std_train_accuracies =  get_std_of_array(train_accuracies,1)
        std_valid_accuracies =  get_std_of_array(valid_accuracies,1)
        std_test_accuracies = get_std_of_array(test_accuracies)
        std_valid_loss =  get_std_of_array(valid_losses,1)
        std_train_loss = get_std_of_array(train_losses,1)
        std_convergence_iterations = get_std_of_array(convergence_iterations)
        std_train_times =  get_std_of_array(train_times,1)
        std_valid_times =  get_std_of_array(valid_times,1)
        std_avg_train_time =  get_std_of_array(mean_train_times)



        stats["mean_train_accuracies"] = mean_train_accuracies.tolist()
        stats["mean_valid_accuracies"] = mean_valid_accuracies.tolist() 
        stats["mean_test_accuracy"] = mean_test_accuracies.tolist()[0]
        stats["mean_valid_loss"] = mean_valid_loss.tolist() 
        stats["mean_train_loss"] = mean_train_loss.tolist() 
        stats["mean_convergence_iteration"] = mean_convergence_iterations.tolist()[0] 
        stats["mean_train_times"] = mean_train_times.tolist() 
        stats["mean_valid_times"] = mean_valid_times.tolist() 
        stats["test_accuracies"] = np.array(test_accuracies).flatten().tolist()
        stats["avg_train_time"] = np.array(avg_train_time).flatten().tolist()[0]


        
        stats["std_train_accuracies"] = std_train_accuracies.tolist() 
        stats["std_valid_accuracies"] = std_valid_accuracies.tolist() 
        stats["std_test_accuracy"] = std_test_accuracies.tolist()[0]
        stats["std_valid_loss"] = std_valid_loss.tolist() 
        stats["std_train_loss"] = std_train_loss.tolist() 
        stats["std_convergence_iterations"] = std_convergence_iterations.tolist()[0] 
        stats["std_train_times"] = std_train_times.tolist() 
        stats["std_valid_times"] = std_valid_times.tolist() 
        stats["std_avg_train_time"] = np.array(std_avg_train_time).flatten().tolist()[0]

        file_helper.save_to_file(stats,args.stats_save_file_path)

    if(args.run_image):
        stats={ }
        all_train_accuracys = []
        all_train_accuracy_stds = []
        all_valid_accuracys = []
        all_valid_accuracy_stds = []
        all_epochs = []
        all_approaches = []
        for filename in os.listdir(args.image_load_folder_path):
            train_accuracies =  []
            valid_accuracies = []
            test_accuracies = []
            train_losses = []
            valid_losses = []
            train_times = []
            valid_times = []
            convergence_iterations = []

            if filename.endswith(".json"):
                file_path = os.path.join(args.image_load_folder_path, filename)
                with open(file_path, "r") as file:
                    data = json.load(file)
                
                if('traditional_history' in filename):
                    all_approaches.append('Vanilla')
                if('ce_history' in filename):
                    all_approaches.append('Resnet32')
                if('filter_last_history' in filename):
                    all_approaches.append('Filter-last')
                if('filter_history' in filename):
                    all_approaches.append('Filter-all')
                if('feature_last_history' in filename):
                    all_approaches.append('Feature-last')
                if('feature_history' in filename):
                    all_approaches.append('Feature-all')

                
                for item in data:

                    if("run" in item):
                        train_accuracies.append(np.ravel(data[item]["train_accs"]))
                        valid_accuracies.append(np.ravel(data[item]["valid_accs"]))
                        train_times.append(np.ravel(data[item]["train_times"]))
                        valid_times.append(np.ravel(data[item]["valid_times"]))

                mean_train_acc = np.mean(train_accuracies, axis=0)
                mean_valid_acc = np.mean(valid_accuracies, axis=0)
                mean_train_std = np.std(train_accuracies, axis=0)
                mean_valid_std = np.std(valid_accuracies, axis=0)

                mean_train_time = np.mean(train_times, axis=0)
                mean_valid_time= np.mean(valid_times, axis=0)
                mean_train_time_std = np.mean(train_times, axis=0)
                mean_valid_time_std= np.mean(valid_times, axis=0)
                epochs = np.arange(1, len(mean_train_acc) + 1)

                all_train_accuracys.append(mean_train_acc)
                all_train_accuracy_stds.append(mean_train_std)
                all_valid_accuracys.append(mean_valid_acc)
                all_valid_accuracy_stds.append(mean_valid_std)
                all_epochs.append(epochs)

        stats["train_accuracys"] = np.array(all_train_accuracys).tolist()
        stats["train_accuracy_stds"] = np.array(all_train_accuracy_stds).tolist()
        stats["valid_accuracys"] = np.array(all_valid_accuracys).tolist()
        stats["valid_accuracy_stds"] = np.array(all_valid_accuracy_stds).tolist()
        stats["epochs"] =  np.array(all_epochs).tolist()
        stats["approaches"] =  np.array(all_approaches).tolist()
        file_helper.save_to_file(stats,r'./data/results/images/image_stats.json')
       

        number_of_rows = 3
        number_of_plots = len(stats["approaches"])
        fig, axes = plt.subplots(number_of_rows,2, figsize=(12,6*number_of_rows))

        for i in range(number_of_rows):
            for j in range(2):
                index = i * 2 + j
                if index < number_of_plots:

                    ax = axes[i,j]
                    ax.errorbar(stats["epochs"][index], stats["train_accuracys"][index], yerr=stats["train_accuracy_stds"][index], label="Training accuracy", ecolor='r')
                    ax.errorbar(stats["epochs"][index], stats["valid_accuracys"][index], yerr=stats["valid_accuracy_stds"][index], label="Training accuracy", ecolor='r')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.set_title(stats["approaches"][index])
                    ax.grid(True)

        for i in range(number_of_plots,number_of_rows*2):
            fig.delaxes(axes.flatten()[i])

        # Show the grid
        plt.tight_layout()

        # Save the plot in PNG format
        plt.savefig('./data/training_validation_accuracies.png')


    if(args.run_bar):
        test_accs = []
        test_stds = []
        train_times = []
        train_time_stds =[]
        approaches = []
        for filename in os.listdir(args.bar_load_folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(args.bar_load_folder_path, filename)
                
                with open(file_path, "r") as file:
                    data = json.load(file)
                    test_acc = data["mean_test_accuracy"]
                    test_std = data["std_test_accuracy"]
                    train_time = data["avg_train_time"]
                    train_time_std = data["std_avg_train_time"]
                    test_accs.append(test_acc)
                    test_stds.append(test_std)
                    train_times.append(train_time)
                    train_time_stds.append(train_time_std)

                    if('vanilla' in filename):
                        approaches.append('Vanilla')
                    if('resnet32' in filename):
                        approaches.append('Resnet32')
                    if('filter' in filename and 'last' in filename):
                        approaches.append('Filter-last')
                    if('filter' in filename and 'all' in filename):
                        approaches.append('Filter-all')
                    if('feature' in filename  and 'last' in filename):
                        approaches.append('Feature-last')
                    if('feature' in filename  and 'all' in filename):
                        approaches.append('Feature-all')
                    # approaches.append(filename.replace(".json", ""))
                


        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the bar chart
        ax.bar(approaches, test_accs, yerr=test_stds, align='center', alpha=0.7, ecolor='black', capsize=10)

        # Add labels and title
        ax.set_xlabel('Approaches')
        ax.set_ylabel('Average Test Accuracy')
        # ax.set_title('Average Test Accuracy/Standard Deviation')

        # Save the plot as a PNG image
        plt.tight_layout()
        plt.savefig(args.bar_save_file_path)


        fig1, ax1 = plt.subplots()

        # Plot the bar chart
        ax1.bar(approaches, train_times, yerr=train_time_stds, align='center', alpha=0.7, ecolor='black', capsize=10)

        # Add labels and title
        ax1.set_xlabel('Approaches')
        ax1.set_ylabel('Average Training Time (Sec)')
        # ax.set_title('Average Test Accuracy/Standard Deviation')

        # Save the plot as a PNG image
        plt.tight_layout()
        plt.savefig(r'data\mean_time_accuracy_approaches_with_std.png')

    if(args.run_hypothesis_test):
        hypothesis_test={}
        test_accs = []
        approaches = []

        for filename in os.listdir(args.hypothesis_test_load_folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(args.hypothesis_test_load_folder_path, filename)
                
                with open(file_path, "r") as file:
                    data = json.load(file)
                
                test_acc = data["test_accuracies"]
                test_accs.append(test_acc)
                approaches.append(filename)

        for i in range(0,len(test_accs)):
            approach = test_accs[i]
            for j in range(0,len(test_accs)):
                reject = False
                compared_approach = test_accs[j]
                differences = np.array(approach) - np.array(compared_approach)
                if(True):
                #statistic, p_value = wilcoxon(differences)
                    statistic, p_value = mannwhitneyu(approach,compared_approach)
                    if p_value < args.alpha:
                        reject = True #result statistically significant.
                    hypothesis_test[approaches[i].replace(".json", "")+ "to" +approaches[j].replace(".json", "")]= {}
                    hypothesis_test[approaches[i].replace(".json", "")+ "to" +approaches[j].replace(".json", "")]= {}
                    hypothesis_test[approaches[i].replace(".json", "")+ "to" +approaches[j].replace(".json", "")]= {}
                    hypothesis_test[approaches[i].replace(".json", "")+ "to" +approaches[j].replace(".json", "")]= {}

                    hypothesis_test[approaches[i].replace(".json", "")+ "to" +approaches[j].replace(".json", "")]["reject"] = reject
                    hypothesis_test[approaches[i].replace(".json", "")+ "to" +approaches[j].replace(".json", "")]["differences"] = differences.tolist()
                    hypothesis_test[approaches[i].replace(".json", "")+ "to" +approaches[j].replace(".json", "")]["statistic"] = statistic
                    hypothesis_test[approaches[i].replace(".json", "")+ "to" +approaches[j].replace(".json", "")]["p_value"] = p_value



        file_helper.save_to_file(hypothesis_test,args.hypothesis_test_save_file_path)


