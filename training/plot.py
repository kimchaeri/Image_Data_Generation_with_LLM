import re
import os
import copy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy(results):
    file_names = list(results.keys())
    label, color = None, None

    plt.figure(figsize=(8, 6))
    plt.title('Accuracy for each Epoch)')
    
    yticks = np.arange(0, 101, 20)
    plt.yticks(yticks)
    plt.ylim(0, 100)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


    for file_name in file_names:   
        result = results[file_name]
        epochs = list(result.keys())
        accuracy_values = list(result.values())
        if "generated" in file_name:
            label = "Generated"
            color = "blue"
        else:
            label = "Origin"
            color = "green"
        accuracy_values_float = [float(x) for x in accuracy_values]
        plt.plot(epochs, accuracy_values_float, linestyle='-', label=label, color=color)
    plt.legend()
    plt.show()



def extract_accuracy_from_txt(txt_path):
    file_path = txt_path
    mode = 'r'
    accuracy_for_epochs = dict()
    
    pattern = r"Accuracy of the network on test set at epoch \d+: (\d+\.\d+) %"
    epoch = 1
    
    with open(file_path, mode) as file:
        lines = file.read().split('\n')
        for line in lines:
            matches = re.findall(pattern, line)
            for match in matches:
                accuracy_for_epochs[epoch] = match
                epoch += 1
                
    return accuracy_for_epochs


if __name__ == "__main__":

    results = dict()
    result_path = "/home/qowodus/바탕화면/test/results"
    
    for file_name in os.listdir(result_path):
        txt_path = os.path.join(result_path, file_name)
        accuracy_for_epochs = extract_accuracy_from_txt(txt_path)
        results[file_name] = copy.deepcopy(accuracy_for_epochs)

    plot_accuracy(results)

