from net import Net
import csv
from numpy import transpose

def read_dataset_csv(filepath): 
    input_data = []
    output_data = []
    with open(filepath, newline='') as csvfile:
        auxiliar_list = []
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            for i in range(len(row)):   
                if i == len(row)-1:
                    output_data.append(list(map(int,row[i])))
                else:
                    auxiliar_list.append(row[i])
            auxiliar_list = list(map(float, auxiliar_list))
            input_data.append(auxiliar_list)
            auxiliar_list = []
        
    return dict(input_data = input_data, output_data = output_data) 


# Find the min and max values for each column
def dataset_minmax(data):
	minmax = list()
	stats = [[min(row), max(row)] for row in transpose(data)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_data(data, minmax):
	for row in data:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


input_train, output_train = read_dataset_csv("./net_info/seeds_train.csv").values()
input_validation, output_validation = read_dataset_csv("./net_info/seeds_validation.csv").values()

normalize_data(input_train, dataset_minmax(input_train))
normalize_data(input_validation, dataset_minmax(input_validation))

rede = Net([7,4,1], [[1,1,1,1,1,1,1], [1,1,1,1], [1]])

training_data = rede.training(input_train, output_train, input_validation, output_validation)

print(training_data["weights"])
results = []
for i in range(len(input_validation)):
    results.append(rede.net_run(input_validation[i], output_validation[i])[0][0])
