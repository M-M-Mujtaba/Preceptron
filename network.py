# Extracting the data
from pandas import read_excel
import numpy as np
import matplotlib.pyplot as plt

def Diff(li1, li2):
    return (list(set(li1) - set(li2)))

def typu(len):
    if len == 9:
        return 1
    elif len == 13:
        return 2
    else:
        return 3


def predict(yhat):
    if yhat == 1.0:
        return "setosa"
    elif yhat == 2.0:
        return "versicolor"
    elif yhat == 3.0:
        return "virginica"
    else:
        return "Unclassified"

def new_predict(yhat):

    if yhat <= 1.5:
        return "setosa"
    elif yhat>1.5 and yhat<=2.5:
        return "versicolor"
    elif yhat > 2.5:
        return "virginica"
    else:
        return "Unclassified"


# Reading data and randomising it
data_set = read_excel("iris dataset.xlsx")
data_set = data_set.sample(frac=1).reset_index(drop=True)

np_data_set = data_set.iloc[:, 0:4].to_numpy().astype(float)
np_data_set = np.c_[np.ones((len(np_data_set), 1)).astype(float), np_data_set]

# separating lables
lables = data_set.iloc[:, 4]
# assigning classes
y_list = [typu(len(lables[i])) for i in range(150)]
# vectorization
y = np.asarray(y_list)
y = np.reshape(y, (1, 150))

# initializing the constants and weights
alpha = 0.00005  # learning rate
training_sample_size = 100
W = np.random.random((1, 5))

# separating training from test samples
training_examples = np_data_set[:training_sample_size, :]
test_examples = np_data_set[training_sample_size:, :]
training_labels = y[:, :training_sample_size]
test_labels = y[:, training_sample_size:]

# intialising the control variables
max_iter = 1000
cost = [None] * ((max_iter // 100) + 1)
cost_index = 0
curr_cost = 10000000
i = 0
# Training the model
while curr_cost > 0 and i <= max_iter:
    # predict the output
    y_hat = np.matmul(W, np.transpose(training_examples))
    # computing the cost
    curr_cost = 0.5 * np.sum(np.power(np.subtract(training_labels, y_hat), 2))
    # saving the new cost
    if i % 100 == 0:
        cost[cost_index] = curr_cost
        cost_index += 1
    # updating the weights
    W = W + (alpha * np.matmul(np.subtract(training_labels, y_hat), training_examples))
    i += 1
# yes just worried idk why
plt.plot([i*100 for i in range(11)], cost, 'c')
plt.ylabel('Cost')
plt.xlabel('Iteration')
plt.show()
print(W)
print("The Final Cost is {}".format(curr_cost))

# Testing the model
y_hat = np.matmul(W, np.transpose(test_examples))
output = [predict(i) for i in y_hat[0]]
print(output)


# Testing with step function
newoutput = [new_predict(i) for i in y_hat[0]]
actual_output = [new_predict(i) for i in test_labels[0]]
print(newoutput)
print("The differnce in predicted vs acutal is {}".format(Diff(actual_output, newoutput)))