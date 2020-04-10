# Extracting the data
from pandas import read_excel
import numpy as np
import matplotlib.pyplot as plt


def typu(len):
    if len == 9:
        return 1
    elif len == 13:
        return 2
    else:
        return 3


data_set = read_excel("iris dataset.xlsx")
data_set = data_set.sample(frac=1).reset_index(drop=True)
np_data_set = data_set.iloc[:, 0:4].to_numpy().astype(float)
np_data_set = np.c_[np.ones((len(np_data_set), 1)).astype(float), np_data_set]
lables = data_set.iloc[:, 4]
y_list = [typu(len(lables[i])) for i in range(150)]
y = np.asarray(y_list)
y = np.reshape(y, (1, 150))

# initializing the weights
epsilon = 0.12
alpha = 0.00005
training_sample_size = 75

W = np.random.random((1, 5))


training_examples = np_data_set[:75, :]
test_examples = np_data_set[75:, :]
training_labels = y[:, :75]
test_labels = y[:, :75]
max_iter = 10000
cost = [None] * ((max_iter // 100) + 1)
m = 75
cost_index = 0
# Learning the weightes
print(np.transpose(training_examples))
for i in range(max_iter + 1):

plt.plot(cost, 'c')  # ² 3Lg(N)*N/)4
# plt.legend([line_up, result_line, line_down], ['Upper Bound 3NLg(N)', ' Result', 'Lower Bound NLg(N)/2'])
# plt.ylabel('Step Count')
# plt.xlabel("array sizes")
plt.show()
print(W)
print(J)
