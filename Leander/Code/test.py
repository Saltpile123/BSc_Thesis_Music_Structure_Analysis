import os,json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import paths as pa
import train_test as tt
import util_dicts as ud

# with open(pa.train_test_dir + '/pred.json', 'r') as file:
#     pred = np.array(json.load(file))
# _, y_test = tt.load_y(pa.train_test_dir)

# print(pred.shape)
# pred_max = np.argmax(pred, axis=1)
# print(pred_max.shape)

# print('')

# print(y_test.shape)
# y_test_max = np.argmax(y_test, axis=1)
# print(y_test_max.shape)

# print('')

# print(confusion_matrix(y_test_max, pred_max))

# print('')

# print(classification_report(y_test_max, pred_max, target_names=ud.labels))

train = pa.reduced_train_test_dir + '/y_train.json'
test = pa.reduced_train_test_dir + '/y_test.json'

with open(train) as tr:
    y_train = np.array(json.load(tr))
with open(test) as te:
    y_test = np.array(json.load(te))

print(y_train.shape)
print(y_test.shape)

for i in range (0,9):
    print('Column', i)
    print(np.unique(y_train[:,i]))
    print(np.unique(y_test[:,i]))