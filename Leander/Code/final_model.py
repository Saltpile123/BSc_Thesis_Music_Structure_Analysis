import CNN, BiLSTM

import util_dicts as ud
import train_test as tt
import paths as pa

import os, json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


features = ['cqt', 'tempogram']
data_dir = pa.train_test_dir

inputs_cnn = [ud.inputs_cnn[feat] for feat in features]
inputs_lstm = [ud.inputs_lstm[feat] for feat in features]

y_train, y_test = tt.load_y(data_dir)
y_test_max = np.argmax(y_test, axis=1)


def final_cnn(inputs_cnn):
    print("Final CNN...")
    X_train_cnn, X_test_cnn = tt.load_X_stacked(data_dir, features)

    cnn_relu_adam_max = CNN.get_cnn(inputs_cnn
                                    , neurons = 128
                                    , activation = 'relu'
                                    , opt = 'adam'
                                    , drop_out = 0.25
                                    , pooling = 'max')

    cnn_relu_adam_max.fit(X_train_cnn, y_train, epochs=100, batch_size=100)

    Y_pred_cnn = cnn_relu_adam_max.predict(X_test_cnn)
    pred_cnn_max = np.argmax(Y_pred_cnn, axis=1)
    
    print('CNN classification report:')
    print(classification_report(y_test_max, pred_cnn_max, labels=ud.labs, target_names=ud.labels, output_dict=False))

    if (input('Save CNN? ').lower() == 'y'):
        cnn_relu_adam_max.save(pa.models_dir + '/cnn_relu_adam_max.h5')


def final_lstm(inputs_lstm):
    print('Final LSTM...')
    X_train_lstm, X_test_lstm = tt.load_X_transposed(data_dir, features)

    bilstm_128_relu_adam_25_max = BiLSTM.get_bilstm(inputs_lstm
                                                    , neurons = 128
                                                    , activation = 'relu'
                                                    , opt = 'adam'
                                                    , drop_out = 0.25
                                                    , pooling = 'max'
                                                    , multi_lstm = False)

    bilstm_128_relu_adam_25_max.fit(X_train_lstm, y_train, epochs=100, batch_size=100)

    Y_pred_lstm = bilstm_128_relu_adam_25_max.predict(X_test_lstm)
    pred_lstm_max = np.argmax(Y_pred_lstm, axis=1)

    print('LSTM classification report:')
    print(classification_report(y_test_max, pred_lstm_max, labels=ud.labs, target_names=ud.labels, output_dict=False))

    if (input('Save LSTM? ').lower() == 'y'):
        bilstm_128_relu_adam_25_max.save(pa.models_dir + '/bilstm_128_relu_adam_25_max.h5')


kind = input('CNN or LSTM? ').lower()
if kind == 'cnn':
    final_cnn(inputs_cnn)
elif kind == 'lstm':
    final_lstm(inputs_lstm)
else:
    print('Whatever...')