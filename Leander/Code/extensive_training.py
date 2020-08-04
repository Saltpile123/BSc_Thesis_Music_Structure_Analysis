import util_dicts as ud
import train_test as tt
import paths as pa
import BiLSTM, CNN

import os, json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def extensive_lstm(tt_dir, results_dir, params):
    y_train, y_test = tt.load_y(tt_dir)
    y_test_max = np.argmax(y_test, axis=1)
 
    for neurons in params['neurons']:
        for activation in params['activation']:
            for optimizer in params['optimizer']:
                for drop_out in params['drop_out']:
                    for pooling in params['pooling']:
                        for epoch in params['epochs']:
                            for batch_size in params['batch_size']:               
                                for feature_combi in params['features']:

                                    X_train, X_test = tt.load_X_transposed(tt_dir, feature_combi)

                                    for multi in [False, True]:
                                        model_name = 'lstm_' \
                                            + str(neurons) + '_' \
                                            + activation + '_' \
                                            + optimizer + '_' \
                                            + str(drop_out) + '_' \
                                            + pooling + '_' \
                                            + str(epoch) + '_' \
                                            + str(batch_size) + '_' \
                                            + str(multi) + '_' \
                                            + '_'.join(feature_combi)

                                        print('Training:', model_name)

                                        inputs = []
                                        for feat in feature_combi: inputs.append(ud.inputs_lstm[feat])     

                                        model = BiLSTM.get_bilstm(inputs
                                                                , neurons = neurons
                                                                , activation = activation
                                                                , opt = optimizer
                                                                , drop_out = drop_out
                                                                , pooling = pooling
                                                                , multi_lstm = multi)

                                        model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size)

                                        predict_and_save_model(model, model_name, X_test, y_test_max, results_dir)


                                        
def extensive_cnn(tt_dir, results_dir, params): 
    y_train, y_test = tt.load_y(tt_dir)
    y_test_max = np.argmax(y_test, axis=1)

    X_train, X_test = tt.load_X_stacked(tt_dir, ud.features)

    for neurons in params['neurons']:
        for activation in params['activation']:
            for optimizer in params['optimizer']:
                for drop_out in params['drop_out']:
                    if neurons == 256 and drop_out != 0.5:
                        continue
                    for pooling in params['pooling']:
                        for epoch in params['epochs']:
                            for batch_size in params['batch_size']:                      
                                for feature_combi in params['features']:
                                    model_name = 'cnn_' \
                                        + str(neurons) + '_' \
                                        + activation + '_' \
                                        + optimizer + '_' \
                                        + str(drop_out) + '_' \
                                        + pooling + '_' \
                                        + str(epoch) + '_' \
                                        + str(batch_size) + '_' \
                                        + '_'.join(feature_combi)

                                    print('Training:', model_name)

                                    inputs = []
                                    for feat in feature_combi: inputs.append(ud.inputs_cnn[feat])

                                    #X_train, X_test = tt.load_X_stacked(tt_dir, feature_combi)

                                    model = CNN.get_cnn(inputs
                                                        , neurons = neurons
                                                        , activation = activation
                                                        , opt = optimizer
                                                        , drop_out = drop_out
                                                        , pooling = pooling)

                                    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size)

                                    predict_and_save_model(model, model_name, X_test, y_test_max, results_dir)



def predict_and_save_model(model, model_name, X_test, y_test_max, results_dir):
    Y_pred = model.predict(X_test)                       
    pred_max = np.argmax(Y_pred, axis=1)

    # with open(results_dir + '/' + model_name + '_pred.json', 'w') as file:
    #     json.dump(Y_pred.tolist(), file)

    with open(results_dir + '/' + model_name + '_conf_matrix.json', 'w') as file:
        json.dump(confusion_matrix(y_test_max, pred_max).tolist(), file)

    with open(results_dir + '/' + model_name + '_class_report.json', 'w') as file:
        report = classification_report(y_test_max, pred_max, labels=ud.labs, target_names=ud.labels, output_dict=True)
        for key in report:
            print(key, '\t', report[key])
        json.dump(report, file)

