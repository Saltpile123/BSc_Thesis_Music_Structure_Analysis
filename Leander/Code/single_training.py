import BiLSTM, CNN
import util_dicts as ud
import train_test as tt
import paths as pa

import os, json
import numpy as np
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report


def plotmodel(model, dir, show_shapes=True):
    plot_model(model, dir, show_shapes)


def single_training(tt_dir
                    , results_dir
                    , features
                    , neurons=128
                    , activation='relu'
                    , optimizer='adam'
                    , drop_out=0.25
                    , pooling='max'
                    , plot_model=False):
    print('Creating Inpus for networks...')
    inputs_cnn = []
    inputs_lstm = []
    for feat in features:
        inputs_cnn.append(ud.inputs_cnn[feat])
        inputs_lstm.append(ud.inputs_lstm[feat])


    p = input('Features set, LSTM or CNN? ')

    if p.lower() == 'lstm':
        multi = input('multi? ').lower() == 'y'
        print('Creating LSTM...')
        model = BiLSTM.get_bilstm(inputs_lstm
                                , neurons=neurons
                                , activation=activation
                                , opt=optimizer
                                , drop_out=drop_out
                                , pooling=pooling
                                , multi_lstm=multi)
        if multi:
            name = '/lstm_architecture_multi.png'
        else:
            name = '/lstm_architecture_single.png'
        plotmodel(model, pa.plot_dir + name)

        if plot_model:
            quit()
        else:
            print('Loading features for LSTM...')
            X_train, X_test = tt.load_X_transposed(tt_dir, features)     
        
    elif p.lower() == 'cnn':
        print('Creating CNN...')
        model = CNN.get_cnn(inputs_cnn
                            , neurons=neurons
                            , activation=activation
                            , opt=optimizer
                            , drop_out=drop_out
                            , pooling=pooling)
        plotmodel(model, pa.plot_dir + '/cnn_architecture.png')

        if plot_model:
            quit()
        else:
            print('Loading features for CNN...')
            X_train, X_test = tt.load_X_stacked(tt_dir, features)
        
    else: quit()


    print('Loading labels...')
    y_train, y_test = tt.load_y(tt_dir)

    print('Starting training of model...')
    model.fit(X_train, y_train, epochs=1, batch_size=10)


    p = input('Eval or Predict? ')

    if p.lower() == 'eval':
        accuracy = model.evaluate(X_test, y_test)
        print(accuracy)

    elif p.lower() == 'predict':

        Y_pred = model.predict(X_test)
        with open(results_dir + '/pred.json', 'w') as file:
            json.dump(Y_pred.tolist(), file)

        y_test_max = np.argmax(y_test, axis=1)
        pred_max = np.argmax(Y_pred, axis=1)

        print('Confusion Matrix')
        conf = confusion_matrix(y_test_max, np.argmax(Y_pred, axis=1))
        print(conf)
        with open(results_dir + '/conf.json', 'w') as file:
            json.dump(conf.tolist(), file)

        print('Classification Report')
        report = classification_report(y_test_max, pred_max, target_names=ud.labels, output_dict=True)
        print(report['accuracy'])
        with open(results_dir + '/report.json', 'w') as file:
            json.dump(report, file)