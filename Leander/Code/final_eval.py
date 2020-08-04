import paths as pa
import train_test as tt
import util_dicts as ud
import set_session as ss
import jam_data as jd

import os, json, jams
import keras.models
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd


ss.initialize_session()

#test_song_dir = pa.result_analysis_dir + '/1200.json'


def print_cm(y_test_max, kind, pred, title):
    '''
    Prints confusion matrix of the provided ground truth and predictions.
    '''
    f_size = 16

    print('\n' + kind, 'confusion matrix:')
    cm = confusion_matrix(y_test_max, pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    #plt.title(title)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ud.labels)
    ax.set_yticklabels([''] + ud.labels)
    #ax.tick_params(axis='both', which='major', labelsize=20)
    #ax.tick_params(axis='both', which='minor', labelsize=20)
    plt.xlabel('Predicted', fontsize=f_size)
    plt.ylabel('True', fontsize=f_size)
    plt.xticks(rotation=45, fontsize=f_size)
    plt.yticks(rotation=45, fontsize=f_size)
    print(cm)
    plt.show()


def print_report(y_test_max, kind, pred):
    '''
    Prints classification report of the provided ground truth and predictions.
    '''
    print('\n' + kind, 'classification report:')
    print(classification_report(y_test_max, pred, labels=ud.labs, target_names=ud.labels, output_dict=False))

#plt.show()

# print('True\tPredicted')
# for i in range(y_test_max.shape[0]):
#     print(ud.labels[y_test_max[i]] + '\t' + ud.labels[pred_cnn_max[i]])

def filter_predictions(pred, min_length = 5):
    filtered = [pred[0]]
    for i in range(1,pred.shape[0] - min_length):
        if pred[i] != filtered[i-1]:
            # counter = {}
            # for j in pred[i+1:i+min_length]:
            #     if j in counter: counter[j] += 1
            #     else: counter[j] = 1
            # filtered.append(sorted(counter, key = counter.get, reverse=True)[0])

            if filtered[i-1] in pred[i+1:i+min_length]:
                filtered.append(filtered[i-1])
            else: filtered.append(pred[i])

        else: filtered.append(pred[i])

    for i in range(min_length):
        filtered.append(filtered[-1])

    filtered = np.array(filtered)
    return filtered


def print_items(y_test_max, pred, pred_filtered):
    print('True\tPred\tFilter')
    for i in range(pred.shape[0]):
        print(ud.short_labels[y_test_max[i]] + '\t' + ud.short_labels[pred[i]] + '\t' + ud.short_labels[pred_filtered[i]])


def pred_to_jam(beats, pred):
    time = 0.0
    label = ud.labels[pred[0]]
    observations = []
    
    n = len(pred)
    for i in range(0,n):
        if ud.labels[pred[i]] == label: continue
        else:
            new_time = beats[i]
            new_label = ud.labels[pred[i]]

            observations.append(jams.Observation(time, new_time-time, label, None))

            time = new_time
            label = new_label
    observations.append(jams.Observation(time, beats[-1]-time, ud.labels[pred[-1]], None))

    annotation = jams.Annotation("segment_open"
                                , observations
                                , jd.get_annotation_metadata()
                                , jd.get_sandbox())
    jam = jams.JAMS([annotation], jams.FileMetadata(duration=beats[-1]))
    return jam


def save_as_csv(beats, y_test_max, pred, pred_filtered, dir):
    t_labs = [ud.labels[x] for x in y_test_max]
    p_labs = [ud.labels[x] for x in pred]
    f_labs = [ud.labels[x] for x in pred_filtered]
    n = min([len(beats),len(t_labs),len(p_labs),len(f_labs)])
    data = {
        'Beat Time' : beats[:n],
        'True Label' : t_labs[:n],
        'Predicted Label' : p_labs[:n],
        'Filtered Label' : f_labs[:n]
    }
    df = pd.DataFrame(data)
    df.index.name = 'Beat'
    #print(df)
    df.to_csv(dir)


def run_cnn(final_cnn, y_test, X_stacked, beats, res_dir, filename):
    pred_cnn = final_cnn.predict(X_stacked)
    pred_cnn_max = np.argmax(pred_cnn, axis=1)

    #print_cm(y_test, 'cnn', pred_cnn_max)
    #print_report(y_test, 'cnn', pred_cnn_max)

    pred_cnn_filtered = filter_predictions(pred_cnn_max)

    save_as_csv(beats, y_test, pred_cnn_max, pred_cnn_filtered, res_dir + '/csv/' + filename + '_cnn.csv')

    jam = pred_to_jam(beats, pred_cnn_filtered)
    jam_path = res_dir + '/cnn/SALAMI_' + filename + '.jams'
    jam.save(jam_path)

    # if input('Save predictions? ').lower() == 'y':
    #     save_as_csv(beats, y_test, pred_cnn_max, pred_cnn_filtered, res_dir + '/' + filename + '_cnn.csv')

    #     jam = pred_to_jam(beats, pred_cnn_filtered)
    #     jam_path = res_dir + '/' + filename + '_cnn.jams'
    #     jam.save(jam_path)
    # else:
    #     print_items(y_test, pred_cnn_max, pred_cnn_filtered)


def run_lstm(final_lstm, y_test, X_transposed, beats, res_dir, filename):
    pred_lstm = final_lstm.predict(X_transposed)
    pred_lstm_max = np.argmax(pred_lstm, axis=1)

    #print_cm(y_test, 'lstm', pred_lstm_max)
    #print_report(y_test, 'lstm', pred_lstm_max)

    pred_lstm_filtered = filter_predictions(pred_lstm_max)

    save_as_csv(beats, y_test, pred_lstm_max, pred_lstm_filtered, res_dir + '/csv/' + filename + '_lstm.csv')

    jam = pred_to_jam(beats, pred_lstm_filtered)
    jam_path = res_dir + '/lstm/SALAMI_' + filename + '.jams'
    jam.save(jam_path)


    # if input('Save predictions? ').lower() == 'y':
    #     save_as_csv(beats, y_test, pred_lstm_max, pred_lstm_filtered, res_dir + '/' + filename + '_lstm.csv')

    #     jam = pred_to_jam(beats, pred_lstm_filtered)
    #     jam_path = res_dir + '/' + filename + '_lstm.jams'
    #     jam.save(jam_path)
    # else:
    #     print_items(y_test, pred_lstm_max, pred_lstm_filtered)
    

# kind = input("CNN or LSTM? ").lower()
# if kind == 'cnn':
#     run_cnn()
# elif kind == 'lstm':
#     run_lstm()
# elif kind == '1200':
#     jam = pred_to_jam(y_test_max)
#     jam.save(pa.result_analysis_dir + '/1200_gt.jams')
# else:
#     print('Whatever...')

features = ['cqt', 'tempogram']
test_dir = pa.final_results_dir

data_dir = pa.train_test_dir

y_train, y_test = tt.load_y(data_dir)
y_test_max = np.argmax(y_test, axis=1)

# inputs_cnn = [ud.inputs_cnn[feat] for feat in features]
# X_test_cnn = tt.load_X_stacked_test(data_dir, features)
# final_cnn = keras.models.load_model(pa.models_dir + '/cnn_relu_adam_max.h5')
# pred_cnn = final_cnn.predict(X_test_cnn)
# print_cm(y_test_max, 'cnn', np.argmax(pred_cnn, axis=1), 'Confusion Matrix of best CNN model evaluated on full testset')

inputs_lstm = [ud.inputs_lstm[feat] for feat in features]
X_test_lstm = tt.load_X_transposed_test(data_dir, features)
final_lstm = keras.models.load_model(pa.models_dir + '/bilstm_128_relu_adam_25_max.h5')
pred_lstm = final_lstm.predict(X_test_lstm)
print_cm(y_test_max, 'lstm', np.argmax(pred_lstm, axis=1), 'Confusion Matrix of best LSTM model evaluated on full testset')


# for song in os.listdir(pa.norm_trans_dir):
#     song_name = song.split('.')[0]
#     print(song_name)
#     test_song_dir = test_dir + '/' + song

#     y_test, X_stacked, X_transposed, beats = tt.prepare_single_song(pa.norm_trans_dir + '/' + song, features)
#     y_test_max = np.argmax(y_test, axis=1)


#     #run_cnn(final_cnn, y_test_max, X_stacked, beats, test_dir, song_name)
#     #run_lstm(final_lstm, y_test_max, X_transposed, beats, test_dir, song_name)

#     y_jam = pred_to_jam(beats, y_test_max)
#     y_jam.save(pa.final_results_dir + '/gt_jams/SALAMI_' + song_name + '.jams')

# y_test, _, _, beats = tt.prepare_single_song(pa.norm_trans_dir + '/1208.json', features)
# jam = pred_to_jam(beats, np.argmax(y_test, axis=1))
# jam.save(pa.final_results_dir + '/gt_jams/SALAM_1208.jams')