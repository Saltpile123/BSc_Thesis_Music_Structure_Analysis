import util_dicts as ud
from paths import *
import jams

import os, json
import numpy as np
from keras.utils import to_categorical


def create_train_test_labels(feat_dir, out_dir, subset_size, train_test_split):
    np_labels_one = []
    
    for data in os.listdir(feat_dir):
        print(data)
        with open(feat_dir + '/' + data, 'r') as song_data:
            song = json.load(song_data)
        
        l = to_categorical(np.array([ud.label_ints[label] for label in song['labels']]).T,9)
        if len(np_labels_one) == 0: np_labels_one = l
        else: np_labels_one = np.concatenate((np_labels_one, l))

    count = np_labels_one.shape[0]
    print('count:', count)
    input()
    indices = np.random.permutation(count)
    subset_count = int(subset_size*count)
    reduced_indices = indices[:subset_count]
    num = int(train_test_split*len(reduced_indices))
    ind_train, ind_test = reduced_indices[:num], reduced_indices[num:]

    with open(out_dir + '/ind_train.json', 'w') as file:
        json.dump(ind_train.tolist(), file)
    with open(out_dir + '/ind_test.json', 'w') as file:
        json.dump(ind_test.tolist(), file)

    with open(out_dir + '/y_train.json', 'w') as file:
        json.dump(np_labels_one[ind_train,:].tolist(), file)
    with open(out_dir + '/y_test.json', 'w') as file:
        json.dump(np_labels_one[ind_test,:].tolist(), file)


def create_train_test_feature(feat_dir, out_dir, feat):
    np_feat = []
    np_feat_stacked = []
    
    for data in os.listdir(feat_dir):
        print(data)
        with open(feat_dir + '/' + data, 'r') as song_data:
            song = json.load(song_data)
        
        label_len = len(song['labels'])
        
        f = np.array(song['features'][feat])[:label_len,:]
        x = np.stack((np.roll(f,-2,0), np.roll(f,-1,0), f, np.roll(f,1,0)), -1)

        if len(np_feat) > 0:
            np_feat = np.concatenate((np_feat, x))
            np_feat_stacked = np.concatenate((np_feat_stacked, np.expand_dims(x,-1)))
        else:
            np_feat = x
            np_feat_stacked = np.expand_dims(x,-1)

    ind_train, ind_test = load_indices(out_dir)

    with open(out_dir + '/' + feat + '_train.json', 'w') as file:
        json.dump((np_feat[ind_train,:,:]).tolist(), file)
    with open(out_dir + '/' + feat + '_test.json', 'w') as file:
        json.dump((np_feat[ind_test,:,:]).tolist(), file)    

    with open(out_dir + '/' + feat + '_train_stacked.json', 'w') as file:
        json.dump((np_feat_stacked[ind_train,:,:]).tolist(), file)
    with open(out_dir + '/' + feat + '_test_stacked.json', 'w') as file:
        json.dump((np_feat_stacked[ind_test,:,:]).tolist(), file)   


def load_indices(dir):
    with open(dir + '/ind_train.json', 'r') as ind:
        ind_train = json.load(ind)
    with open(dir + '/ind_test.json', 'r') as ind:
        ind_test = json.load(ind)
    return ind_train, ind_test


def load_X(dir, features):
    X_train = {}
    X_test = {}
    for feat in features:
        print('Loading', feat + '...')
        with open(dir + '/' + feat + '_train.json', 'r') as x:
            X_train[feat] = np.array(json.load(x))
        with open(dir + '/' + feat + '_test.json', 'r') as x:
            X_test[feat] = np.array(json.load(x))
    return X_train, X_test

def load_X_test(dir, features):
    X_test = {}
    for feat in features:
        print('Loading', feat + '...')
        with open(dir + '/' + feat + '_test.json', 'r') as x:
            X_test[feat] = np.array(json.load(x))
    return X_test


def load_X_transposed(dir, features):
    X_train = {}
    X_test = {}
    for feat in features:
        print('Loading', feat + '...')
        with open(dir + '/' + feat + '_train.json', 'r') as x:
            X_train[feat] = np.transpose(np.array(json.load(x)), (0,2,1))
        with open(dir + '/' + feat + '_test.json', 'r') as x:
            X_test[feat] = np.transpose(np.array(json.load(x)), (0,2,1))
    return X_train, X_test

def load_X_transposed_test(dir, features):
    X_test = {}
    for feat in features:
        print('Loading', feat + '...')
        with open(dir + '/' + feat + '_test.json', 'r') as x:
            X_test[feat] = np.transpose(np.array(json.load(x)), (0,2,1))
    return X_test


def load_X_stacked(dir, features):
    X_train_stacked = {}
    X_test_stacked = {}
    for feat in features:
        print('Loading', feat + '...')
        with open(dir + '/' + feat + '_train_stacked.json', 'r') as x:
            X_train_stacked[feat] = np.array(json.load(x))
        with open(dir + '/' + feat + '_test_stacked.json', 'r') as x:
            X_test_stacked[feat] = np.array(json.load(x))
    return X_train_stacked, X_test_stacked

def load_X_stacked_test(dir, features):
    X_test_stacked = {}
    for feat in features:
        print('Loading', feat + '...')
        with open(dir + '/' + feat + '_test_stacked.json', 'r') as x:
            X_test_stacked[feat] = np.array(json.load(x))
    return X_test_stacked


def load_y(dir):
    with open(dir + '/y_train.json', 'r') as ytr:
        y_train = np.array(json.load(ytr))
    with open(dir + '/y_test.json', 'r') as yte:
        y_test = np.array(json.load(yte))
    return y_train, y_test


def prepare_single_song(in_dir, features):
    with open(in_dir, 'r') as jsong:
        song = json.load(jsong)

    y_test = to_categorical(np.array([ud.label_ints[label] for label in song['labels']]).T,9)

    X_transposed = {}
    X_stacked = {}
    for feat in features:
        f = np.array(song['features'][feat])[:len(song['labels']),:]
        X = np.stack((np.roll(f,-2,0), np.roll(f,-1,0), f, np.roll(f,1,0)), -1)
        
        X_stacked[feat] = np.expand_dims(X,-1)
        X_transposed[feat] = np.transpose(X, (0,2,1))

    beats = []
    for beat in song['beat_track']['beats']:
        beats.append(beat * 1024 / 22050)

    return y_test, X_stacked, X_transposed, beats