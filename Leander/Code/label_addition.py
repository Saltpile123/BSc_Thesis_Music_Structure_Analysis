import os, json, jams
import util_dicts as ud
import numpy as np
import paths as pa

# data_dir = "C:/help_me_pls/_SCRIPTIE/data/75_overlap"
# jams_dir = "C:/help_me_pls/_SCRIPTIE/bachelor-scriptie-musical-sctructure-analysis/Database/salami jams"
# test_jam = jams.load(jams_dir + "/SALAMI_1000.jams")

def list_labels(data_dir, jams_dir):
    labels = {}
    not_existing = [] # Filter out songs that not have annotations
    for song in os.listdir(data_dir):
        jam_path = jams_dir + '/SALAMI_' + song.split('.')[0] + '.jams'
        if os.path.exists(jam_path):
            jam = jams.load(jam_path)
            # jam is jams file
            for annotation in jam['annotations']:
                # annotation is list of different annotations
                if annotation['namespace'] == "segment_salami_function":
                    # check if annotation is a segment function annotation
                    for obs in annotation['data']:
                        # obs is item in annotation data (which is a list of segments)
                        # of type jams.Observation
                        labels[obs.value] = labels.get(obs.value, 0) + 1
                    break
        else:
            not_existing.append(song)

    for label in labels:
        print(label, "-", labels[label])
    print('')
    for non_existing in not_existing:
        print(non_existing)
    if input("Remove " + str(len(not_existing)) + " songs?") == "Y":
        for song in not_existing:
            os.remove(data_dir + "/" + song)


def add_labels_to_feautures(data_dir, jams_dir):
    for data in os.listdir(data_dir):
        print(data)
        with open(data_dir + "/" + data, mode='r') as song_data:
            song = json.load(song_data)

        beats = np.array(song['beat_track']['beats'])

        jam_path = jams_dir + '/SALAMI_' + data.split('.')[0] + '.jams'
        jam = jams.load(jam_path)
        annotation = [ann for ann in jam['annotations'] if ann['namespace'] == "segment_salami_function"][0]

        labels = []
        ind = len(beats) - 1
        #print(ind)
        for obs in reversed(annotation['data']):
            #print('obs.time', obs.time)
            #print('beats[ind]', beats[ind] * 512 / 22050)
            while ind >= 0 and beats[ind] * 512 / 22050 > obs.time:
                labels = np.append(labels, ud.label_grouping[obs.value])
                ind = ind - 1
        #print(len(beats))
        #print(len(labels))
        #print(labels)

        song['labels'] = np.flip(labels).tolist()
        with open(data_dir + "/" + data, mode='w') as song_data:
            json.dump(song, song_data)


def unique_labels(data_dir):
    labels = {}
    for data in os.listdir(data_dir):
        with open(data_dir + "/" + data, 'r') as song_data:
            song = json.load(song_data)
        for label in song['labels']:
            labels[label] = labels.get(label, 0) + 1
    for key in labels:
        print(key, ': ', labels[key])


#test_dir = "C:/help_me_pls/_SCRIPTIE/data/test"
#list_labels(data_dir, jams_dir)
#add_labels_to_feautures(test_dir, jams_dir)
#unique_labels(test_dir)

#add_labels_to_feautures(data_dir, jams_dir)
#unique_labels(pa.norm_trans_dir)
  