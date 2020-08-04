import librosa as lb
import librosa.display
import numpy as np
import json
import jams
import os

import matplotlib.pyplot as plt
import util_dicts as ud

# mp3_dir = "C:/help_me_pls/_SCRIPTIE/bachelor-scriptie-musical-sctructure-analysis/Database/music"
# wav_dir = "C:/help_me_pls/_SCRIPTIE/wav"
# data_dir = "C:/help_me_pls/_SCRIPTIE/data/75_overlap"
# jams_dir = "C:/help_me_pls/_SCRIPTIE/bachelor-scriptie-musical-sctructure-analysis/Database/salami jams"

def convert_to_wav(mp3_dir, dest_dir):
    for filename in os.listdir(mp3_dir):
        file = mp3_dir + '/' + filename
        export = dest_dir + '/' + filename.split('.')[0] + '.wav'
        os.system(f"""ffmpeg -i {file} -acodec pcm_u8 -ar 22050 {export}""")


def get_delta(data):
    return lb.feature.delta(data)


def visualize_data(data : list):
    plt.figure()
    total = len(data)
    for i in range(total):
        plt.subplot(total,1,i+1)
        lb.display.specshow(data[i], x_axis='time')
        plt.colorbar()
        plt.title('Data ' + str(i))
    plt.tight_layout()
    plt.show()


def show_tempograms(y, sr):
    oenv = lb.onset.onset_strength(y=y, sr=sr, hop_length=512)
    fourier_tempogram = lb.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=512)
    tempogram = lb.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=512)
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(oenv, label='Onset strength')
    plt.xticks([])
    plt.legend(frameon=True)
    plt.axis('tight')
    
    plt.subplot(3,1,2)
    lb.display.specshow(np.abs(fourier_tempogram), sr=sr, hop_length=512, x_axis='time', y_axis='fourier_tempo', cmap='magma')
    plt.title('Fourier Tempogram')

    plt.subplot(3,1,3)
    lb.display.specshow(tempogram, sr=sr, hop_length=512, x_axis='time', y_axis='tempo', cmap='magma')
    plt.title('Tempogram')
    plt.tight_layout()
    plt.show()


def get_features(file, fft=4096, hop=1024, ref=np.max, norm=np.inf):
    print(file)
    features = {}

    y, sr = lb.load(file)
    # print('y:', y.shape)
    # print('sr:', sr)

    tempo, beats = lb.beat.beat_track(y=y, sr=sr, trim=False, hop_length=hop)
    beat_track = {'bpm' : tempo, 'beats' : beats.tolist()}

    lin_cqt = np.abs(lb.cqt(y=y, sr=sr, hop_length=hop, norm=norm)) ** 2
    cqt = lb.amplitude_to_db(lin_cqt, ref=ref)
    features['cqt'] = lb.util.sync(cqt, beats).tolist()

    lin_cens = np.abs(lb.feature.chroma_cens(y=y, sr=sr, hop_length=hop)) ** 2
    cens = lb.amplitude_to_db(lin_cens, ref=ref)
    features['cens'] = lb.util.sync(cens, beats).tolist()

    harmony, _ = lb.effects.hpss(y=y)
    pcp_cqt = np.abs(lb.hybrid_cqt(harmony, sr=sr, hop_length=hop, norm=norm, fmin=27.5)) ** 2
    pcp = lb.feature.chroma_cqt(C=pcp_cqt, sr=sr, hop_length=hop, n_octaves=6, fmin=27.5)
    features['pcp'] = lb.util.sync(pcp, beats).tolist()

    tonnetz = lb.feature.tonnetz(chroma=pcp)
    features['tonnetz'] = lb.util.sync(tonnetz, beats).tolist()

    mel = lb.feature.melspectrogram(y=y, sr=sr, n_fft=fft, hop_length=hop)
    log_mel = lb.amplitude_to_db(mel, ref=ref)
    mfcc = lb.feature.mfcc(S=log_mel, n_mfcc=14)
    features['mfcc'] = lb.util.sync(mfcc, beats).tolist()

    tempogram = lb.feature.tempogram(y=y, sr=sr, hop_length=hop, win_length=192)
    features['tempogram'] = lb.util.sync(tempogram, beats).tolist()

    return {'beat_track' : beat_track, 'features' : features}


def create_features(audio_dir, output_dir):
    for song in os.listdir(audio_dir):
        features = get_features(audio_dir + '/' + song)
        with open(output_dir + '/' + song.split('.')[0] + '.json', mode='w') as song_feat:
            json.dump(features, song_feat) 


def normalize_features(input_dir, output_dir):
    for data in os.listdir(input_dir):
        print(data)
        with open(input_dir + '/' + data, 'r') as song_data:
            song = json.load(song_data)

        for feat in ud.features:
            np_feat = np.array(song['features'][feat])
            song['features'][feat] = ((np_feat - np.min(np_feat)) / np.ptp(np_feat)).tolist()

        with open(output_dir + '/' + data, 'w') as song_data:
            json.dump(song, song_data)


def transpose_features(in_dir, out_dir):
    for data in os.listdir(in_dir):
        print(data)
        with open(in_dir + '/' + data, 'r') as song_data:
            song = json.load(song_data)
        
        for feat in ud.features:
            np_feat = np.array(song['features'][feat])
            song['features'][feat] = np_feat.T.tolist()

        with open(out_dir + '/' + data, 'w') as song_data:
            json.dump(song, song_data)