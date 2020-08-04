import feature_extraction as fe
import label_addition as la
import train_test as tt
import util_dicts as ud
from paths import *


# Filter out songs that have no annotation
#la.list_labels(wav_dir, jams_dir)

# input()
# print('Creating Features...')
# fe.create_features(wav_dir, raw_dir)
# print('Creating Features Done.')

# input()
# print('Adding Labels to Features...')
# la.add_labels_to_feautures(raw_dir, jams_dir)
# print('Adding Labels Done.')

# input()
# print('Normalizing Features...')
# fe.normalize_features(raw_dir, norm_dir)
# print('Normalizing Features Done.')

# input()
# print('Transposing Features...')
# fe.transpose_features(norm_dir, norm_trans_dir)
# print('Transposing Features Done.')

#tt.create_train_test_labels(norm_trans_dir, reduced_train_test_dir, 0.1, 0.8)
#tt.create_train_test_labels(norm_trans_dir, half_train_test_dir, 0.5, 0.8)

for feat in ud.features:
    print(feat)
    tt.create_train_test_feature(norm_trans_dir, reduced_train_test_dir, feat)
    tt.create_train_test_feature(norm_trans_dir, half_train_test_dir, feat)