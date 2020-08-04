import single_training as st
import extensive_training as et
import paths as pa
import util_dicts as ud
from set_session import initialize_session


initialize_session()


#tt_dir = pa.train_test_dir
#results_dir = pa.results_dir

tt_dir = pa.reduced_train_test_dir
results_dir = pa.reduced_results_dir

#tt_dir = pa.half_train_test_dir
#results_dir = pa.half_results_dir

#params = ud.learning_params
params = ud.learning_params_new

features = ['cqt', 'mfcc', 'tempogram']
#features = ['cqt', 'cens', 'pcp', 'mfcc', 'tempogram']


training = input('single or extensive training? ').lower()

if training == 'single':
    plot = input('Plot models? ').lower() == 'y'
    st.single_training(tt_dir
                    , pa.reduced_train_test_dir
                    , features
                    , 128
                    , 'relu'
                    , 'adam'
                    , 0.1
                    , 'max'
                    , plot_model=plot)


elif training == 'extensive':
    model = input('LSTM or CNN? ').lower()

    if model == 'lstm':
        et.extensive_lstm(tt_dir, results_dir, params)

    elif model == 'cnn':
        et.extensive_cnn(tt_dir, results_dir, params)
