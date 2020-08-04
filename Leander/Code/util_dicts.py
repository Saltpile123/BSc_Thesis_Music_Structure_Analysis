'''
This module includes handy dictionaries and lists/arrays.
'''
features = ['cqt', 'cens', 'pcp', 'tonnetz', 'mfcc', 'tempogram']
labs = [0,1,2,3,4,5,6,7,8]
labels = ['silence', 'no_function', 'intro', 'verse', 'interlude', 'solo', 'chorus', 'bridge', 'outro']
short_labels = ['silence', 'no_fun', 'intro', 'verse', 'inter', 'solo', 'chorus', 'bridge', 'outro']

label_grouping = {
    "silence" : "silence",

    "no_function" : "no_function",
    "spoken" : "no_function",
    "stage_sounds" : "no_function",
    "crowd_sounds" : "no_function",
    "applause" : "no_function",

    "intro" : "intro",
    "head" : "intro",

    "verse" : "verse",
    "voice" : "verse",
    "pre-verse" : "verse",

    "interlude" : "interlude",
    "transition" : "interlude",
    "break" : "interlude",

    "solo" : "solo",
    "instrumental" : "solo",
    "theme" : "solo",
    "main theme" : "solo",

    "chorus" : "chorus",
    "pre-chorus" : "chorus",
    "post-chorus" : "chorus",

    "bridge" : "bridge",
    "build" : "bridge",

    "outro" : "outro",
    "fade-out" : "outro",
    "coda" : "outro"
}

label_ints = {
    'silence' : 0,
    'no_function' : 1,
    'intro' : 2,
    'verse' : 3,
    'interlude' : 4,
    'solo' : 5,
    'chorus' : 6,
    'bridge' : 7,
    'outro' : 8
}

inputs_cnn = {
    'cqt' : ((84,4,1), 'cqt'),
    'cens' : ((12,4,1), 'cens'),
    'pcp' : ((12,4,1), 'pcp'),
    'tonnetz' : ((6,4,1), 'tonnetz'),
    'mfcc' : ((14,4,1), 'mfcc'),
    'tempogram' : ((192,4,1), 'tempogram')
}

inputs_lstm = {
    'cqt' : ((4,84), 'cqt'),
    'cens' : ((4,12), 'cens'),
    'pcp' : ((4,12), 'pcp'),
    'tonnetz' : ((4,6), 'tonnetz'),
    'mfcc' : ((4,14), 'mfcc'),
    'tempogram' : ((4,192), 'tempogram')
}

learning_params = {
    'neurons' : [128, 256, 512],
    'activation' : ['relu', 'elu'],
    'optimizer' : ['rmsprop', 'adam'],
    'drop_out' : [0.1, 0.2, 0.3, 0.4, 0.5],
    'pooling' : ['', 'max', 'average'],
    'epochs' : [1,10,50,100],
    'batch_size' : [1,5,10,50,100], 
    'features' : [
        ['cqt', 'cens', 'pcp', 'tonnetz', 'mfcc', 'tempogram'],
        ['cqt', 'mfcc'],
        ['cens', 'mfcc'],
        ['pcp', 'mfcc'],
        ['tonnetz', 'mfcc'],
        ['cqt', 'mfcc', 'tempogram'],
        ['cens', 'mfcc', 'tempogram'],
        ['pcp', 'mfcc', 'tempogram'],
        ['tonnetz', 'mfcc', 'tempogram'],
        ['cqt', 'tempogram'],
        ['cens', 'tempogram'],
        ['pcp', 'tempogram'],
        ['tonnetz', 'tempogram'],
        ['mfcc', 'tempogram']
    ]
}

learning_params_new = {
    #'neurons' : [128, 256, 512], 
    'neurons' : [512],
    'activation' : ['relu'],
    'optimizer' : ['adam'],
    #'drop_out' : [0.1, 0.25, 0.5],
    'drop_out' : [0.5],
    'pooling' : ['', 'max', 'average'],
    #'pooling' : ['max', 'average'],
    'epochs' : [10, 50, 100],
    'batch_size' : [10,50,100],
    'features': [
        ['cqt', 'tempogram'],
        ['cqt', 'mfcc'],
        ['cqt', 'mfcc', 'tempogram']
        #['cqt', 'cens', 'pcp', 'mfcc', 'tempogram']
    ]
}