import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, AveragePooling1D, MaxPooling1D


def get_bilstm(input_shapes_names:[], neurons=256, activation='relu', opt='adam', drop_out=0.5, pooling='', multi_lstm=False):
    '''
        Creates a Bi-directional Long Short-Term Memory Artificial Neural Network.
        - Already compiled
        - Input dependent on given input shapes
    '''
    inputs = []
    for shape, name in input_shapes_names:
        inputs.append(Input(shape=shape, name=name))
    conc = concatenate(inputs, axis=2)
    
    if multi_lstm:
        bi = Bidirectional(LSTM(neurons, activation=activation, return_sequences=True))(conc)
        bi = Dropout(drop_out)(bi)

        if pooling == 'max':
            bi = MaxPooling1D(2)(bi)
        elif pooling == 'average':
            bi = AveragePooling1D(2)(bi)

    else:
        bi = Bidirectional(LSTM(neurons, activation=activation, return_sequences=False))(conc)
        bi = Dropout(drop_out)(bi)


    if multi_lstm:
        bi = Bidirectional(LSTM(128, activation=activation))(bi)
        bi = Dropout(drop_out)(bi)
    else:
        bi = Dense(128, activation=activation)(bi)
        bi = Dropout(drop_out)(bi)

    output = Dense(9, activation='softmax')(bi)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model