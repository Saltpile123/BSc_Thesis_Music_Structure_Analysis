import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Input, Flatten, concatenate, Dense, Dropout


def get_cnn(input_shapes_names:[], neurons=256, kernel_size=(3,1), activation='relu', opt='adam', drop_out=0.5, pooling='max'):
    '''
        Creates a Convolutional (Artificial) Neural Nerwork.
        - Already Compiled
        - Input dependent on given input shapes
    '''
    inputs = []
    for shape, name in input_shapes_names:
        inputs.append(Input(shape=shape, name=name))

    conv_layers = []
    for input in inputs:
        conv_layer = Conv2D(neurons, kernel_size=kernel_size, activation=activation)(input)
        if pooling == 'max':
            conv_layer = MaxPooling2D((2,2))(conv_layer)
        elif pooling == 'average':
            conv_layer = AveragePooling2D((2,2))(conv_layer)
        conv_layer = Conv2D(128, kernel_size=kernel_size, activation=activation)(conv_layer)
        conv_layers.append(conv_layer)
    x = concatenate(conv_layers, axis=1)
    output = Conv2D(64, kernel_size=kernel_size, activation=activation)(x)
    output = Flatten()(output)
    output = Dense(9, activation='softmax')(output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model