import tensorflow as tf

def initialize_session():
    devs = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devs[0], True)
    #print('MEMORY GROWTH:', tf.config.experimental.get_memory_growth(devs[0]))

# devs = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devs[0], True)
# print('MEMORY GROWTH:', tf.config.experimental.get_memory_growth(devs[0]))