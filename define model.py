import tensorflow as tf
import os
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

# for filename in os.listdir():
#     if filename.endswith(".h5"):
#         model = tf.keras.models.load_model(filename)
#         model.save_weights(f"weights/{filename[:-3]}_weights.h5")



def define_old_model():
    '''
    This is to define simple and old model
    '''
    inputs1 = Input(shape=(1000,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    h_c = Dense(512, activation='relu')(inputs1)
    h,c = tf.split(h_c, num_or_size_splits=2, axis=1)

    # LSTM sequence model
    inputs2 = Input(shape=(MAX_DESC_LEN,))
    se1 = Embedding(VOCAB_SIZE, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2, initial_state=[h,c])

    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(VOCAB_SIZE, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam())
    return model


def define_new_model():
    '''
    This is to define complex and new model
    '''
    inputs1 = Input(shape=(1000,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    hc1 = Dense(2048, activation='relu')(inputs1)
    h_c = Dense(512, activation='relu')(hc1)
    h,c = tf.split(h_c, num_or_size_splits=2, axis=1)
    ee1 = Dense(256, activation='relu')(h_c)
    ee2 = tf.tile(tf.expand_dims(ee1, axis=1), [1,47,1])

    # LSTM sequence model
    inputs2 = Input(shape=(MAX_DESC_LEN,))
    se1 = Embedding(VOCAB_SIZE, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    added1 = add([ee2, se2])
    se3 = LSTM(256)(added1, initial_state=[h,c])


    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(VOCAB_SIZE, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam())
    return model

