#Tensorflow implementation of the gender model

from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, Flatten, Input 
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def Convolution(input, filters, filter_size):
    x = Conv2D(filters=filters, kernel_size=filter_size, activation='relu',
               padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(input)
    return x


def create_model(input_shape):
    inputs = Input(input_shape)

    x = Normalization()(inputs)

    x = Convolution(x, 32, (5, 5))
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Convolution(x, 64, (4, 4))
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Convolution(x, 128, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    x = Flatten()(x)
    x = Dropout(0.8)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=["binary_crossentropy"],
                  optimizer='adam', metrics=["accuracy"])

    return model

def train_gender_model(X_train, X_test, y_train, y_test, img_shape, batch_size = 100, epochs = 100, model_save = 'g_final.h5', monitor='val_loss'):

    model = create_model(img_shape)

    checkpointer = ModelCheckpoint(
        filepath=model_save,
        monitor=monitor,
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    Early_stop = EarlyStopping(
        patience=25,
        monitor=monitor,
        restore_best_weights=True
    )
    callback_list = [Early_stop, checkpointer]

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[callback_list])
    return model
