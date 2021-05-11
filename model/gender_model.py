from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

def Convolution(input, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(0.00015))(input)
    x = Dropout(0.2)(x)
    return x

def create_model(input_shape):
    inputs = Input(input_shape)

    x = Convolution(inputs, 16)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Convolution(x, 32)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Convolution(x, 64)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Convolution(x, 128)
    x = Convolution(x, 128)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Convolution(x, 256)
    x = Convolution(x, 512)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)

    x = Dropout(0.1)(x)

    output = Dense(2, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=["binary_crossentropy"], optimizer="Adam", metrics=["accuracy"])

    return model


def train_gender_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, model_save, monitor='val_loss'):

    model = create_model(img_shape)
    #model.summary()

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
    callback_list = [checkpointer, Early_stop]

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        epochs=epochs,
        verbose=1,
        callbacks=[callback_list]
    )
    return model
