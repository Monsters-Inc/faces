from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, Flatten, Input, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split


def Convolution(input_tensor, filters):

    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
               strides=(1, 1), kernel_regularizer=l2(0.00015))(input_tensor)
    x = Dropout(0.1)(x)
    x = Activation('relu')(x)

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
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Convolution(x, 256)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Convolution(x, 512)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = Dropout(0.2)(x)

    output_1 = Dense(1, activation="sigmoid", name='women_out')(x)
    output_2 = Dense(1, activation="sigmoid", name='men_out')(x)

    model = Model(inputs=[inputs], outputs=[output_1, output_2])
    model.compile(loss=["binary_crossentropy", "binary_crossentropy"], optimizer="Adam", metrics=["accuracy"])

    return model


def train_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, model_save):

    model = create_model(img_shape)
    model.summary()

    checkpointer = ModelCheckpoint(
        filepath=model_save,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    Early_stop = EarlyStopping(
        patience=25,
        monitor='val_loss',
        restore_best_weights=True
    )
    callback_list = [checkpointer, Early_stop]

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        epochs=epochs,
        callbacks=[callback_list]
    )
    return model
