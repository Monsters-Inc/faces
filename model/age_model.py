from tensorflow.keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data import augumentation

def Convolution(input, filters, filter_size):
    x = Conv2D(filters=filters, kernel_size=filter_size, activation='relu',
               padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(input)
    #x = Dropout(0.2)(x)
    return x


def create_model(input_shape):
    inputs = Input(input_shape)

    x = Convolution(inputs, 32, (5, 5))
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Convolution(x, 64, (4, 4))
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Convolution(x, 128, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    x = Flatten()(x)
    x = Dropout(0.8)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation="relu")(x)

    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=["mae"],
                  optimizer='adam')

    return model

def train_age_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, aug, model_save, monitor='val_loss'):

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
  callback_list = [checkpointer, Early_stop]

  if aug:
    datagen = augumentation(X_train)
    model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[callback_list])
  else:
    print('hej')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[callback_list])

  return model