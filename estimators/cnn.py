# TODO: Retrain CNN with data_aug set to 3 and 30 epochs.
# TODO: Remove XGB

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical


def train_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    """
    Train model on all training data

    :param X_train: Training images, np.array of shape nxpxpx1
    :param y_train: Training labels, np.array of shape n
    :param X_test: Test images, np.array of shape mxpxpx1
    :param y_test: Test labels, np.array of shape m
    :param epochs: Number of training epochs
    :param batch_size: Training Batch size
    :return: training history information, trained model
    """

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    model = get_model()
    optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='acc',
                                                patience=5,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.000001)

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=2,
                        callbacks=[learning_rate_reduction])

    return history, model


def get_model(input_shape=(28, 28, 1), feature_extraction_layer_name='feature_extraction'):
    """
    Get the CNN architecture

    :param input_shape: Input shape of the data, not including the number of training examples
    :param feature_extraction_layer_name: Name for the final dense layer before the output layer

    :return: The uncompiled keras sequential model
    """
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', name=feature_extraction_layer_name))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def get_feature_extraction_model(model, feature_extraction_layer_name='feature_extraction'):
    """
    Get the model for the last dense layer before the output layer

    :param model: The keras model that has already been trained
    :param feature_extraction_layer_name: Name of the final dense layer before the output layer

    :return: keras model
    """
    return Model(inputs=model.input, outputs=model.get_layer(feature_extraction_layer_name).output)
