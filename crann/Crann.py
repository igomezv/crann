import tensorflow as tf
from astroNN.nn.layers import MCDropout


class Crann:
    def __init__(self, X_train, Y_train, X_val, Y_val, topology, split=0.8, batch_size=64, epochs=200,
                 min_delta=0, patience=10, lr=0.0001, saveModel=False, models_dir='models'):
        self.topology = topology
        split = split
        self.batch_size = batch_size
        self.epochs = epochs
        min_delta = min_delta
        patience = patience
        self.lr = lr
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val


    def ff_nn_mc_do(self, num_hidden):
        # Defeine Keras model for regression
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(batch_input_shape=((None, 1))))
        model.add(tf.keras.layers.Dense(units=num_hidden[0],
    #                     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    #                     kernel_initializer='he_normal',
                        activation='relu'))
        model.add(MCDropout(0.3))
        model.add(tf.keras.layers.Dense(units=num_hidden[1],
    #                     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    #                     kernel_initializer='he_normal',
                        activation='relu'))
        model.add(MCDropout(0.3))
        model.add(tf.keras.layers.Dense(units=num_hidden[2],
    #                     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    #                     kernel_initializer='he_normal',
                        activation='relu'))
        model.add(MCDropout(0.3))
        model.add(tf.keras.layers.Dense(units=2, activation="linear"))
        return model

    def ffnn(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.num_units.val, input_shape=(int(self.X_train.shape[1]),)))

        for i in range(self.deep.val):
            model.add(tf.keras.layers.Dense(self.num_units.val, activation=self.act_fn.val))
        #             model.add(keras.layers.Dropout(0.3))
        # model.add(tf.keras.layers.Dense(int(self.Y_train.shape[1]), activation=tf.nn.softmax))
        model.add(tf.keras.layers.Dense(int(self.Y_train.shape[1]), activation=self.last_act_fn.val))
        return model


    def train(self, model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate.val, beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-3)
        batch_size = 4
        # optimizer = Adam(lr=.005)

        # Compile Keras model
        model.compile(loss='mse', optimizer=optimizer)
        # model2_train = model.fit(zz_train, yy_train,
        # #                          validation_split=0.0,
        #                          batch_size=batch_size, epochs=1000, verbose=1,
        #                          validation_data=(zz_test, yy_test))
        model_train = model.fit(self.X_train, self.Y_train, batch_size=batch_size,
                                 epochs=800, verbose=1,
                                 validation_data=(self.X_val, self.Y_val))