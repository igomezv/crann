import tensorflow as tf
from astroNN.nn.layers import MCDropout

class Crann:
    def __init__(self, X_train, Y_train, X_val, Y_val, topology, split=0.8, batch_size=64, epochs=200,
                 min_delta=0, patience=10, lr=0.0001, mcdo=False, dropval=0.3, saveModel=False, models_dir='models'):
        # topology refers only for the hidden layers, the input and output layers are infered by the X, Y datasets.
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
        self.last_act_fn.val = 'linear'
        self.act_fn.val = 'relu'

        if mcdo:
            self.model = self.ff_nn_mc_do(dropval)
        else:
            self.model = self.ffnn()
    def ff_nn_mc_do(self, dropval=0.3):
        # Defeine Keras model for regression
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.num_units.val, input_shape=(int(self.X_train.shape[1]),)))

        for nnodes in self.topology:
            model.add(tf.keras.layers.Dense(nnodes, activation=self.act_fn.val))
            model.add(MCDropout(dropval))

        model.add(tf.keras.layers.Dense(int(self.Y_train.shape[1]), activation=self.last_act_fn.val))
        return model

    def ffnn(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.num_units.val, input_shape=(int(self.X_train.shape[1]),)))

        for nnodes in self.topology:
            model.add(tf.keras.layers.Dense(nnodes, activation=self.act_fn.val))

        model.add(tf.keras.layers.Dense(int(self.Y_train.shape[1]), activation=self.last_act_fn.val))
        return model

    def train(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate.val, beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-3)

        # optimizer = Adam(lr=.005)
        # Compile Keras model
        self.model.compile(loss='mse', optimizer=optimizer)
        # model2_train = model.fit(zz_train, yy_train,
        # #                          validation_split=0.0,
        #                          batch_size=batch_size, epochs=1000, verbose=1,
        #                          validation_data=(zz_test, yy_test))
        model_train = self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size,
                                    epochs=self.epochs, verbose=1,
                                    validation_data=(self.X_val, self.Y_val))

        return model_train