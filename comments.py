#model.add(Dense(DATA_SIZE, input_shape=(DATA_SIZE,), activation='sigmoid'))
#model.add(SimpleRNN(DATA_SIZE*2, input_shape=(DATA_SIZE, DATA_SIZE * 2)))

#timesteps = 1
#input_dim = DATA_SIZE #inputs.shape[1]
#model.add(LSTM(1, input_shape=(input_dim, 1), return_state=True, activation='sigmoid'))
#model.add(Dropout(0.2))

#model.add(Activation('sigmoid'))
#model.add(TimeDistributed(Dense(DATA_SIZE * 2, activation='sigmoid')))
#model.add(Activation('sigmoid'))
#model.add(TimeDistributed(Dense(2, activation='sigmoid')))
#model.add(Activation('sigmoid'))
#opt = keras.optimizers.adadelta_v2(learning_rate=0.001) #instead of sgd
# mse: mean_squared_error