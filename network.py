from test_data import INPUTS, OUTPUTS, DATA_SIZE
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, Flatten, TimeDistributed, LSTM, Dropout
from test_data import INPUTS, OUTPUTS

model = Sequential([
    keras.Input(shape=(1, DATA_SIZE)),
    LSTM(128),
    Dense(2, activation="sigmoid"),
])
optimizer = keras.optimizers.RMSprop(learning_rate=0.01) #adam, sgd
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) #steps_per_execution
model.summary()

for index, x in enumerate(model.layers):
    for y in x.weights:
        print(f'{index}: {y.name}, {y.shape}')

# Train model
model.fit(INPUTS, OUTPUTS, epochs=250, batch_size=8, use_multiprocessing=True, verbose=2) #callbacks=[Callback()],
var, accuracy = model.evaluate(INPUTS, OUTPUTS)
print(f'Wow: {var} w/ {accuracy*100:.2f}%')
