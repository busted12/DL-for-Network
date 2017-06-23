from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import *
import numpy as np
import matplotlib.pyplot as plt




real_answer = np.array([12, 8, 8, 4, 0, 11, 5, 10, 0])

#i = 30000
#(input_array, output_array, number_of_samples) = data_generate.generate_data(i)
#print(number_of_samples)


num_of_neurons = range(10,100,10)
d = np.zeros(len(num_of_neurons))
index = 0


for i in num_of_neurons:

 #   seed = 1
  #  np.random.seed(seed)
    input_array = np.loadtxt('x_data_set')
    output_array = np.loadtxt('y_data_set')
    number_of_samples = np.shape(input_array)[0]

    model = Sequential()
    model.add(Dense(input_dim=5, output_dim=i, kernel_initializer=RandomNormal(seed=1), activation='relu'))
    model.add(Dense(output_dim=i, kernel_initializer=RandomNormal(seed=1), activation='relu'))
    model.add(Dense(output_dim=9, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='Adam')

    history = model.fit(input_array, output_array, validation_split=0.33, batch_size=128, epochs=20)
    d[index] = history.history['val_loss'][-1]
    index += 1
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.close()

plt.plot(d)
plt.xlabel('number of neurons')
plt.ylabel('training loss')
plt.title('2 Hidden layers')
plt.show()
plt.savefig('2 Hidden layers, reurons from 10 to 100')






'''
adc = np.array([20, 0, 0, -5, -15])
x_predict = np.reshape(adc, newshape=(1, 5))
y_predict = model.predict(x=x_predict)
print(y_predict)
real_answer = np.vstack((real_answer, y_predict))

print(real_answer)
'''




'''
for step in range(30001):
    cost = model.train_on_batch(x_train, y_train)
    if step % 100 == 0:
        print('train cost,', cost)
        weights = model.layers[0].get_weights()[0]
        biases = model.layers[0].get_weights()[1]
'''











#W1, b1 = model.layers[0]
#W2, b2 = model.layers[1]
#print(W1)
#print(b1)
