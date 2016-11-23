import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numpy.random.seed(7)

dataframe = pandas.read_csv('inmet.csv', usecols=['Temp Max'], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')


#temp = []
#for i in range(365):
#    x = i*24
#    y = x+24
#    temp.append( (numpy.sum(dataset[x:y])/24 ), )
#
#dataset = []
#dataset = pandas.DataFrame(temp)
#print(dataset)

# normalização dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='tanh'))
#model.add(Dense(6,init='uniform',activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=500, batch_size=8, verbose=2)


#plot(model, to_file='model.png')

# funcao de convergencia por meio de early stopping
def train_until_convergence(model,train_input,train_output):
    previous_trainScore = 60000
    current_trainScore = 50000
    epoch = 0
    while(current_trainScore < previous_trainScore):
        model.fit(train_input,train_output,nb_epoch=1,batch_size=32,verbose=0)
        #model.train_on_batch(train_input,train_output)
        previous_trainScore = current_trainScore
        current_trainScore = model.evaluate(train_input,train_output,verbose=0)
        epoch += 1
    return epoch

def train_until_specific_error(model,train_input,train_output,error):
    trainScore = 1000000
    epoch = 0
    while(trainScore > error):
        model.fit(train_input,train_output,nb_epoch=1,batch_size=32,verbose=0)
        trainScore = model.evaluate(train_input,train_output,verbose=0)
        epoch += 1
    return epoch

#print( train_until_convergence(model,trainX,trainY) )


# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# desnormalizar os dados
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
