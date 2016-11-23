import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.layers import Dense
from keras.layers import Input, merge
from keras.models import Model

numpy.random.seed(7)

dataframe = pandas.read_csv('inmet.csv', usecols=['Temp Max'], engine='python', skipfooter=3)
dataset_temp_max = dataframe.values
dataset_temp_max = dataset_temp_max.astype('float32')

dataframe = pandas.read_csv('inmet.csv', usecols=['Temp Min'], engine='python', skipfooter=3)
dataset_temp_min = dataframe.values
dataset_temp_min = dataset_temp_min.astype('float32')

# split into train and test sets
train_size = int(len(dataset_temp_min) * 0.7)
test_size = len(dataset_temp_min) - train_size
train_temp_max, test_temp_max = dataset_temp_max[0:train_size,:], dataset_temp_max[train_size:len(dataset_temp_max),:]
train_temp_min, test_temp_min = dataset_temp_min[0:train_size,:], dataset_temp_min[train_size:len(dataset_temp_min),:]


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
train_temp_max_X, train_temp_max_Y = create_dataset(train_temp_max, look_back)
train_temp_min_X, train_temp_min_Y = create_dataset(train_temp_min, look_back)
test_temp_max_X, test_temp_max_Y = create_dataset(test_temp_max, look_back)
test_temp_min_X, test_temp_min_Y = create_dataset(test_temp_min, look_back)

# modelo da rna
# entrada: temp max e temp min e suas respectivas janelas
# oculta: 1 neurônio? a ser testado
# saída: temp max e temp min, somente 1 valor

ativacao = 'relu'

temp_max_input = Input(shape=(look_back,),name='temp_max_input')
temp_min_input = Input(shape=(look_back,),name='temp_min_input')

merged = merge([temp_max_input,temp_min_input],mode='concat')
x = Dense(8,activation=ativacao)(merged)

temp_max_output = Dense(1,activation=ativacao, name='temp_max_output')(x)
temp_min_output = Dense(1,activation=ativacao, name='temp_min_output')(x)

model = Model(input=[temp_max_input,temp_min_input], output=[temp_max_output,temp_min_output])

model.compile(loss='mean_squared_error', optimizer='adam')


model.fit({'temp_max_input': train_temp_max_X, 'temp_min_input': train_temp_min_X}, {'temp_max_output': train_temp_max_Y, 'temp_min_output': train_temp_min_Y}, nb_epoch=200, batch_size=8, verbose=2)

# funcao de convergencia por meio de early stopping
def train_until_convergence(model,train_input,train_output):
    previous_trainScore = 60000
    current_trainScore = 50000
    epoch = 0
    while(current_trainScore < previous_trainScore):
        model.fit(train_input,train_output,nb_epoch=1,batch_size=32,verbose=2)
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

# Estimate model performance
trainScore = model.evaluate({'temp_max_input': train_temp_max_X, 'temp_min_input': train_temp_min_X}, {'temp_max_output': train_temp_max_Y, 'temp_min_output': train_temp_min_Y}, verbose=0)
print('Train Score: %.2f and %.2f MSE (%.2f and %.2f RMSE)' % (trainScore[0],trainScore[1], math.sqrt(trainScore[0]), math.sqrt(trainScore[1])))
testScore = model.evaluate({'temp_max_input': test_temp_max_X, 'temp_min_input': test_temp_min_X}, {'temp_max_output': test_temp_max_Y, 'temp_min_output': test_temp_min_Y}, verbose=0)
print('Test Score: %.2f and %.2f MSE (%.2f and %.2f RMSE)' % (testScore[0], testScore[1], math.sqrt(testScore[0]), math.sqrt(testScore[1])))

# generate predictions for training
trainPredict = model.predict({train_temp_max_X,train_temp_min_X})
testPredict = model.predict({test_temp_max_X,test_temp_min_X})

# shift train_temp_max predictions for plotting
train_temp_max_PredictPlot = numpy.empty_like(dataset_temp_max)
train_temp_max_PredictPlot[:, :] = numpy.nan
train_temp_max_PredictPlot[look_back:len(trainPredict[0])+look_back, :] = trainPredict[0]

# shift train_temp_min predictions for plotting
train_temp_min_PredictPlot = numpy.empty_like(dataset_temp_min)
train_temp_min_PredictPlot[:, :] = numpy.nan
train_temp_min_PredictPlot[look_back:len(trainPredict[1])+look_back, :] = trainPredict[1]

# shift test_temp_max predictions for plotting
test_temp_max_PredictPlot = numpy.empty_like(dataset_temp_max)
test_temp_max_PredictPlot[:, :] = numpy.nan
test_temp_max_PredictPlot[len(trainPredict[0])+(look_back*2)+1:len(dataset_temp_max)-1, :] = testPredict[0]

# shift test_temp_min predictions for plotting
test_temp_min_PredictPlot = numpy.empty_like(dataset_temp_min)
test_temp_min_PredictPlot[:, :] = numpy.nan
test_temp_min_PredictPlot[len(trainPredict[1])+(look_back*2)+1:len(dataset_temp_min)-1, :] = testPredict[1]

# plot baseline and predictions
plt.plot(dataset_temp_max)
plt.plot(dataset_temp_min)

plt.plot(train_temp_max_PredictPlot)
plt.plot(train_temp_min_PredictPlot)
plt.plot(test_temp_max_PredictPlot)
plt.plot(test_temp_min_PredictPlot)


plt.xlabel("Observações")
plt.ylabel("Celsius (ºC)")
plt.show()
