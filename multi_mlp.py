import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.layers import Dense
from keras.layers import Input, merge
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numpy.random.seed(7)

dataframe = pandas.read_csv('data/data_selection.csv', usecols=['UmidadeRelativaMedia'], engine='python', skipfooter=3)
dataset_umidade_media = dataframe.values
dataset_umidade_media = dataset_umidade_media.astype('float32')

dataframe = pandas.read_csv('data/data_selection.csv', usecols=['TempMaximaMedia'], engine='python', skipfooter=3)
dataset_temp_max = dataframe.values
dataset_temp_max = dataset_temp_max.astype('float32')

dataframe = pandas.read_csv('data/data_selection.csv', usecols=['TempMinimaMedia'], engine='python', skipfooter=3)
dataset_temp_min = dataframe.values
dataset_temp_min = dataset_temp_min.astype('float32')

# normalizar os dados
def normalize(dataset):
    new_data = []
    x_min = min(dataset)
    x_max = max(dataset)
    y = x_max - x_min
    z = y*0.2
    maior = x_max + z
    menor = x_min - z
    for i in range(len(dataset)):
        new_data.append( (dataset[i] - menor)/(maior-menor) )
    return new_data

#dataset_umidade_media = normalize(dataset_umidade_media)
#dataset_temp_max = normalize(dataset_temp_max)
#dataset_temp_min = normalize(dataset_temp_min)

scaler = MinMaxScaler(feature_range=(0,1))
dataset_umidade_media = scaler.fit_transform(dataset_umidade_media)
dataset_temp_max = scaler.fit_transform(dataset_temp_max)
dataset_temp_min = scaler.fit_transform(dataset_temp_min)

# split into train and test sets
train_size = int(len(dataset_temp_max) * 0.7)
test_size = len(dataset_temp_max) - train_size
train_umidade_media, test_umidade_media = dataset_umidade_media[0:train_size,:], dataset_umidade_media[train_size:len(dataset_umidade_media),:]
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

train_umidade_media_X, train_umidade_media_Y = create_dataset(train_umidade_media, look_back)
test_umidade_media_X, test_umidade_media_Y = create_dataset(test_umidade_media, look_back)
train_temp_max_X, train_temp_max_Y = create_dataset(train_temp_max, look_back)
train_temp_min_X, train_temp_min_Y = create_dataset(train_temp_min, look_back)
test_temp_max_X, test_temp_max_Y = create_dataset(test_temp_max, look_back)
test_temp_min_X, test_temp_min_Y = create_dataset(test_temp_min, look_back)

# modelo da rna
# entrada: temp max e temp min e suas respectivas janelas
# oculta: 1 neurônio? a ser testado
# saída: temp max e temp min, somente 1 valor

ativacao = 'sigmoid'

umidade_media_input = Input(shape=(look_back,),name='umidade_media_input')
temp_max_input = Input(shape=(look_back,),name='temp_max_input')
temp_min_input = Input(shape=(look_back,),name='temp_min_input')

merged = merge([umidade_media_input,temp_max_input,temp_min_input],mode='concat')
x = Dense(8,activation=ativacao)(merged)

temp_max_output = Dense(1,activation=ativacao, name='temp_max_output')(x)
temp_min_output = Dense(1,activation=ativacao, name='temp_min_output')(x)

model = Model(input=[umidade_media_input,temp_max_input,temp_min_input], output=[temp_max_output,temp_min_output])

model.compile(loss='mean_squared_error', optimizer='adam')


model.fit({ 'umidade_media_input': train_umidade_media_X ,'temp_max_input': train_temp_max_X, 'temp_min_input': train_temp_min_X}, {'temp_max_output': train_temp_max_Y, 'temp_min_output': train_temp_min_Y}, nb_epoch=10, batch_size=2, verbose=2)

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



# generate predictions for training
trainPredict = model.predict({ 'umidade_media_input': train_umidade_media_X ,'temp_max_input': train_temp_max_X, 'temp_min_input': train_temp_min_X})
testPredict = model.predict({'umidade_media_input':test_umidade_media_X,'temp_max_input':test_temp_max_X,'temp_min_input':test_temp_min_X})

# inverter as predições
trainPredict = scaler.inverse_transform(trainPredict)



# shift train_temp_max predictions for plotting
train_temp_max_PredictPlot = numpy.empty_like(dataset_umidade_media)
train_temp_max_PredictPlot[:, :] = numpy.nan
train_temp_max_PredictPlot[look_back:len(trainPredict[0])+look_back, :] = trainPredict[0]

# shift train_temp_min predictions for plotting
train_temp_min_PredictPlot = numpy.empty_like(dataset_temp_max)
train_temp_min_PredictPlot[:, :] = numpy.nan
train_temp_min_PredictPlot[look_back:len(trainPredict[1])+look_back, :] = trainPredict[1]

# shift test_temp_max predictions for plotting
test_temp_max_PredictPlot = numpy.empty_like(dataset_umidade_media)
test_temp_max_PredictPlot[:, :] = numpy.nan
test_temp_max_PredictPlot[len(trainPredict[0])+(look_back*2)+1:len(dataset_umidade_media)-1, :] = testPredict[0]

# shift test_temp_min predictions for plotting
test_temp_min_PredictPlot = numpy.empty_like(dataset_temp_max)
test_temp_min_PredictPlot[:, :] = numpy.nan
test_temp_min_PredictPlot[len(trainPredict[1])+(look_back*2)+1:len(dataset_temp_max)-1, :] = testPredict[1]

# plot baseline and predictions
plt.plot(dataset_umidade_media)
plt.plot(dataset_temp_max)

plt.plot(train_temp_max_PredictPlot)
plt.plot(train_temp_min_PredictPlot)
plt.plot(test_temp_max_PredictPlot)
plt.plot(test_temp_min_PredictPlot)


plt.xlabel("Observações")
plt.ylabel("Celsius (ºC)")
plt.show()
