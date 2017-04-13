import pandas as pd
import numpy as np
import fuzzy

temp_min = pd.read_csv('data_selection.csv',usecols=['TempMinimaMedia'])
temp_min = temp_min.values
temp_min = temp_min.astype('float32')
temp_min = np.swapaxes(temp_min,0,1)

temp_max = pd.read_csv('data_selection.csv',usecols=['TempMaximaMedia'])
temp_max = temp_max.values
temp_max = temp_max.astype('float32')
temp_max = np.swapaxes(temp_max,0,1)

umidade_media = pd.read_csv('data_selection.csv',usecols=['UmidadeRelativaMedia'])
umidade_media = umidade_media.values
umidade_media = umidade_media.astype('float32')
umidade_media = np.swapaxes(umidade_media,0,1)

precipitacao_total = pd.read_csv('data_selection.csv',usecols=['PrecipitacaoTotal'])
precipitacao_total = precipitacao_total.values
precipitacao_total = precipitacao_total.astype('float32')
precipitacao_total = np.swapaxes(precipitacao_total,0,1)

# separar em conjuntos de treinamento e teste - método holdout
train_size = int(len(temp_min[0]) * 0.7)
test_size = len(temp_min[0]) - train_size
train_temp_min, test_temp_min = temp_min[:,0:train_size], temp_min[:,train_size:len(temp_min[0])]
train_temp_max, test_temp_max = temp_max[:,0:train_size], temp_max[:,train_size:len(temp_max[0])]
train_umidade_media, test_umidade_media = umidade_media[:,0:train_size], umidade_media[:,train_size:len(umidade_media[0])]
train_precipitacao_total, test_precipitacao_total = precipitacao_total[:,0:train_size], precipitacao_total[:,train_size:len(precipitacao_total[0])]

result = fuzzy.generate_time_series_rule_base( np.concatenate((train_temp_min,train_temp_max,train_umidade_media,train_precipitacao_total)), np.concatenate( (train_temp_min,train_temp_max) ) ,num_regions=1,window=12,horizon=1,label=False,only_regions=False)
cleaned_result = fuzzy.clean_conflicting_rule_base(result)

#print(len(cleaned_result))

#final = fuzzy.time_series_fuzzy_inference( np.concatenate((test_temp_min,test_temp_max,test_umidade_media,test_precipitacao_total)) ,cleaned_result,12)
