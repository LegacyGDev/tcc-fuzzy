import pandas as pd
import numpy as np
import fuzzy

temp_max = pd.read_csv('data_selection.csv',usecols=['TempMaximaMedia'])
temp_max = temp_max.values
temp_max = temp_max.astype('float32')
temp_max = np.swapaxes(temp_max,0,1)

temp_min = pd.read_csv('data_selection.csv',usecols=['TempMinimaMedia'])
temp_min = temp_min.values
temp_min = temp_min.astype('float32')
temp_min = np.swapaxes(temp_min,0,1)

umidade_media = pd.read_csv('data_selection.csv',usecols=['UmidadeRelativaMedia'])
umidade_media = umidade_media.values
umidade_media = umidade_media.astype('float32')
umidade_media = np.swapaxes(umidade_media,0,1)

precipitacao_total = pd.read_csv('data_selection.csv',usecols=['PrecipitacaoTotal'])
precipitacao_total = precipitacao_total.values
precipitacao_total = precipitacao_total.astype('float32')
precipitacao_total = np.swapaxes(precipitacao_total,0,1)


result = fuzzy.generate_time_series_rule_base( np.concatenate((temp_min,temp_max)), np.concatenate( (temp_min,temp_max) ) ,num_regions=1,window=12,horizon=1,label=False)
cleaned_result = fuzzy.clean_conflicting_rule_base(result)
