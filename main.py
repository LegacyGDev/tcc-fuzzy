import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
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

margem = 20
qr = 2
janela = 12
# designate fuzzy regions from data
temp_min_fuzzy_regions = fuzzy.divide_into_fuzzy_regions([np.min(temp_min),np.max(temp_max)],qr,margem)
temp_max_fuzzy_regions = temp_min_fuzzy_regions
umidade_media_fuzzy_regions = fuzzy.divide_into_fuzzy_regions(umidade_media,qr,margem)
precipitacao_total_fuzzy_regions = fuzzy.divide_into_fuzzy_regions(precipitacao_total,qr,margem)

# separar em conjuntos de treinamento e teste - método holdout
train_size = int(len(temp_min[0]) * 0.7)
test_size = len(temp_min[0]) - train_size
train_temp_min, test_temp_min = temp_min[:,0:train_size], temp_min[:,train_size-janela:len(temp_min[0])]
train_temp_max, test_temp_max = temp_max[:,0:train_size], temp_max[:,train_size:len(temp_max[0])]
train_umidade_media, test_umidade_media = umidade_media[:,0:train_size], umidade_media[:,train_size:len(umidade_media[0])]
train_precipitacao_total, test_precipitacao_total = precipitacao_total[:,0:train_size], precipitacao_total[:,train_size-janela:len(precipitacao_total[0])]

test_temp_min_Y = temp_min[:,train_size:len(temp_min[0])]
test_temp_max_Y = temp_max[:,train_size:len(temp_max[0])]

# todas as variáveis
todos_rule_base = fuzzy.generate_time_series_rule_base({'temp_min': train_temp_min}, {'temp_min': train_temp_min}, {'temp_min': temp_min_fuzzy_regions}, window=janela, horizon=1, label=True, only_regions=False)
todos_clean = fuzzy.clean_conflicting_rule_base(result)
todos_output = fuzzy.time_series_fuzzy_inference({'temp_min': test_temp_min}, ['temp_min'], {'temp_min': temp_min_fuzzy_regions}, clean, window=janela, inverse_inference=True)

# temps + umidade
umidade_rule_base = fuzzy.generate_time_series_rule_base({'temp_min': train_temp_min}, {'temp_min': train_temp_min}, {'temp_min': temp_min_fuzzy_regions}, window=janela, horizon=1, label=True, only_regions=False)
umidade_clean = fuzzy.clean_conflicting_rule_base(result)
umidade_output = fuzzy.time_series_fuzzy_inference({'temp_min': test_temp_min}, ['temp_min'], {'temp_min': temp_min_fuzzy_regions}, clean, window=janela, inverse_inference=True)

# temps + precipitacao
precip_rule_base = fuzzy.generate_time_series_rule_base({'temp_min': train_temp_min}, {'temp_min': train_temp_min}, {'temp_min': temp_min_fuzzy_regions}, window=janela, horizon=1, label=True, only_regions=False)
precip_clean = fuzzy.clean_conflicting_rule_base(result)
precip_output = fuzzy.time_series_fuzzy_inference({'temp_min': test_temp_min}, ['temp_min'], {'temp_min': temp_min_fuzzy_regions}, clean, window=janela, inverse_inference=True)

# somente temperaturas
temps_rule_base = fuzzy.generate_time_series_rule_base({'temp_min': train_temp_min}, {'temp_min': train_temp_min}, {'temp_min': temp_min_fuzzy_regions}, window=janela, horizon=1, label=True, only_regions=False)
temps_clean = fuzzy.clean_conflicting_rule_base(result)
temps_output = fuzzy.time_series_fuzzy_inference({'temp_min': test_temp_min}, ['temp_min'], {'temp_min': temp_min_fuzzy_regions}, clean, window=janela, inverse_inference=True)


todos_temp_min_pred = [i[0] for i in todos_output]
todos_temp_max_pred = [i[1] for i in todos_output]

umidade_temp_min_pred = [i[0] for i in umidade_output]
umidade_temp_max_pred = [i[1] for i in umidade_output]

precip_temp_min_pred = [i[0] for i in precip_output]
precip_temp_max_pred = [i[1] for i in precip_output]

temps_temp_min_pred = [i[0] for i in temps_output]
temps_temp_max_pred = [i[1] for i in temps_output]

test_temp_min_Y = np.swapaxes(test_temp_min_Y,0,1)
test_temp_max_Y = np.swapaxes(test_temp_max_Y,0,1)

todos_temp_min_abs_error = [abs(todos_temp_min_pred[i] - test_temp_min_Y) for i in range(len(test_temp_min_Y))]
todos_temp_max_abs_error = [abs(todos_temp_max_pred[i] - test_temp_max_Y) for i in range(len(test_temp_max_Y))]

umidade_temp_min_abs_error = [abs(umidade_temp_min_pred[i] - test_temp_min_Y) for i in range(len(test_temp_min_Y))]
umidade_temp_max_abs_error = [abs(umidade_temp_max_pred[i] - test_temp_max_Y) for i in range(len(test_temp_max_Y))]

precip_temp_min_abs_error = [abs(precip_temp_min_pred[i] - test_temp_min_Y) for i in range(len(test_temp_min_Y))]
precip_temp_max_abs_error = [abs(precip_temp_max_pred[i] - test_temp_max_Y) for i in range(len(test_temp_max_Y))]

temps_temp_min_abs_error = [abs(temps_temp_min_pred[i] - test_temp_min_Y) for i in range(len(test_temp_min_Y))]
temps_temp_max_abs_error = [abs(temps_temp_max_pred[i] - test_temp_max_Y) for i in range(len(test_temp_max_Y))]

rng = pd.data_range('11/2009',periods=80,freq='M')

todos_data = pd.DataFrame({'temperatura minima media prevista': todos_temp_min_pred, 'temperatura maxima media prevista': todos_temp_max_pred, 'temperatura minima media real': test_temp_min_Y, 'temperatura maxima media real': test_temp_max_Y, 'temperatura minima media erro absoluto': todos_temp_min_abs_error, 'temperatura maxima media erro absoluto': todos_temp_max_abs_error})
todos_address = "todos-{}.xlsx".format(qr)
todos_data.to_excel(address)

umidade_data = pd.DataFrame({'temperatura minima media prevista': umidade_temp_min_pred, 'temperatura maxima media prevista': umidade_temp_max_pred, 'temperatura minima media real': test_temp_min_Y, 'temperatura maxima media real': test_temp_max_Y, 'temperatura minima media erro absoluto': umidade_temp_min_abs_error, 'temperatura maxima media erro absoluto': umidade_temp_max_abs_error})
umidade_address = "umidade-{}.xlsx".format(qr)
umidade_data.to_excel(address)

precip_data = pd.DataFrame({'temperatura minima media prevista': precip_temp_min_pred, 'temperatura maxima media prevista': precip_temp_max_pred, 'temperatura minima media real': test_temp_min_Y, 'temperatura maxima media real': test_temp_max_Y, 'temperatura minima media erro absoluto': precip_temp_min_abs_error, 'temperatura maxima media erro absoluto': precip_temp_max_abs_error})
precip_address = "precip-{}.xlsx".format(qr)
precip_data.to_excel(address)

temps_data = pd.DataFrame({'temperatura minima media prevista': temps_temp_min_pred, 'temperatura maxima media prevista': temps_temp_max_pred, 'temperatura minima media real': test_temp_min_Y, 'temperatura maxima media real': test_temp_max_Y, 'temperatura minima media erro absoluto': temps_temp_min_abs_error, 'temperatura maxima media erro absoluto': temps_temp_max_abs_error})
temps_address = "temps-{}.xlsx".format(qr)
temps_data.to_excel(address)
