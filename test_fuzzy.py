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

print( np.concatenate((temp_max,temp_min)) )

result = fuzzy.generate_time_series_rule_base(temp_min,num_regions=1,window=12,horizon=1,label=False)
