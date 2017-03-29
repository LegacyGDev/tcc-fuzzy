import pandas as pd
import fuzzy

temp_max = pd.read_csv('data_selection.csv',usecols=['TempMaximaMedia'])
temp_max = temp_max.values
temp_max = temp_max.astype('float32')

result = fuzzy.generate_time_series_rule_base(temp_max,num_regions=1,window=12,horizon=1)
print( result )
