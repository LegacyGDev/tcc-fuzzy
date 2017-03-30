import pandas
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np

def divide_into_fuzzy_regions(variable,n):
    regions = []
    num_regions = (2*n)+1
    interval = (min(variable),max(variable))
    region_length = (interval[1] - interval[0])/n
    for i in range(num_regions):
        if(i==0):
            regions.append((interval[0],interval[0],interval[0]+(region_length/2)))
        elif(i==num_regions-1):
            regions.append((interval[1]-(region_length/2),interval[1],interval[1]))
        else:
            lower_bound = interval[0] + ((region_length/2) * (i-1))
            regions.append( ( lower_bound , lower_bound + (region_length/2) , lower_bound + region_length ) )
    return regions

def divide_into_fuzzy_regions_and_label(variable,n):
    regions = {}
    num_regions = (2*n)+1
    interval = (variable.min(),variable.max())
    region_length = (interval[1] - interval[0])/n
    for i in range(num_regions):
        if(i==0):
            regions["S{}".format(n)] = (interval[0],interval[0],interval[0]+(region_length/2))
        elif(i==num_regions-1):
            regions["B{}".format(n)] = (interval[1]-(region_length/2),interval[1],interval[1])
        else:
            lower_bound = interval[0] + ((region_length/2) * (i-1))
            if(i < n):
            	regions["S{}".format(n-i)] = ( lower_bound , lower_bound + (region_length/2) , lower_bound + region_length )
            elif(i > n):
            	regions["B{}".format(i-n)] = ( lower_bound , lower_bound + (region_length/2) , lower_bound + region_length )
            elif(i == n):
            	regions["CE"] = ( lower_bound , lower_bound + (region_length/2) , lower_bound + region_length )
    return regions

def determine_degrees_and_assign(x,regions):
    degrees = []
    for r in regions:
    	degrees.append(fuzz.trimf(np.asarray([x]),r))
    assignment = {}
    assignment[x] = regions[np.argmax(degrees)]
    return assignment

def determine_degrees_and_assign_and_label(x,regions):
    degrees = {}
    for k,v in regions.items():
        degrees[k] = fuzz.trimf(np.asarray([x]),v)
    #print(degrees)
    assignment = {}
    max_key = max(degrees, key=lambda k: degrees[k])
    assignment[x] = max_key
    return assignment

# input: {'1st_variable': [variable_regions,values], '2nd_variable': [variable_regions,values]}
# current_input: [(variable_regions,variable_values),(variable_regions,variable_values),(,)]
# current_output: {'if': [{value: region},{nother_value: its_region}], 'then': same_thing_as_if}
def generate_fuzzy_rule(inputs,outputs,label=True):
    antecedents = []
    consequents = []
    for i in inputs:
        for j in i[1]:
            if label:
                antecedents.append(determine_degrees_and_assign_and_label(j,i[0]))
            else:
                antecedents.append(determine_degrees_and_assign(j,i[0]))
    for o in outputs:
        for p in o[1]:
            if label:
                consequents.append(determine_degrees_and_assign_and_label(p,o[0]))
            else:
                consequents.append(determine_degrees_and_assign(p,o[0]))
    rule = {'if': antecedents, 'then': consequents}
    return rule

def generate_time_series_rule_base(data,num_regions=1,window=3,horizon=1,label=True):
    data_regions = []
    for d,item in enumerate(data):
        if label:
            data_regions.append(divide_into_fuzzy_regions_and_label(item,num_regions))
        else:
            data_regions.append(divide_into_fuzzy_regions(item,num_regions))
    observations = len(data[0])
    rule_base = []
    for i in range(window,observations-horizon,1):
        array_window = data[:,i-window:i]
        array_window = np.ravel(array_window)
        array_horizon = data[:,i+horizon]
        rule_base.append( generate_fuzzy_rule( [(data_regions[0],array_window)],[(data_regions[0],array_horizon)],label ))
    return rule_base

#def clean_conflicting_rule_base(rule_base):
#    for r in rule_base:
#        rule_with_max_degree = r
#        for b in rule_base:
#            if(r['if'] == b['if']):
