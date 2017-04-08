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

def determine_degrees_and_assign(x,regions,only_regions=False):
    degrees = []
    for r in regions:
    	degrees.append(fuzz.trimf(np.asarray([x]),r))
    max_value = regions[np.argmax(degrees)]
    if only_regions:
        return max_value
    else:
        return (x,regions[np.argmax(degrees)])

def determine_degrees_and_assign_and_label(x,regions,only_regions=False):
    degrees = {}
    for k,v in regions.items():
        degrees[k] = fuzz.trimf(np.asarray([x]),v)
    max_key = max(degrees, key=lambda k: degrees[k])
    if only_regions:
        return max_key
    else:
        return (x,max_key)

# current_input: (array_of_values,array_of_regions)
# current_output: {'if': [{value: region},{nother_value: its_region}], 'then': same_thing_as_if}
def generate_fuzzy_rule(inputs,outputs,label=True,only_regions=False):
    antecedents = []
    consequents = []
    for i,items in enumerate(inputs[0]):
        for j in items:
            if label:
                antecedents.append(determine_degrees_and_assign_and_label(j,inputs[1][i]),only_regions)
            else:
                antecedents.append(determine_degrees_and_assign(j,inputs[1][i],only_regions))
    for o,item in enumerate(outputs[0]):
        if label:
            consequents.append(determine_degrees_and_assign_and_label(item,outputs[1][o],only_regions))
        else:
            consequents.append(determine_degrees_and_assign(item,outputs[1][o],only_regions))
    rule = {'if': antecedents, 'then': consequents}
    return rule

def generate_time_series_rule_base(input_data,output_data,num_regions=1,window=3,horizon=1,label=True,only_regions=False):
    input_data_regions = []
    output_data_regions = []
    for d,item in enumerate(input_data):
        if label:
            input_data_regions.append(divide_into_fuzzy_regions_and_label(item,num_regions))
        else:
            input_data_regions.append(divide_into_fuzzy_regions(item,num_regions))
    for d,item in enumerate(output_data):
        if label:
            output_data_regions.append(divide_into_fuzzy_regions_and_label(item,num_regions))
        else:
            output_data_regions.append(divide_into_fuzzy_regions(item,num_regions))
    inputs = []
    outputs = []
    observations = len(input_data[0])
    rule_base = []
    for i in range(window,observations-horizon,1):
        array_window = input_data[:,i-window:i]
        array_horizon = output_data[:,i]
        rule_base.append( generate_fuzzy_rule( (array_window,input_data_regions),(array_horizon,output_data_regions),label,only_regions ))
    return rule_base

def check_conflicting_rules(rule_base):
    rule_base_without_pair = []
    for r in rule_base:
        rule_base_without_pair.append({'if': [rule[1] for rule in r['if']], 'then': [rule[1] for rule in r['then']]})
    new_rule_base = []
    aux = set()
    conflicting_rules = []
    for i, rule in enumerate(rule_base_without_pair):
        for j, bule in enumerate(rule_base_without_pair):
            if(rule['if'] == bule['if']):
                aux.add(i)
                aux.add(j)
        conflicting_rules.append(aux)
        aux.clear()
    return conflicting_rules

def clean_conflicting_rule_base(rule_base):
    rule_base_without_pair = []
    aux = {}
    for r in rule_base:
        aux['if'] = [rule[1] for rule in r['if']]
        aux['then'] = [rule[1] for rule in r['then']]
        rule_base_without_pair.append(aux)
    new_rule_base = []
    for i,rule in enumerate(rule_base_without_pair):
        rule_with_max_degree = rule
        for j,bule in enumerate(rule_base_without_pair):
            if(rule['if'] == bule['if']):
                deg_r, deg_b = 1,1
                for k,ant in enumerate(rule['if']):
                        deg_r *= fuzz.trimf(np.asarray([rule_base[i]['if'][k][0]]),ant)
                for k,con in enumerate(rule['then']):
                        deg_r *= fuzz.trimf(np.asarray([rule_base[i]['then'][k][0]]),ant)
                for k,ant in enumerate(bule['if']):
                        deg_b *= fuzz.trimf(np.asarray([rule_base[i]['if'][k][0]]),ant)
                for k,con in enumerate(bule['then']):
                        deg_b *= fuzz.trimf(np.asarray([rule_base[i]['then'][k][0]]),ant)
                if deg_r >= deg_b:
                    rule_with_max_degree = rule
                else:
                    rule_with_max_degree = bule
        new_rule_base.append(rule_with_max_degree)
        print(i)
    return new_rule_base

def fuzzy_inference(inputs,rule_base):
    aux = []
    flattened_input = np.ravel(inputs)
    for rule in rule_base:
        for i,ant in enumerate(rule['if']):
            aux.append(fuzz.trimf(np.asarray([flattened_input[i]]),ant[1]))
        single = min(aux)
        #for i,con in enumerate(rule['then']):
    return aux


def time_series_fuzzy_inference(inputs,rule_base,window=3):
    output_data = []
    observations = len(inputs[0])
    for i in range(window,observations,1):
        array_window = inputs[:,i-window:i]
        output_data.append(fuzzy_inference(array_window,rule_base))
    return output_data
