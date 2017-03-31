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

def generate_time_series_rule_base(input_data,output_data,num_regions=1,window=3,horizon=1,label=True):
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
    # separate into input
    inputs = []
    
    # separate into output
    outputs = []
    
    observations = len(input_data[0])
    print(observations)
    rule_base = []
    print(output_data_regions)
    for i in range(window,observations-horizon,1):
        print(i)
        array_window = input_data[:,i-window:i]
        #print(array_window)
        array_horizon = output_data[:,i]
        #print(array_horizon)
        for j,items in enumerate(array_window):
            inputs.append( (input_data_regions[j],items) )
        for j,items in enumerate(array_horizon):
            outputs.append( (output_data_regions[j],np.asarray([items])) )
        #print(inputs[0][1][0])
        #print(outputs[0][1])
        rule_base.append( generate_fuzzy_rule( inputs,outputs,label ))
    return rule_base

def clean_conflicting_rule_base(rule_base):
    new_rule_base = []
    for r in rule_base:
        rule_with_max_degree = r
        for b in rule_base:
            if(r['if'] == b['if'] and r['then'] != b['then']):
                deg_r, deg_b = 1,1
                for ant in r['if']:
                    for k,v in ant.items():
                        #print("k{} and v{}".format(k,v))
                        deg_r *= fuzz.trimf(np.asarray([k]),v)
                for con in r['then']:
                    for k,v in con.items():
                        deg_r *= fuzz.trimf(np.asarray([k]),v)
                for ant in b['if']:
                    for k,v in ant.items():
                        deg_b *= fuzz.trimf(np.asarray([k]),v)
                for con in b['then']:
                    for k,v in con.items():
                        deg_b *= fuzz.trimf(np.asarray([k]),v)
                #print("{} and {}".format(deg_r,deg_b))
                if deg_r >= deg_b:
                    rule_with_max_degree = r
                else:
                    rule_with_max_degree = b
        new_rule_base.append(rule_with_max_degree)
    return new_rule_base
