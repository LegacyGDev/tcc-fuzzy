import pandas
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np

#dataset = pandas.read_csv("Dads.csv", usecols=['Temperatura'])
#dataset = dataset['Temperatura']

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
    interval = (min(variable),max(variable))
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
    print(degrees)
    assignment = {}
    max_key = max(degrees, key=lambda k: degrees[k])
    assignment[x] = max_key
    return assignment

# inputs is a dictionary in which the keys are the name of them inputs
# outputs is also a dictionary in which they keys are the name of them outputs
# regions is a dict in which them keys are the same name as of them inputs
def generate_fuzzy_rule(inputs,outputs,inputs_regions,outputs_regions):
    antecedents = []
    consequents = []
    for i in inputs:
        for j in i:
            antecedents.append(determine_degrees_and_assign_and_label(j,inputs_regions[i]))
    for o in outputs:
        for p in o:
            consequents.append(determine_degrees_and_assign_and_label(o,outputs_regions[p]))
    rule = {'if': antecedents, 'then': consequents}
    return rule

#def remove_duplicate_rules(fuzzy_rule_base):
#    for f in fuzzy_rule_base:
#        for r in fuzzy_rule_base:
#            if(r['if'] == f['if']):

