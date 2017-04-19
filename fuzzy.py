import numpy as np
import progressbar
import itertools
import skfuzzy as fuzz

# to be deleted
import sys


def divide_into_fuzzy_regions(variable,n,safe_margin=0,label=True):
    regions = []
    num_regions = (2*n)+1
    minmax = (np.min(variable),np.max(variable))
    interval_length = minmax[1] - minmax[0]
    interval = (np.min(variable) - (interval_length * (safe_margin/100)),np.max(variable) + (interval_length * (safe_margin/100)))
    region_length = (interval[1] - interval[0])/n
    if label:
        for i in range(num_regions):
            if(i==0):
                regions.append( ("S{}".format(n),(interval[0],interval[0],interval[0]+(region_length/2))) )
            elif(i==num_regions-1):
                regions.append( ("B{}".format(n),(interval[1]-(region_length/2),interval[1],interval[1])) )
            else:
                lower_bound = interval[0] + ((region_length/2) * (i-1))
                if(i < n):
            	    regions.append( ("S{}".format(n-i),( lower_bound , lower_bound + (region_length/2) , lower_bound + region_length )) )
                elif(i > n):
            	    regions.append( ("B{}".format(i-n),( lower_bound , lower_bound + (region_length/2) , lower_bound + region_length )) )
                elif(i == n):
            	    regions.append( ("CE",( lower_bound , lower_bound + (region_length/2) , lower_bound + region_length )) )
    else:
        for i in range(num_regions):
            if(i==0):
                regions.append((interval[0],interval[0],interval[0]+(region_length/2)))
            elif(i==num_regions-1):
                regions.append((interval[1]-(region_length/2),interval[1],interval[1]))
            else:
                lower_bound = interval[0] + ((region_length/2) * (i-1))
                regions.append( ( lower_bound , lower_bound + (region_length/2) , lower_bound + region_length ) )
    return regions

def determine_degrees_and_assign(x,regions,only_regions=False):
    degrees = []
    for r in regions:
        degrees.append(fuzz.trimf(np.asarray([x]),r[1]))
    max_value = regions[np.argmax(degrees)]
    if only_regions:
        return max_value
    else:
        return (x,regions[np.argmax(degrees)])

def determine_degrees_and_assign_and_label(x,regions,only_regions=False):
    degrees = {}
    for r in regions:
        degrees[r[0]] = fuzz.trimf(np.asarray([x]),r[1])
    max_key = max(degrees, key=lambda k: degrees[k])
    if only_regions:
        return max_key
    else:
        return (max_key,degrees[max_key])

# current_input: (array_of_values,array_of_regions)
# current_output: {'if': [{value: region},{nother_value: its_region}], 'then': same_thing_as_if}
def generate_fuzzy_rule(inputs, outputs, regions, label=True, only_regions=False):
    antecedents = []
    consequents = []
    for k,v in inputs.items():
        for i in v[0]:
            if label:
                antecedents.append(determine_degrees_and_assign_and_label(i,regions[k],only_regions))
            else:
                antecedents.append(determine_degrees_and_assign(i,regions[k],only_regions))
    for k,v in outputs.items():
        if label:
            consequents.append(determine_degrees_and_assign_and_label(v[0],regions[k],only_regions))
        else:
            consequents.append(determine_degrees_and_assign(v[0],regions[k],only_regions))
    rule = {'if': antecedents, 'then': consequents}
    return rule

def generate_time_series_rule_base(input_data, output_data, variable_regions, window=3, horizon=1, label=True, only_regions=False):
    observations = len(next(iter(input_data.values()))[0])
    rule_base = []
    bar = progressbar.ProgressBar(maxval=observations, widgets=['Wang-Mendel: ',progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range(window,observations-horizon,1):
        #array_window = input_data[:,i-window:i]
        #array_horizon = output_data[:,i]
        rule_base.append( generate_fuzzy_rule({key: value[:,i-window:i] for (key,value) in input_data.items()}, {key: value[:,i] for (key,value) in output_data.items()}, variable_regions, label, only_regions) )
        bar.update(i)
    bar.finish()
    return rule_base

def check_conflicting_rules(rule_base):
    rule_base_without_pair = []
    for r in rule_base:
        rule_base_without_pair.append({'if': [rule[0] for rule in r['if']], 'then': [rule[0] for rule in r['then']]})
    for i, rule in enumerate(rule_base_without_pair):
        for j, bule in enumerate(rule_base_without_pair):
            if(rule['if'] == bule['if']):
                if(i==j):
                    print("{} itself".format(i))
                else:
                    print("{}:{}".format(i,j))

def clean_conflicting_rule_base(rule_base):
    rule_base_without_pair = []
    for r in rule_base:
        rule_base_without_pair.append({'if': [rule[0] for rule in r['if']],'then': [rule[0] for rule in r['then']]})
    new_rule_base = []
    done = []
    for i,rule in enumerate(rule_base_without_pair):
        if i not in done:
            rule_with_max_degree = rule
            for j,bule in enumerate(rule_base_without_pair):
                if(rule['if'] == bule['if']):
                    done.append(j)
                    deg_r, deg_b = 1,1
                    for k,ant in enumerate(rule['if']):
                        deg_r *= rule_base[i]['if'][k][1]
                    for k,con in enumerate(rule['then']):
                        deg_r *= rule_base[i]['then'][k][1]
                    for k,ant in enumerate(bule['if']):
                        deg_b *= rule_base[i]['if'][k][1]
                    for k,con in enumerate(bule['then']):
                        deg_b *= rule_base[i]['then'][k][1]
                    if deg_r >= deg_b:
                        rule_with_max_degree = rule
                    else:
                        rule_with_max_degree = bule
            new_rule_base.append(rule_with_max_degree)
    return new_rule_base


def defuzz_coa(result, region):
    dividend = 0
    for r in result:
        for reg in region:
            if reg[0] == r[0]:
                dividend += (r[1]*reg[1][1])
    divisor = 0
    for r in result:
        divisor += r[1]
    return dividend/divisor


def fuzzy_inference(inputs, outputs_names, regions, rule_base):
    # fuzzify inputs
    fuzzified_inputs = []
    for k,v in inputs.items():
        for it in v[0]:
            fuzzified_inputs.append( [(reg[0],fuzz.trimf(np.asarray([it]),reg[1])) for reg in regions[k] if fuzz.trimf(np.asarray([it]),reg[1]) != 0] )
    high_mf = []
    high_mf_degree = []
    low_mf = []
    low_mf_degree = []
    for f in fuzzified_inputs:
        if f[0][1] > f[1][1]:
            high_mf.append(f[0][0])
            high_mf_degree.append(f[0][1])
            low_mf.append(f[1][0])
            low_mf_degree.append(f[1][1])
        else:
            high_mf.append(f[1][0])
            high_mf_degree.append(f[1][1])
            low_mf.append(f[0][0])
            low_mf_degree.append(f[0][1]) 
    # inference
    #print(high_mf)
    #print(low_mf)
    result = []
    for i in range(len(rule_base[0]['then'])):
        result.append([])
    for choices in itertools.product([0,1],repeat=len(high_mf)):
        for rule in rule_base:
            if [(high_mf[i] if choice else low_mf[i]) for i, choice in enumerate(choices)] == rule['if']:
                for i,con in enumerate(rule['then']):
                    result[i].append( (con,min(min([ (high_mf_degree[i] if choice else low_mf_degree[i]) for i,choice in enumerate(choices)]))) )
    # aggregation
    aggregated = []
    for i,name in enumerate(outputs_names):
        aggregated.append([])
        for v in regions[name]:
            aggregated[i].append( (v[0],max([r[1] for r in result[i] if r[0] == v[0]])) )
    # defuzz
    defuzzified_result = []
    for i,agg in enumerate(aggregated):
        defuzzified_result.append(defuzz_coa(agg,regions[ outputs_names[i] ]))
    return defuzzified_result


def time_series_fuzzy_inference(inputs, outputs_names, regions, rule_base, window=3):
    output_data = []
    observations = len(next(iter(inputs.values()))[0])
    bar = progressbar.ProgressBar(maxval=observations, widgets=['Fuzzy inference: ', progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range(window,observations,1):
        #array_window = inputs[:,i-window:i]
        output_data.append( fuzzy_inference({key: value[:,i-window:i] for (key,value) in inputs.items()}, outputs_names, regions, rule_base) )
        bar.update(i)
    bar.finish()
    return output_data
