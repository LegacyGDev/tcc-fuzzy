import pandas
import matplotlib.pyplot as plt
import skfuzzy as fuzz

dataset = pandas.read_csv("Dads.csv", usecols=['Temperatura'])
dataset = dataset['Temperatura']

def divide_into_fuzzy_regions(variable,n):
    regions = []
    num_regions = (2*n)+1
    interval = (min(variable),max(variable))
    region_length = (interval[1] - interval[0])/num_regions
    for i in range(num_regions):
        if(i==0):
            regions.append((interval[0],interval[0],interval[0]+region_length))
        elif(i==num_regions-1):
            regions.append((interval[1]-region_length,interval[1],interval[1]))
        else:
            lower_bound = interval[0] + (region_length * (i-1))
            regions.append( ( lower_bound , lower_bound + (region_length/2) , lower_bound + region_length ) )
    return regions

def determine_degrees(x,regions):
    degrees = []
    for r in regions:
        degrees.append(fuzz.trimf(x,r))
    return degrees
