import pandas
import matplotlib.pyplot as plt

dataset = pandas.read_csv("Dads.csv", usecols=['Temperatura'])
dataset = dataset['Temperatura']
interval = max(dataset) - min(dataset)

print(interval)

def divide_into_fuzzy_regions(minimum,maximum,n):
    regions = []
    interval = maximum - minimum
    num_regions = (2*n)+1
    region_interval = interval/num_regions
    for i in range(num_regions):
