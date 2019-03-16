import pandas as pd 

def print_description(data):
    for name, values in data.iteritems():
        print(values.describe(), '\n')