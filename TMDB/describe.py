import pandas as pd 

def print_description(data):
    for name, values in data.iteritems():
        print(values.describe(), '\n')
        
def file_description(data, filename):
    text_file = open(filename, "w+")
    for name, values in data.iteritems():
        text_file.write(values.describe().to_string())
        text_file.write('\n')
    text_file.close()