import pandas as pd
pd.options.display.float_format = '{:20,.2f}'.format


def description(data: pd.DataFrame, path: str = None)-> None:
    if path:
        text_file = open(path, "w+")
    for name, values in data.iteritems():
        value = 'COLUMN : ' + name + '\n\n' + values.describe().to_string() + \
                '\n------------------------------------------------------------------------------------\n '
        print(value)
        if path:
            text_file.write(value)
    if path:
        text_file.close()
