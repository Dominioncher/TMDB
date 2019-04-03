import pandas as pd


def weight_code_production_companies(data: pd.DataFrame)-> pd.DataFrame:
    data['companies'] = data['production_companies'].map(lambda x: sorted([d['id'] for d in get_dictionary(x)]))

    companies_weight = dict()  # Словарь с весами компаний

    for i, row in data.iterrows():
        revenue = 0
        counter = 0
        for company in row['companies']:
            filled_revenue = data[data['companies'].apply(lambda x: company in x)]
            mean_revenue = filled_revenue['revenue'].dropna().mean()

            revenue += mean_revenue
            counter += 1
        else:
            data.set_value(i, 'company_weight', float(revenue) / float(counter))

    data = data.drop(columns=['companies'])

    data['company_weight'] = (data['company_weight']-data['company_weight'].min())/(data['company_weight'].max()-data['company_weight'].min())
    return data


# Используется для парса Json с данными
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d
