import pandas as pd
from TMDB.Modules.Helpers.LabelEncoding import simple_encoder_fit, dummy_code


def encode_labels(data: pd.DataFrame, test_data: pd.DataFrame):

    train, test = dummy_code(data['genres'], test_data['genres'])
    data = data.join(train)
    test_data = test_data.join(test)
    data = data.drop(["genres"], axis=1)
    test_data = test_data.drop(["genres"], axis=1)

    train, test = dummy_code(data['production_countries'], test_data['production_countries'])
    data = data.join(train)
    test_data = test_data.join(test)
    data = data.drop(["production_countries"], axis=1)
    test_data = test_data.drop(["production_countries"], axis=1)

    train, test = dummy_code(data['spoken_languages'], test_data['spoken_languages'])
    data = data.join(train)
    test_data = test_data.join(test)
    data = data.drop(["spoken_languages"], axis=1)
    test_data = test_data.drop(["spoken_languages"], axis=1)

    for col in data.columns:
        if data[col].dtype == pd.np.object:
            train, test = simple_encoder_fit(data[col], test_data[col])
            data[col] = train
            test_data[col] = test

    return data, test_data

