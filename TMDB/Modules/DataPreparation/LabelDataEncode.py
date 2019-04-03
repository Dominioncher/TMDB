import pandas as pd
from TMDB.Modules.Helpers.LabelEncoding import simple_encoder_fit, dummy_code_arrays


def encode_labels(data: pd.DataFrame, test_data: pd.DataFrame):

    train, test = dummy_code_arrays(data['genres'], test_data['genres'])

    data = data.join(train)
    test_data = test_data.join(test)

    for col in data.columns:
        if data[col].dtype == pd.np.object:
            train, test = simple_encoder_fit(data[col], test_data[col])
            data[col] = train
            test_data[col] = test

    return data, test_data

