import pandas as pd
from Content.Roots import OTHER_ROOT


# Статистика результата обучения
def result_frame(target_test, target_predict) -> pd.DataFrame:
    result = pd.DataFrame(target_test)
    result.columns = ['target']
    result["predict"] = target_predict
    result["|predict - target|"] = (result["predict"] - result["target"]).abs()
    return result

def kaggle_result(predict, data) -> pd.DataFrame:
    result = pd.DataFrame(data.index)
    result.columns = ['id']
    result["revenue"] = predict
    
    path= OTHER_ROOT + '/submission.csv'
    result.to_csv(path, index=False)
    
    return result