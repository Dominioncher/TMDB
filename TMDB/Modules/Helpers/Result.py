import pandas as pd


# Статистика результата обучения
def result_frame(target_test, target_predict) -> pd.DataFrame:
    result = pd.DataFrame(target_test)
    result.columns = ['target']
    result["predict"] = target_predict
    result["|predict - target|"] = (result["predict"] - result["target"]).abs()
    return result
