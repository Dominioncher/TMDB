from sklearn.metrics import accuracy_score, r2_score


# Метрика accuracy
def accuracy(target_train, target_predict):
    score = accuracy_score(target_train, target_predict)
    print("accuracy = {}%".format(score * 100))


# Коэфициент детерминации
def r2(target_train, target_predict):
    score = r2_score(target_train, target_predict)
    print("Determination = {}".format(score))


# Прогнать через все метрики
def all_metrics(target_train, target_predict):
    accuracy(target_train, target_predict)
    r2(target_train, target_predict)
