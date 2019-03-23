from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

RandomForest = RandomForestRegressor()
MLP = MLPRegressor()
Linear = linear_model.LinearRegression()

Lasso = linear_model.Lasso(alpha=0.1, normalize=True)

Bayes = linear_model.BayesianRidge(normalize=True)

Hyber = linear_model.HuberRegressor()

Logic = linear_model.LogisticRegression()