

import numpy as np
import pandas as pd

import warnings
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, validation_curve
from sklearn.exceptions import ConvergenceWarning

#pip install lazypredict
import lazypredict
from lazypredict.Supervised import LazyRegressor


warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


df= pd.read_csv("d_encoded.csv")

df.drop("Unnamed: 0", axis=1, inplace=True)
df.head()

#Bagımlı ve bagımsız degiskenlerimizi ayırıyoruz
X = df.drop(["SALARY_AVG_TL"], axis=1)
y = df["SALARY_AVG_TL"]


# Veri setini train (%80) ve test (%20) şeklinde iki parçaya ayır.
# Her çalıştırdığımızda aynı değerleri almak için random_state değerini set ettik.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)



reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)

models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

#cıktı:

#                                Adjusted R-Squared  R-Squared     RMSE  Time Taken
# Model
# BayesianRidge                                0.62       0.63 17941.75        0.05
# RidgeCV                                      0.62       0.63 17942.21        0.02
# LassoLars                                    0.62       0.63 17942.53        0.04
# Lasso                                        0.62       0.63 17942.70        0.04
# Ridge                                        0.62       0.63 17942.79        0.01
# TransformedTargetRegressor                   0.62       0.63 17942.87        0.02
# Lars                                         0.62       0.63 17942.87        0.02
# LinearRegression                             0.62       0.63 17942.87        0.04
# LassoLarsIC                                  0.62       0.63 17945.37        0.05
# LarsCV                                       0.62       0.63 17949.17        0.09
# LassoLarsCV                                  0.62       0.63 17951.71        0.15
# LassoCV                                      0.62       0.63 17953.30        0.21
# SGDRegressor                                 0.61       0.62 18100.01        0.02
# OrthogonalMatchingPursuit                    0.61       0.62 18159.89        0.02
# GradientBoostingRegressor                    0.61       0.62 18171.46        0.28
# PoissonRegressor                             0.61       0.62 18216.66        0.03
# OrthogonalMatchingPursuitCV                  0.60       0.62 18244.18        0.04
# RandomForestRegressor                        0.59       0.60 18544.87        0.99
# BaggingRegressor                             0.58       0.59 18840.10        0.12
# ElasticNet                                   0.58       0.59 18878.48        0.02
# PassiveAggressiveRegressor                   0.57       0.59 18931.64        0.07
# HuberRegressor                               0.57       0.59 18932.00        0.07
# LGBMRegressor                                0.57       0.58 19050.49        0.14
# HistGradientBoostingRegressor                0.57       0.58 19050.49        0.87
# KNeighborsRegressor                          0.56       0.58 19154.31        0.19
# TweedieRegressor                             0.53       0.55 19828.58        0.02
# XGBRegressor                                 0.49       0.51 20672.37        0.43
# GammaRegressor                               0.49       0.51 20691.63        0.02
# AdaBoostRegressor                            0.47       0.49 21094.06        0.17
# ExtraTreesRegressor                          0.47       0.48 21197.46        1.12
# DecisionTreeRegressor                        0.37       0.39 23032.02        0.03
# ExtraTreeRegressor                           0.36       0.38 23209.25        0.02
# RANSACRegressor                              0.17       0.19 26478.77        0.23
# ElasticNetCV                                 0.08       0.11 27786.29        0.10
# GaussianProcessRegressor                     0.06       0.09 28114.15        2.84
# DummyRegressor                              -0.03      -0.00 29503.98        0.01
# KernelRidge                                 -0.05      -0.02 29797.45        0.84
# MLPRegressor                                -0.06      -0.02 29860.13        2.54
# NuSVR                                       -0.09      -0.05 30276.30        1.18
# SVR                                         -0.12      -0.09 30755.05        1.53
# QuantileRegressor                           -0.13      -0.09 30827.73      991.44
# LinearSVR                                   -0.53      -0.48 35880.44        0.03














