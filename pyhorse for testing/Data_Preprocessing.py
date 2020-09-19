#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Data Preprocessing
"""

#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
pd.set_option('mode.chained_assignment', None)
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
# from keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

class Raw(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def __name__(self):
        return 'Raw'

    def fit(self, X):
        return self

    def transform(self, X):
        """
        No Preprocessing
        """
        X_copy = X.copy()
        X_copy.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)

        #Convert all features into floats
        for col in X_copy.columns[2:]:
            X_copy.loc[:,col] = X_copy.loc[:,col].astype(float)

        return X_copy


class Normalise_Race(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def __name__(self):
        return 'Normalise_Race'

    def fit(self, X):
        return self

    def transform(self, X):
        X_copy = X.copy()
        try :
            X_copy.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)
        except :
            pass
        X_Transformed = X_copy.loc[:, X_copy.columns[:2]]

        """
        Normalise Data by Race
        """
        races = X_copy.groupby('RARID')
        means = races.transform(np.mean)
        std = races.transform(np.std)

        X_Transformed = pd.concat([X_Transformed, (X_copy[means.columns] - means) / std], axis=1)
        #If a column within a race is equal, it will yield a NaN, we fill this with NA
        X_Transformed.fillna(0, inplace = True)

        #Convert all features into floats
        for col in X_Transformed.columns[2:]:
            X_Transformed.loc[:,col] = X_Transformed.loc[:,col].astype(float)

        return X_Transformed


class Normalise_Profile(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def __name__(self):
        return 'Normalise_Profile'

    def fit(self, X):

        """
        Normalise Data by Profile
        """
        X_copy = X.copy()

        X_copy.drop(['RARID'], axis = 1, inplace = True)
        groups = X_copy.groupby(['RALOC','RADIS','RATRA'])

        self.profile_mean_sd = {}
        for name, group in groups:
            group_mean = group.mean()
            group_std = group.std()
            self.profile_mean_sd[name] = {'mean' : group_mean, 'std' : group_std}

        return self

    def transform(self, X):

        """
        Normalise Data by Profile
        """
        X_copy = X.copy()
        groups = X_copy.groupby(['RALOC','RADIS','RATRA'])

        X_Transformed = pd.DataFrame()
        for name, group in groups:
            means = self.profile_mean_sd[name]['mean']
            stds = self.profile_mean_sd[name]['std']
            for col in group.columns[5:]:
                group.loc[:,col] = (group.loc[:,col] - means.loc[col]) / stds.loc[col]
            X_Transformed = X_Transformed.append(group)
        X_Transformed.sort_values(by = ['RARID','HNAME'], inplace = True)
        X_Transformed.reset_index(inplace = True, drop = True)

        #If a column within a race is equal, it will yield a NaN, we fill this with NA
        X_Transformed.fillna(0, inplace = True)

        #Drop used columns
        try :
            X_Transformed.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)
        except :
            pass

        #Convert all features into floats
        for col in X_Transformed.columns[2:]:
            X_Transformed.loc[:,col] = X_Transformed.loc[:,col].astype(float)

        return X_Transformed


class Log_x(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def __name__(self):
        return 'Log_x'

    def fit(self, X):
        return self

    def transform(self, X):
        X_copy = X.copy()
        try :
            X_copy.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)
        except :
            pass
        """
        Apply Log(1+X-min(X)) to Dataset
        """
        for col in X_copy.columns[2:]:
            min_value = min(X_copy.loc[:,col])
            X_copy.loc[:,col] = X_copy.loc[:,col].apply(lambda x : np.log(1+x-min_value))
            #Convert all features into floats
            X_copy.loc[:,col] = X_copy.loc[:,col].astype(float)

        """
        Normalise Data by Race
        """
        X_Transformed = X_copy.loc[:, X_copy.columns[:2]]
        races = X_copy.groupby('RARID')
        means = races.transform(np.mean)
        std = races.transform(np.std)

        X_Transformed = pd.concat([X_Transformed, (X_copy[means.columns] - means) / std], axis=1)
        #If a column within a race is equal, it will yield a NaN, we fill this with NA
        X_Transformed.fillna(0, inplace = True)

        return X_copy


class pca(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def __name__(self):
        return 'pca'

    def fit(self, X):

        X_copy = X.copy()
        try :
            X_copy.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)
        except :
            pass
        """
        Normalise Data by Race
        """
        races = X_copy.groupby('RARID')
        means = races.transform(np.mean)
        std = races.transform(np.std)
        X_Transformed = X_copy.loc[:, X_copy.columns[:2]]
        X_Transformed = pd.concat([X_Transformed, (X_copy[means.columns] - means) / std], axis=1)

        #If a column within a race is equal, it will yield a NaN, we fill this with NA
        X_Transformed.fillna(0, inplace = True)

        """
        Train PCA transformer
        """
        self.transformer = PCA(n_components = 0.9999, svd_solver = 'full')#(n_components='mle')
        self.transformer.fit(X_Transformed.loc[:, X_Transformed.columns[2:]])

        return self

    def transform(self, X):

        X_copy = X.copy()
        try :
            X_copy.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)
        except :
            pass
        """
        Normalise Data by Race
        """
        races = X_copy.groupby('RARID')
        means = races.transform(np.mean)
        std = races.transform(np.std)
        X_Transformed = X_copy.loc[:, X_copy.columns[:2]]
        X_Transformed = pd.concat([X_Transformed, (X_copy[means.columns] - means) / std], axis=1)

        #If a column within a race is equal, it will yield a NaN, we fill this with NA
        X_Transformed.fillna(0, inplace = True)

        """
        PCA Transform
        """
        X_pca = self.transformer.transform(X_Transformed.loc[:, X_Transformed.columns[2:]])
        X_pca = pd.DataFrame(X_pca, columns = ['PC'+ str(i) for i in range(X_pca.shape[1])])
        X_RARIDHNAME = X_Transformed.loc[:, X_Transformed.columns[:2]]
        X_pca = pd.concat([X_RARIDHNAME, X_pca], axis=1)

        #Convert all features into floats
        for col in X_pca.columns[2:]:
            X_pca.loc[:,col] = X_pca.loc[:,col].astype(float)

        return X_pca


class NN_Features(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def __name__(self):
        return 'NN_Features'

    def fit(self, X):
        pass


    def transform(self, X):
        pass

class Box_Cox(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def __name__(self):
        return 'Box_Cox'

    def fit(self, X):

        X_copy = X.copy()
        try :
            X_copy.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)
        except :
            pass

        """
        Train Box_Cox transformer
        """
        self.transformer = PowerTransformer(method = 'yeo-johnson', standardize = False)
        self.transformer.fit(X_copy.loc[:,X_copy.columns[2:]])

        return self

    def transform(self ,X):
        X_copy = X.copy()
        try :
            X_copy.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)
        except :
            pass
        X_Transformed = X_copy.loc[:, X_copy.columns[:2]]

        """
        Box Cox Transform
        """
        X_boc_cox = pd.DataFrame(self.transformer.transform(X_copy.loc[:, X_copy.columns[2:]]))
        X_boc_cox.columns = X_copy.columns[2:]
        X_boc_cox = pd.concat([X_Transformed, X_boc_cox], axis=1)

        """
        Normalise Data by Race
        """
        races = X_boc_cox.groupby('RARID')
        means = races.transform(np.mean)
        std = races.transform(np.std)
        X_Transformed = pd.concat([X_Transformed, (X_boc_cox[means.columns] - means) / std], axis=1)

        #If a column within a race is equal, it will yield a NaN, we fill this with NA
        X_Transformed.fillna(0, inplace = True)

        #Convert all features into floats
        for col in X_Transformed.columns[2:]:
            X_Transformed.loc[:,col] = X_Transformed.loc[:,col].astype(float)

        return X_Transformed


class Poly2(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def __name__(self):
        return 'Poly2'

    def fit(self, X):
        return self

    def transform(self, X):
        X_copy = X.copy()
        try :
            X_copy.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)
        except :
            pass
        """
        Create Polynomial Variables -> Normalise
        """
        power_df = np.power(X_copy.loc[:,X_copy.columns[2:]], 2)
        power_df.columns = [i + ' ** 2' for i in power_df.columns]
        X_Power = X_copy.join(power_df)

        """
        Normalise Data by Race
        """
        X_Transformed = X_Power.loc[:, X_Power.columns[:2]]
        races = X_Power.groupby('RARID')
        means = races.transform(np.mean)
        std = races.transform(np.std)
        X_Transformed = pd.concat([X_Transformed, (X_Power[means.columns] - means) / std], axis=1)

        #If a column within a race is equal, it will yield a NaN, we fill this with NA
        X_Transformed.fillna(0, inplace = True)

        #Convert all features into floats
        for col in X_Transformed.columns[2:]:
            X_Transformed.loc[:,col] = X_Transformed.loc[:,col].astype(float)

        return X_Transformed


class PolyInter(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def __name__(self):
        return 'PolyInter'

    def fit(self, X):
        return self

    def transform(self, X):
        X_copy = X.copy()
        try :
            X_copy.drop(['RADIS', 'RALOC', 'RATRA'], axis = 1, inplace = True)
        except :
            pass
        X_Transformed = X_copy.loc[:, X_copy.columns[:2]]

        """
        Create Interaction Variables
        """
        poly = PolynomialFeatures(2, interaction_only = True, include_bias = False)
        X_Inter = pd.DataFrame(poly.fit_transform(X_copy.loc[:,X_copy.columns[2:]]))
        X_Inter.columns = poly.get_feature_names(X_copy.columns[2:])
        X_Inter = pd.concat([X_Transformed, X_Inter], axis=1)

        """
        Normalise Data by Race
        """
        races = X_Inter.groupby('RARID')
        means = races.transform(np.mean)
        std = races.transform(np.std)
        X_Transformed = pd.concat([X_Transformed, (X_Inter[means.columns] - means) / std], axis=1)

        #If a column within a race is equal, it will yield a NaN, we fill this with NA
        X_Transformed.fillna(0, inplace = True)

        #Convert all features into floats
        for col in X_Transformed.columns[2:]:
            X_Transformed.loc[:,col] = X_Transformed.loc[:,col].astype(float)

        return X_Transformed


Preprocessing_Dict = {'Raw': 'Normalise Data by Race',
                      'Normalise_Race' : 'Normalise Data by Race',
                      'Normalise_Profile' : 'Normalise Data by Race Profile',
                      'Log_x' : 'Apply Log(1+X) to Dataset',
                      'pca' : 'Apply Principle Component Analysis transformation on Dataset',
                      'NN_Features' : 'Predict Features using Neural Net model',
                      'Box_Cox' : 'Apply Box Cox Tranformation on Dataset',
                      # 'SMOTE' : 'Not Yet Implemented',
                      'Poly2' : 'Create Polynomial Variables'}
                      # 'PolyInter' : 'Create Interaction Variables'}
