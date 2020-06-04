import pandas as pd
import matplotlib.pyplot as plt
import catboost
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class FitError(Exception):
    pass


class LabelEncoding:

    def _label_encoding(self, data, col, values):
        enc = LabelEncoder().fit(values)
        kwargs = {col: lambda x: enc.transform(pd.DataFrame(data[col]))}
        return data.assign(**kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            X_copy = X.copy()
            X_copy = self._label_encoding(X_copy, 'sex', X_copy.sex.unique())
            X_copy = self._label_encoding(X_copy, 'smoker', X_copy.smoker.unique())
            return X_copy
        except FitError:
            raise

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


class Scaler:
    _sc_list = None

    def _fit_scaler(self, data, cols):
        self._sc_list = []
        for col in cols:
            sc = StandardScaler()
            sc.fit(pd.DataFrame(data[col]))
            self._sc_list += [sc]
        return self._sc_list

    def _scaler(self, data, cols):
        if self._sc_list is None:
            raise FitError('Scaler')
        kwargs = {}
        for sc, col in zip(self._sc_list, cols):
            kwargs[col] = lambda x: sc.transform(pd.DataFrame(data[col]))
        return data.assign(**kwargs)

    def fit(self, X, y=None):
        cols = ['age', 'bmi', 'children']
        self._fit_scaler(X, cols)
        return self

    def transform(self, X, y=None):
        try:
            X_copy = X.copy()
            cols = ['age', 'bmi', 'children']
            X_copy = self._scaler(X_copy, cols)
            return X_copy
        except FitError:
            raise

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


class OneHotEncoding:
    def _one_hot_encoder(self, data, col):
        enc = OneHotEncoder()
        new_cols = pd.DataFrame(enc.fit_transform(pd.DataFrame(data[col])).todense(),
                                columns=map(lambda x: x.replace('x0', col),
                                            enc.get_feature_names()),
                                index=data.index)
        return data.join(new_cols).drop(col, axis=1)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            X_copy = X.copy()
            X_copy = self._one_hot_encoder(X_copy, 'region')
            return X_copy
        except FitError:
            raise

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


def draw_feature_impot_catboost(boosting_cls, xlabel='', ylabel='', title='',
                       save=False, name=None, show=True, fontsize=14):
    draw_feature_impot(boosting_cls.feature_names_, boosting_cls.get_feature_importance(),
                       xlabel, ylabel, title, save, name, show, fontsize)

    
def draw_feature_impot(names, impot, xlabel='', ylabel='', title='',
                       save=False, name=None, show=True, fontsize=14):
    # get correct names
    lst_zip = list(zip(names, impot))
    lst_zip = sorted(lst_zip, key=lambda tup: tup[1], reverse=True)
    impot_features = []
    for tup in lst_zip:
        impot_features.append(tup[0])
    impot_features.reverse()

    plt.figure(figsize=(10, 6))

    # Reorder it following the values:
    my_range=range(1,len(lst_zip)+1)

    # Horisontal version
    plt.hlines(y=my_range, xmin=0, xmax=sorted(impot),
               color='#badcbe', linewidth=6.0)
    plt.plot(sorted(impot),
             my_range, "o", ms=8, color='#549ebb')
    plt.xticks(fontsize=fontsize-2)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize+2)
    ax = plt.gca()
    plt.yticks(my_range, impot_features, #fontstyle='italic',
               fontsize=fontsize, fontfamily='monospace')
    plt.grid(True)
    ax.set_facecolor('#fbfbfe')
    if save:
        if name is None:
            name = 'img.png'
        plt.savefig(name, edgecolor='none', bbox_inches='tight')
    if show:
        plt.show()
