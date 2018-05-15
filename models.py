import numpy as np
from sklearn import linear_model

class Models:
    """
    API:
    models.coefs: get the model coefs.
    models.results: print out the regression table.
    models.residuals: the array of residuals of the model
    """

    def __init__(self, df, model="linear"):
        self.varnames = df.columns.values.tolist()
        self.X = df.iloc[:, 1:].values
        self.Y = df.iloc[:, 0].values
        if self.X.dtype != 'float64':
            self.X = self.X.astype(float)
        if self.Y.dtype != 'float64':
            self.Y = self.Y.astype(float)
        self.model = model

    @property
    def regr(self):
        try:
            return self._regr
        except AttributeError:
            if self.model == "linear":
                regr = linear_model.LinearRegression()
                self.Y_fit = np.copy(self.Y)
            elif self.model == "log-linear":
                regr = linear_model.LinearRegression()
                self.Y[self.Y <= 0] = 1
                self.Y_fit = np.log(self.Y)
            regr.fit(self.X, self.Y_fit)
            self._regr = regr
            return regr

    @property
    def coefs(self):
        return self.regr.coef_

    @property
    def intercept(self):
        return self.regr.intercept_

    @property
    def num_factors(self):
        return self.X.shape[1]

    @property
    def population_size(self):
        return self.X.shape[0]

    @property
    def residuals(self):
        return self.Y_fit - np.matmul(self.X, self.coefs) - self.intercept

    @property
    def y_bar(self):
        try:
            return self._y_bar
        except AttributeError:
            self._y_bar = np.mean(self.Y)
            return self._y_bar

    @property
    def smearing_factor(self):
        try:
            return self._smearing_factor
        except AttributeError:
            self._smearing_factor = np.mean(np.exp(self.residuals))
            return self._smearing_factor

    def predict(self):
        if self.model == "linear":
            return np.matmul(self.X, self.coefs) + self.intercept
        if self.model == "log-linear":
            predict = np.matmul(self.X, self.coefs) + self.intercept
            return np.exp(predict) * self.smearing_factor

    def y_remain(self):
        return self.Y * np.mean(self.Y) / self.predict()

    @property
    def sse(self):
        return np.sum(np.multiply(self.residuals, self.residuals)) / (self.population_size - self.num_factors)

    @property
    def X_with_remain(self):
        remain = self.y_remain().reshape(-1, 1)
        return np.append(self.X, remain, axis=1)
