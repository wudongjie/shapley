import numpy as np
from numpy import empty, uint8
from math import factorial
import c_improve
from c_improve import gini


def perms(n):
    """
    output an array which enclude all sets of permutation exhaustively
    """
    f = 1
    p = empty((2 * n - 1, factorial(n)), uint8)
    for i in range(n):
        p[i, :f] = i
        p[i + 1: 2 * i + 1, :f] = p[:i, :f]
        for j in range(i):
            p[:i + 1, f * (j + 1):f * (j + 2)] = p[j + 1:j + i + 2, :f]
        f = f * (i + 1)
    return p[:n, :]


def perms_steps(n, s):
    """
    output an array which randomly select s permutation.
    """
    p = empty((s, n), uint8)

    for i in range(s):
        k = np.random.permutation(n)
        while (k == p).all(1).any():
            k = np.random.permutation(n)
        p[i, :] = k[:n]
    assert p.shape == (s, n), "The shape doesn't match!"
    return np.copy(p[:s, :].T, order='C')


class Shapley:
    def __init__(self, x_index, model=None, data=None, coefs=None, method="exhausted",
                 steps=0, verbose=True, relative=True):
        """
        model: your model
        method: "exhausted": compute Shapley values using all steps
                "sampling": compute Shapley values using sampled steps
        steps: identify the steps for the computation
        verbose: speak out the computation procedure
        relative: whether the Shapley value is printed out in percentage
        """
        self.x_index = x_index
        if model:
            self.model = model
            self.intercept = model.intercept
            self.df_y = model.Y
            self.df_x = model.X_with_remain
            self.coefs = self.model.coefs
            self.num_factors = self.model.num_factors + 1
        elif data and coef:
            self.intercept = coefs[-1]
            self.df_y = data.iloc[:, 0].values
            self.df_x = data.iloc[:, 1:].values
            self.coefs = coefs[:-1]
            self.num_factors = self.df_x.shape[1]
        self.method = method
        self.relative = relative
        self.verbose = verbose
        if steps <= factorial(self.num_factors):
            self.steps = steps
        else:
            print("Steps exceeds the maximum steps!")
            self.steps = factorial(self.num_factors)
        self.df_new = np.copy(self.df_x)


    @property
    def mc_dist(self):
        try:
            return self._mc_dist
        except AttributeError:
            self._mc_dist = self.compute_mc(self.num_factors)
        return self._mc_dist

    @property
    def gini_y(self):
        try:
            return self._gini
        except AttributeError:
            self._gini = gini(self.df_y)
            return self._gini

    @property
    def mc_converge(self):
        cummean = np.cumsum(self.mc_dist) / np.arange(1, self.mc_dist.shape[0] + 1)
        return cummean

    @property
    def shapley_value(self):
        if self.relative:
            return np.mean(self.mc_dist) / self.gini_y
        return np.mean(self.mc_dist)

    @property
    def shapley_value_ci(self):
        ci = 1.96 * np.std(self.mc_dist) / np.sqrt(self.mc_dist.shape[0])
        mean = np.mean(self.mc_dist)
        if self.relative:
            return (mean - ci, mean + ci) / self.gini_y
        else:
            return (mean - ci, mean + ci)


    def transform_df(self, permu_list, x_index, x_include=True):
        self.df_new = np.copy(self.df_x, order='C')
        x_value = permu_list[x_index]
        if x_include:
            parray = permu_list >= x_value
        else:
            parray = permu_list > x_value
        c_improve.mean_c(self.df_new, parray)

    def prediction(self, X):
        if self.model.model == 'log-linear':
            return np.exp(np.matmul(X, self.coefs) + self.intercept) * self.model.smearing_factor
        else:
            return np.matmul(X, self.coefs) + self.intercept

    def compute_mc(self, num_factors):
        mc_dist = np.array([])
        if self.method == "exhausted":
            self.steps = factorial(num_factors)
            permu_set = perms(num_factors)
        else:
            permu_set = perms_steps(num_factors, self.steps)
        permu_set = np.random.permutation(permu_set.T).T
        num_steps = 0
        for i in np.nditer(permu_set, flags=['external_loop'], order='F'):
            self.transform_df(i, self.x_index, x_include=False)
            obs_y = self.prediction(self.df_new[:, :-1]) * self.df_new[:, -1] / self.model.y_bar
            self.transform_df(i, self.x_index, x_include=True)
            fix_y = self.prediction(self.df_new[:, :-1]) * self.df_new[:, -1] / self.model.y_bar
            single_mc = gini(obs_y) - gini(fix_y)
            mc_dist = np.append(mc_dist, single_mc)
            if self.verbose and (num_steps % 2000 == 0):
                print("Margin Distribution for Step {0} is: {1}".format(num_steps, np.mean(mc_dist)))
            num_steps += 1
        return mc_dist

