
# License: GPL
# Author: Rubén Rodriguez (rrunix). Original implementation by Alexandre Passos in python 2.7 (https://gist.github.com/alextp)


from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from collections.abc import Iterable
import itertools as it
import math

import numpy as np
from numpy import array as A, dtype
from numpy import sign
from numpy import zeros as Z

from sklearn.utils.validation import check_X_y, check_array
from sklearn.base import BaseEstimator, ClassifierMixin


def poly(degree):
    def kernel(a, b):
        norm = np.sqrt(np.dot(a, a)*np.dot(b, b))
        if norm == 0.:
            return 0.
        return ((1+np.dot(a, b)/norm)**degree)
    return kernel


def rbf(var):
    def kernel(a, b):
        d = a-b
        return math.exp(-var*np.dot(d, d))
    return kernel


class LaSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma=0.1, degree=4, tau=1e-3, eps=0.001, verbose=0):
        self.S = []
        self.a = []
        self.g = []
        self.y = []
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.tau = tau
        self.eps = eps
        self.b = 0
        self.delta = 0
        self.i = 0
        self.misses = 0
        self.verbose = verbose

        if kernel == 'rbf':
            self.k = rbf(self.gamma)
        elif kernel == 'poly':
            self.k = poly(self.degree)
        else:
            raise NotImplemented(f"Kernel not supported {kernel}")

    def _A(self, i):
        return min(0, self.C*self.y[i])

    def _B(self, i):
        return max(0, self.C*self.y[i])

    def _tau_violating(self, i, j):
        return ((self.a[i] < self._B(i)) and
                (self.a[j] > self._A(j)) and
                (self.g[i] - self.g[j] > self.tau))

    def _extreme_ij(self):
        S = self.S
        i = np.argmax(list((self.g[i] if self.a[i] < self._B(i) else -np.inf)
                           for i in range(len(S))))
        j = np.argmin(list((self.g[i] if self.a[i] > self._A(i) else np.inf)
                           for i in range(len(S))))
        return i, j

    def _lbda(self, i, j):
        S = self.S
        l = min((self.g[i]-self.g[j])/(self.k(S[i], S[i])+self.k(S[j], S[j])-self.k(S[i], S[j])),
                self._B(i)-self.a[i],
                self.a[j]-self._A(j))
        self.a[i] += l
        self.a[j] -= l
        for s in range(len(S)):
            self.g[s] -= l*(self.k(S[i], S[s])-self.k(S[j], S[s]))
        return l

    def _lasvm_process(self, v, cls, w):
        self.S.append(v)
        self.a.append(0)
        self.y.append(cls)
        self.g.append(cls - self._target_function(v))
        if cls > 0:
            i = len(self.S)-1
            _, j = self._extreme_ij()
        else:
            j = len(self.S)-1
            i, _ = self._extreme_ij()
        if not self._tau_violating(i, j):
            return
        S = self.S
        lbda = self._lbda(i, j)

    def _lasvm_reprocess(self):
        S = self.S
        i, j = self._extreme_ij()
        if not self._tau_violating(i, j):
            return
        lbda = self._lbda(i, j)
        i, j = self._extreme_ij()
        to_remove = []
        for s in range(len(S)):
            if self.a[s] < self.eps:
                to_remove.append(s)
        for s in reversed(to_remove):
            del S[s]
            del self.a[s]
            del self.y[s]
            del self.g[s]
        i, j = self._extreme_ij()
        self.b = (self.g[i]+self.g[j])/2.
        self.delta = self.g[i]-self.g[j]

    def predict(self, X):
        check_array(X)
        y_hat = self.decision_function(X)
        return np.where(y_hat >= 0, 1, 0)

    def decision_function(self, X):
        return np.apply_along_axis(self._target_function, axis=1, arr=X) + self.b

    def _target_function(self, v):
        return sum(self.a[i]*self.k(self.S[i], v) for i in range(len(self.S)))

    def fit(self, X, y, class_weights=None):
        check_X_y(X, y)
        
        if class_weights is not None:
            if isinstance(X, Iterable):
                check_X_y(X, class_weights)
            else:
                class_weights = np.full_like(y, class_weights, dtype=np.float32)

        for _x, _y in zip(X, y):
            self.update(_x, _y)

    def update(self, x, y, w=1.0):
        if y == 0:
            y = -1

        if len(self.S) < 10:
            self.S.append(x)
            self.y.append(y)
            self.a.append(y)
            self.g.append(0)
            for i in range(len(self.S)):
                self.g[i] = self.y[i]-self._target_function(self.S[i])
        else:
            if y*(self._target_function(x) + self.b) < 0:
                self.misses += 1

            self.i += 1
            self._lasvm_process(x, y, w)
            self._lasvm_reprocess()
            self._lasvm_reprocess()
            if self.i % 1000 == 0:
                self.misses = 0
                
                if self.verbose > 0:
                    print("m", self.misses, "s", len(self.S))
