import numpy as np
from sklearn.linear_model import LogisticRegression

class ShrinkageLogistic:

    def __init__(self, l1_ratio):
        self.l1_ratio = l1_ratio

        if self.l1_ratio == 1:
            self.solver = "saga"
        else:
            self.solver = "lbfgs"


    def select_lambda(
            self,
            lambda_grid,
            X_train,
            y_train,
            w_train,
            X_val,
            y0_val,
            y1_val
    ):
        welfare_grid = np.zeros(len(lambda_grid))

        for idx in range(len(lambda_grid)):

            lam = lambda_grid[idx]

            C = 1.0 / lam
            clf = LogisticRegression(
                l1_ratio = self.l1_ratio,
                C = C,
                solver = self.solver,
                fit_intercept = False,
                max_iter = 2000
            )
            clf.fit(X_train, y_train, sample_weight = w_train)
            pred = np.sign(X_val @ clf.coef_.reshape(-1,1))
            welfare_grid[idx] = np.mean( (1+pred)/2 * y1_val + (1-pred)/2 * y0_val )

        idx = np.argmax(welfare_grid)
        best_lambda = lambda_grid[idx]

        self.best_lambda_ = best_lambda


    def fit(self, X_train, y_train, w_train):
        C = 1.0 / self.best_lambda_
        self.model_ = LogisticRegression(
            l1_ratio = self.l1_ratio,
            C = C,
            solver = self.solver,
            fit_intercept = False,
            max_iter = 2000
        )
        self.model_.fit(X_train, y_train, sample_weight = w_train)


