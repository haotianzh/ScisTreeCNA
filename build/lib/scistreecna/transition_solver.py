"""
No Mutation
"""

import numpy as np
import scipy as scp
import pandas as pd

# import popgen
# CN_MAX = 4
# CN_MIN = 2
# LAMBDA_C = 1
# LAMBDA_S = 1
# LAMBDA_T = 4
# CNV_TIMEOUT  = LAMBDA_C / (LAMBDA_C + LAMBDA_T)
# CNV_SNV = LAMBDA_C / (LAMBDA_C + LAMBDA_S)
# N = int((CN_MAX-CN_MIN+1) * (CN_MAX+CN_MIN+2) / 2)


class TransitionProbability:
    def __init__(
        self,
        CN_MAX=2,
        CN_MIN=0,
        LAMBDA_C=1,
        LAMBDA_S=1,
        LAMBDA_T=4,
        log_transform=False,
    ):
        self.CN_MAX = CN_MAX
        self.CN_MIN = CN_MIN
        self.LAMBDA_C = LAMBDA_C
        self.LAMBDA_S = LAMBDA_S
        self.LAMBDA_T = LAMBDA_T
        self.CNV_TIMEOUT = LAMBDA_C / (LAMBDA_C + LAMBDA_T)
        self.CNV_SNV = LAMBDA_C / (LAMBDA_C + LAMBDA_S)
        self.N = int((CN_MAX - CN_MIN + 1) * (CN_MAX + CN_MIN + 2) / 2)
        self.log_transform = log_transform
        self._indexing()

    def valid(self, g0, g1):
        return min(g0, g1) >= 0 and self.CN_MIN <= g0 + g1 <= self.CN_MAX

    def _indexing(self):
        state2index = {}
        index2state = {}
        for i in range(0, self.CN_MAX + 1):
            for j in range(0, self.CN_MAX + 1):
                if self.valid(i, j):
                    index = int(
                        (i + j) * (i + j + 1) / 2
                        + i
                        - int((self.CN_MIN) * (self.CN_MIN + 1) / 2)
                    )
                    state2index[(i, j)] = index
                    index2state[index] = (i, j)
        self.state2index = state2index
        self.index2state = index2state
        # print(self.state2index)

    """
        Index for state
    """

    def index_gt(self, i, j):
        return self.state2index[(i, j)]

    """
        CN profie at index
    """

    def cn_profile_at_index(self, index):
        return self.index2state[index]

    """
        Index for transition matrix
    """

    def index(self, i, j, p, q):
        # use yang's tri indexing
        i1 = self.index_gt(i, j)
        i2 = self.index_gt(p, q)
        return int(i1 * self.N + i2)

    def format(self, mat):
        names = [self.cn_profile_at_index(i) for i in range(self.N)]
        df = pd.DataFrame(mat, columns=names, index=names)
        return df

    # def index(self, i, j, p ,q):
    #     # use yang's tri indexing
    #     i1 = (i+j) * (i+j+1) / 2 + i - int((self.CN_MIN) * (self.CN_MIN + 1) / 2)
    #     i2 = (p+q) * (p+q+1) / 2 + p - int((self.CN_MIN) * (self.CN_MIN + 1) / 2)
    #     return int(i1*self.N + i2)

    # def index_gt(self, i, j):
    #     return int((i+j) * (i+j+1) / 2 + i - int((self.CN_MIN) * (self.CN_MIN + 1) / 2))

    def allow_increase(self, g0, g1):
        return g0 + g1 < self.CN_MAX and g0 + g1 >= 1

    def allow_decrease(self, g0, g1):
        return g0 + g1 > self.CN_MIN

    def valid(self, g0, g1):
        return min(g0, g1) >= 0 and self.CN_MIN <= g0 + g1 <= self.CN_MAX

    def identity(self, g0, g1, g0_, g1_):
        return g0 == g0_ and g1 == g1_

    def equation_no_mutation(self):
        eqs = []
        vals = []
        for g0 in range(self.CN_MAX + 1):
            for g1 in range(self.CN_MAX + 1):
                if not self.valid(g0, g1):
                    continue
                for g0_ in range(self.CN_MAX + 1):
                    for g1_ in range(self.CN_MAX + 1):
                        if not self.valid(g0_, g1_):
                            continue
                        eq = np.zeros(self.N * self.N, dtype=float)
                        if not self.allow_increase(g0, g1) and not self.allow_decrease(
                            g0, g1
                        ):
                            eq[self.index(g0, g1, g0_, g1_)] = 1
                            val = self.identity(g0, g1, g0_, g1_)
                            eqs.append(eq)
                            vals.append(val)
                            continue
                        p = self.allow_increase(g0, g1) / (
                            self.allow_increase(g0, g1) + self.allow_decrease(g0, g1)
                        )
                        eq[self.index(g0, g1, g0_, g1_)] = -1
                        if self.valid(g0 + 1, g1):
                            eq[self.index(g0 + 1, g1, g0_, g1_)] = (
                                p * self.CNV_TIMEOUT * (g0 / (g0 + g1))
                            )
                        if self.valid(g0, g1 + 1):
                            eq[self.index(g0, g1 + 1, g0_, g1_)] = (
                                p * self.CNV_TIMEOUT * (g1 / (g0 + g1))
                            )
                        if self.valid(g0 - 1, g1):
                            eq[self.index(g0 - 1, g1, g0_, g1_)] = (
                                (1 - p) * self.CNV_TIMEOUT * (g0 / (g0 + g1))
                            )
                        if self.valid(g0, g1 - 1):
                            eq[self.index(g0, g1 - 1, g0_, g1_)] = (
                                (1 - p) * self.CNV_TIMEOUT * (g1 / (g0 + g1))
                            )
                        val = -self.identity(g0, g1, g0_, g1_) * (1 - self.CNV_TIMEOUT)
                        eqs.append(eq)
                        vals.append(val)
        return np.array(eqs), np.array(vals)

    # def equation_mutation(self):
    #     x = self.solve_no_mutation(verbose=False)
    #     eqs = []
    #     vals = []
    #     for g0 in range(self.CN_MAX+1):
    #         for g1 in range(self.CN_MAX+1):
    #             if not self.valid(g0, g1):
    #                 continue
    #             for g0_ in range(self.CN_MAX+1):
    #                 for g1_ in range(self.CN_MAX+1):
    #                     if not self.valid(g0_, g1_):
    #                         continue
    #                     eq = np.zeros(self.N*self.N, dtype=float)
    #                     if g1 > 0:
    #                         eq[self.index(g0, g1, g0_, g1_)] = 1
    #                         val = 0
    #                         eqs.append(eq)
    #                         vals.append(val)
    #                         continue
    #                     if g0 == 0:
    #                         eq[self.index(g0, g1, g0_, g1_)] = 1
    #                         val = self.identity(g0, g1, g0_, g1_)
    #                         # val = 0
    #                         eqs.append(eq)
    #                         vals.append(val)
    #                         continue
    #                     if not self.allow_increase(g0, g1) and not self.allow_decrease(g0, g1):
    #                         eq[self.index(g0, g1, g0_, g1_)] = 1
    #                         val = self.identity(g0-1, g1+1, g0_, g1_)
    #                         eqs.append(eq)
    #                         vals.append(val)
    #                         continue
    #                     p = self.allow_increase(g0, g1) / (self.allow_increase(g0, g1) + self.allow_decrease(g0, g1))
    #                     eq[self.index(g0, g1, g0_, g1_)] = -1
    #                     if self.valid(g0+1, g1):
    #                         # eq[self.index(g0+1, g1, g0_, g1_)] = self.CNV_TIMEOUT * self.CNV_SNV * p
    #                         eq[self.index(g0+1, g1, g0_, g1_)] = self.CNV_SNV * p
    #                     if self.valid(g0-1, g1):
    #                         # eq[self.index(g0-1, g1, g0_, g1_)] = self.CNV_TIMEOUT * self.CNV_SNV * (1-p)
    #                         eq[self.index(g0-1, g1, g0_, g1_)] = self.CNV_SNV * (1-p)
    #                     # val = -self.CNV_TIMEOUT * (1 - self.CNV_SNV) * x[self.index(g0-1, g1, g0_, g1_)] -(1-self.CNV_TIMEOUT) * self.identity(g0-1, g1+1, g0_, g1_)
    #                     val = -(1-self.CNV_SNV) * x[self.index(g0-1, g1+1, g0_, g1_)]
    #                     eqs.append(eq)
    #                     vals.append(val)
    #     return np.array(eqs), np.array(vals)

    def equation_mutation(self):
        x = self.solve_no_mutation(return_matrix=False, verbose=False)
        eqs = []
        vals = []
        for g0 in range(self.CN_MAX + 1):
            for g1 in range(self.CN_MAX + 1):
                if not self.valid(g0, g1):
                    continue
                for g0_ in range(self.CN_MAX + 1):
                    for g1_ in range(self.CN_MAX + 1):
                        if not self.valid(g0_, g1_):
                            continue
                        eq = np.zeros(self.N * self.N, dtype=float)
                        if g1 > 0 or g0 == 0:
                            eq[self.index(g0, g1, g0_, g1_)] = 1
                            val = 0
                            eqs.append(eq)
                            vals.append(val)
                            continue

                        if (
                            not self.allow_increase(g0, g1)
                            and not self.allow_decrease(g0, g1)
                            and g0 > 1
                        ):
                            eq[self.index(g0, g1, g0_, g1_)] = 1
                            val = self.identity(g0 - 1, g1 + 1, g0_, g1_)
                            eqs.append(eq)
                            vals.append(val)
                            continue

                        if g0 == 1:
                            # print(g0, g1, g0_, g1_)
                            eq[self.index(g0, g1, g0_, g1_)] = -1
                            if self.valid(g0 + 1, g1):
                                eq[self.index(g0 + 1, g1, g0_, g1_)] = self.CNV_SNV
                            val = -(1 - self.CNV_SNV) * x[self.index(0, 1, g0_, g1_)]
                            eqs.append(eq)
                            vals.append(val)
                            continue

                        p = self.allow_increase(g0, g1) / (
                            self.allow_increase(g0, g1) + self.allow_decrease(g0, g1)
                        )
                        eq[self.index(g0, g1, g0_, g1_)] = -1
                        if self.valid(g0 + 1, g1):
                            eq[self.index(g0 + 1, g1, g0_, g1_)] = self.CNV_SNV * p
                        if self.valid(g0 - 1, g1):
                            eq[self.index(g0 - 1, g1, g0_, g1_)] = self.CNV_SNV * (
                                1 - p
                            )
                        val = (
                            -(1 - self.CNV_SNV)
                            * x[self.index(g0 - 1, g1 + 1, g0_, g1_)]
                        )
                        eqs.append(eq)
                        vals.append(val)
        return np.array(eqs), np.array(vals)

    def solve_no_mutation(self, return_matrix=True, verbose=True):
        eqs, vals = self.equation_no_mutation()
        x = scp.linalg.solve(eqs, vals)  # using numpy.linalg will cause segfault
        x_ = np.abs(np.round(x, 2))
        if verbose:
            print("P(CNV before TIMEOUT): ", self.CNV_TIMEOUT)
            for g0 in range(self.CN_MAX + 1):
                for g1 in range(self.CN_MAX + 1):
                    if not self.valid(g0, g1):
                        continue
                    ss = 0
                    for g0_ in range(self.CN_MAX + 1):
                        for g1_ in range(self.CN_MAX + 1):
                            if not self.valid(g0_, g1_):
                                continue
                            print(
                                f"({g0},{g1})->({g0_},{g1_}) = ",
                                x_[self.index(g0, g1, g0_, g1_)],
                            )
                            ss += x_[self.index(g0, g1, g0_, g1_)]
                    print(ss)
        return self.to_matrix(x) if return_matrix else x

    def solve_mutation(self, return_matrix=True, verbose=True):
        eqs, vals = self.equation_mutation()
        x = scp.linalg.solve(eqs, vals)
        x_ = np.abs(np.round(x, 2))
        if verbose:
            print("P(CNV before TIMEOUT): ", self.CNV_TIMEOUT)
            print("P(CNV before SNV): ", self.CNV_SNV)
            for g0 in range(self.CN_MAX + 1):
                for g1 in range(self.CN_MAX + 1):
                    if not self.valid(g0, g1):
                        continue
                    ss = 0
                    for g0_ in range(self.CN_MAX + 1):
                        for g1_ in range(self.CN_MAX + 1):
                            if not self.valid(g0_, g1_):
                                continue
                            print(
                                f"({g0},{g1})->({g0_},{g1_}) = ",
                                x_[self.index(g0, g1, g0_, g1_)],
                            )
                            ss += x[self.index(g0, g1, g0_, g1_)]
                    print(ss)
        return self.to_matrix(x) if return_matrix else x

    def to_matrix(self, x):
        mat = np.zeros([self.N, self.N])
        for g0 in range(self.CN_MAX + 1):
            for g1 in range(self.CN_MAX + 1):
                if not self.valid(g0, g1):
                    continue
                for g0_ in range(self.CN_MAX + 1):
                    for g1_ in range(self.CN_MAX + 1):
                        if not self.valid(g0_, g1_):
                            continue
                        # print(self.index_gt(g0, g1), self.index_gt(g0_, g1_), x[self.index(g0, g1, g0_, g1_)])
                        val = x[self.index(g0, g1, g0_, g1_)]
                        val = 0 if val < 0 else val
                        mat[self.index_gt(g0, g1), self.index_gt(g0_, g1_)] = (
                            np.log(val) if self.log_transform else val
                        )
        return mat


if __name__ == "__main__":

    np.set_printoptions(suppress=True)
    prob = TransitionProbability(
        log_transform=True,
        LAMBDA_C=1000,
        LAMBDA_S=1,
        LAMBDA_T=2 * 100 - 1,
        CN_MAX=5,
        CN_MIN=1,
    )
    x = prob.solve_no_mutation(verbose=False)
    print(prob.format(np.exp(x)))
