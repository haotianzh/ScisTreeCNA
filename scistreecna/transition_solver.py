"""
Transition probabilities solver
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
    """
    Builds the JSC-model genotype transition matrices by solving a linear system.

    Each genotype state is a pair (g0, g1) = (#wild-type-base copies,
    #mutant-base copies); total copy number g0+g1 is bounded to [CN_MIN, CN_MAX],
    giving N valid states. The model's branch events are:
      - CNM (copy-number mutation): gain/loss of one copy, chosen uniformly among
        existing copies; gated by allow_increase/allow_decrease.
      - PM  (point mutation): the single infinite-sites mutation (one wild-type
        copy g0 -> mutant g1), carried only by solve_mutation's branch.
      - timeout: absorbing event encoding branch length.
    Rate parameters LAMBDA_C/S/T yield CNV_TIMEOUT = P(a CNM occurs before the
    timeout) and CNV_SNV = P(a CNM occurs before the point mutation).

    Produces two NxN matrices via solve_no_mutation (tau, branch with no PM) and
    solve_mutation (tau_M, the branch carrying the single PM). Set log_transform
    to emit log-probabilities.
    """

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
        """True if (g0, g1) is a non-negative genotype with total CN in [CN_MIN, CN_MAX]."""
        return min(g0, g1) >= 0 and self.CN_MIN <= g0 + g1 <= self.CN_MAX

    def _indexing(self):
        """Build the (g0, g1) <-> flat-index maps using triangular indexing.

        Populates self.state2index / self.index2state so every valid genotype
        gets a unique index in [0, N).
        """
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
        """Flat state index for genotype (i, j)."""
        return self.state2index[(i, j)]

    """
        CN profie at index
    """

    def cn_profile_at_index(self, index):
        """Inverse of index_gt: the (g0, g1) genotype at a flat state index."""
        return self.index2state[index]

    """
        Index for transition matrix
    """

    def index(self, i, j, p, q):
        """Flat index into the N*N solution vector for transition (i, j) -> (p, q)."""
        # row-major: source state i1 times N plus destination state i2
        i1 = self.index_gt(i, j)
        i2 = self.index_gt(p, q)
        return int(i1 * self.N + i2)

    def format(self, mat):
        """Wrap an NxN matrix as a DataFrame labelled by (g0, g1) genotypes."""
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
        """True if a copy can be gained: total CN below CN_MAX and at least one copy present."""
        return g0 + g1 < self.CN_MAX and g0 + g1 >= 1

    def allow_decrease(self, g0, g1):
        """True if a copy can be lost: total CN above CN_MIN."""
        return g0 + g1 > self.CN_MIN

    def valid(self, g0, g1):
        """True if (g0, g1) is a non-negative genotype with total CN in [CN_MIN, CN_MAX]."""
        return min(g0, g1) >= 0 and self.CN_MIN <= g0 + g1 <= self.CN_MAX

    def identity(self, g0, g1, g0_, g1_):
        """Indicator that source (g0, g1) equals destination (g0_, g1_)."""
        return g0 == g0_ and g1 == g1_

    def equation_no_mutation(self):
        """Assemble the linear system for tau (branch with NO point mutation).

        Returns (eqs, vals): a flattened (N*N) x (N*N) coefficient matrix and
        right-hand side. Each equation expresses one transition probability
        T(g0,g1 -> g0_,g1_) as a recurrence over CNM moves: starting at (g0,g1)
        the chain either (a) times out and stays put (the identity / RHS term),
        or (b) undergoes one copy-number gain/loss (with prob CNV_TIMEOUT) into a
        neighbouring state and continues from there. p is the conditional
        probability of a gain (vs loss); the per-copy factor g0/(g0+g1) or
        g1/(g0+g1) picks which base copy is gained/lost. Solving the system gives
        all transition probabilities simultaneously.
        """
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
                        # Absorbing state (no CNM move possible): T = identity.
                        if not self.allow_increase(g0, g1) and not self.allow_decrease(
                            g0, g1
                        ):
                            eq[self.index(g0, g1, g0_, g1_)] = 1
                            val = self.identity(g0, g1, g0_, g1_)
                            eqs.append(eq)
                            vals.append(val)
                            continue
                        # p = conditional prob of a gain given a CNM happens.
                        p = self.allow_increase(g0, g1) / (
                            self.allow_increase(g0, g1) + self.allow_decrease(g0, g1)
                        )
                        # Recurrence: -T(here) + sum over neighbours reached by one
                        # CNM (weight = direction prob * CNV_TIMEOUT * per-copy share).
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
                        # RHS: prob of timing out immediately and staying put.
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
        """Assemble the linear system for tau_M (branch carrying the single PM).

        Same recurrence style as equation_no_mutation, but this branch must fire
        the one infinite-sites point mutation (one wild-type copy -> mutant), and
        CNMs race the PM with prob CNV_SNV. The RHS couples to x, the already
        solved no-mutation transitions, at the post-PM state (g0-1, g1+1).
        Returns (eqs, vals) for scipy.linalg.solve.
        """
        # x: no-mutation transition probabilities, reused on the RHS below.
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
                        # PM can only fire from a pure-wild-type state with a
                        # copy to mutate; otherwise this entry is forced to 0.
                        if g1 > 0 or g0 == 0:
                            eq[self.index(g0, g1, g0_, g1_)] = 1
                            val = 0
                            eqs.append(eq)
                            vals.append(val)
                            continue

                        # No CNM possible: PM fires immediately, moving to (g0-1, g1+1).
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

                        # Single wild-type copy: PM (prob 1-CNV_SNV) lands at
                        # (0,1); otherwise a CNM gain happens first.
                        if g0 == 1:
                            # print(g0, g1, g0_, g1_)
                            eq[self.index(g0, g1, g0_, g1_)] = -1
                            if self.valid(g0 + 1, g1):
                                eq[self.index(g0 + 1, g1, g0_, g1_)] = self.CNV_SNV
                            val = -(1 - self.CNV_SNV) * x[self.index(0, 1, g0_, g1_)]
                            eqs.append(eq)
                            vals.append(val)
                            continue

                        # General case: CNM (prob CNV_SNV, split gain/loss by p)
                        # may precede the PM; with prob 1-CNV_SNV the PM fires and
                        # the chain continues as no-mutation from (g0-1, g1+1).
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
        """Solve for tau, the no-point-mutation transition probabilities.

        Returns the NxN matrix (or the raw flat solution vector if
        return_matrix is False). verbose prints each transition and row sums.
        """
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
        """Solve for tau_M, the transitions on the branch carrying the single PM.

        Returns the NxN matrix (or the raw flat solution vector if
        return_matrix is False). verbose prints each transition and row sums.
        """
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
        """Reshape the flat solution x into an NxN transition matrix.

        Rows are source states, columns destination states. Tiny negative
        round-off values are clamped to 0; entries are log-transformed when
        self.log_transform is set.
        """
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
