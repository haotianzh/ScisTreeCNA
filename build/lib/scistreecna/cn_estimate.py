import numpy as np
import itertools
from . import util


class CNEstimator:
    """
    Max parsimony for estimating total copy-number.
    """

    def __init__(self, cns):
        self.cns = cns
        self.max_cn = max(cns)
        self.min_cn = min(cns)
        self.range_cn = range(self.min_cn, self.max_cn + 1)
        self.traversor = util.TraversalGenerator()

    def __call__(self, tree):
        """
        tree should be 0-based numbered.
        """
        for node in self.traversor(tree, order="post"):
            if node.is_leaf():
                node.cn = self.cns[int(node.name)]
                node.mins = {}
                for c in self.range_cn:
                    node.mins[c] = abs(c - node.cn)
            else:
                node.mins = {}
                for c in self.range_cn:
                    mins = []
                    children = node.get_children()
                    for a, b in itertools.product(self.range_cn, self.range_cn):
                        mins.append(
                            children[0].mins[a]
                            + children[1].mins[b]
                            + abs(c - a)
                            + abs(c - b)
                        )
                    node.mins[c] = min(mins)
        return min(tree.root.mins.values())


if __name__ == "__main__":
    cns = [2, 2, 2, 3, 2, 1]
    tree = util.from_newick("(((0,1),2),(3,(4,5)));")
    estimator = CNEstimator(cns)

    print(estimator(tree))
