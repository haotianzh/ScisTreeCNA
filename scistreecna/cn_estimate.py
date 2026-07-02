import numpy as np
import itertools
from . import util


class CNEstimator:
    """
    Max-parsimony estimate of the minimum number of total-copy-number changes
    needed to explain observed leaf copy numbers on a tree.

    Used to calibrate the LAMBDA_C (CNM) rate. cns is the per-leaf observed
    total copy number, indexed by integer leaf name.
    """

    def __init__(self, cns):
        self.cns = cns
        self.max_cn = max(cns)
        self.min_cn = min(cns)
        self.range_cn = range(self.min_cn, self.max_cn + 1)
        self.traversor = util.TraversalGenerator()

    def __call__(self, tree):
        """
        Run the post-order Sankoff DP and return the minimum total CN-change cost.

        tree should be 0-based numbered (leaf names index into self.cns).
        node.mins[c] holds the minimum cost to reconcile this node's subtree
        assuming the node has total copy number c.
        """
        # Post-order so every child's mins[] is ready before its parent.
        for node in self.traversor(tree, order="post"):
            if node.is_leaf():
                # Leaf: cost to assign CN c is the distance from the observed CN.
                node.cn = self.cns[int(node.name)]
                node.mins = {}
                for c in self.range_cn:
                    node.mins[c] = abs(c - node.cn)
            else:
                # Internal (binary) node: for each candidate CN c, pick child CNs
                # a, b minimizing child subtree costs plus the |c-a|, |c-b| edge
                # changes along the two branches.
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
        # Best root assignment gives the overall minimum number of CN changes.
        return min(tree.root.mins.values())
