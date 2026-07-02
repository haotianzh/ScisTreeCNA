from collections import OrderedDict, defaultdict
from .node import Node
import pptree
import copy
import _pickle as cPickle


def deepcopy(obj):
    """Full deep copy via fast cPickle round-trip, falling back to copy.deepcopy.

    A true deep copy: duplicates everything attached to nodes, including GPU/CuPy
    or numpy arrays. Use struct_copy_tree for a cheap topology-only copy instead.
    """
    try:
        return cPickle.loads(cPickle.dumps(obj, -1))
    except Exception:
        return copy.deepcopy(obj)


class BaseTree(object):
    """
    A tree class, for extracting features from genealogical trees in the future.
    Arguments:
        args:
    # >>> tree = BaseTree();
    # >>> tree = popgen.utils.treeutils.from_newick('((1,2),(3,4));') # or
    """

    def __init__(self, root=None):
        """Wrap a root Node and build the id -> node index (`_nodes`)."""
        self.root = root
        self._update()

    def __contains__(self, node):
        # Membership test by Node or by identifier.
        # if a node is in this tree
        if isinstance(node, Node):
            node = node.identifier
        if node in self._nodes:
            return True
        else:
            return False

    def __getitem__(self, identifier) -> Node:
        # Look up a node by its identifier.
        # get a node from node list
        return self._nodes[identifier]

    def __len__(self):
        # Number of nodes in the tree.
        # obtain number of nodes in a tree
        return len(self._nodes)

    def __eq__(self, tree):
        # Topology equality: two trees are equal iff they induce the same set of
        # non-trivial bipartitions (splits). Splits are anchored on a shared leaf
        # so that complementary splits are compared consistently.
        anchor = tree[tree.get_leaves()[0]].name
        return tree.get_splits(True, anchor) == self.get_splits(True, anchor)

    def __hash__(self):
        # Hash by split set so equal topologies hash equally.
        return hash(self.get_splits(True))

    def __str__(self):
        """Newick (sorted) string representation of the tree."""
        return self.output()

    def _update(self):
        """(Re)build `_nodes` (id -> node) from the current root and descendants."""
        if self.root is not None:
            self._nodes = self.root.get_descendants_dict()
            self._nodes[self.root.identifier] = self.root

    def create_node(self, identifier=None, name=None, parent=None) -> Node:
        """Create a Node, attach it under `parent` (or as root), and return it."""
        node = Node(identifier=identifier, name=name)
        self.add_node(node, parent)
        return node

    def add_node(self, node, parent=None):
        """Add an existing node to the tree.

        With `parent=None` the node becomes the root (only if no root exists yet);
        otherwise it is linked as a child of `parent` (a Node or its identifier).
        Raises on duplicate ids, missing parent, or a pre-existing root.
        """
        if node.identifier in self._nodes:
            raise Exception("cannot add the node that has already been in the tree.")
        if parent is None:
            if self.root:
                raise Exception("root has already existed and parent cannot be none.")
            self.root = node
            self._nodes[node.identifier] = node
            # /* set root and make its level as 0 */
            node.set_parent(None)
            return
        pid = parent.identifier if isinstance(parent, Node) else parent
        if not pid in self._nodes:
            raise Exception("parent not found in this tree.")
        self._nodes[node.identifier] = node
        # /* link node with its parent */
        node.set_parent(self[pid])
        self[pid].add_child(node)
        return

    def get_all_nodes(self):
        """Return the id -> node dict of all nodes in the tree."""
        return self._nodes

    def get_leaves(self, return_label=False):
        """Return leaf identifiers (or names if `return_label`)."""
        leaves = [
            node.name if return_label else node.identifier
            for node in self.root.get_leaves()
        ]
        return leaves

    def get_splits(self, return_label=False, contains_leaf=None):
        """Return the set of non-trivial bipartitions (splits) of the leaf set.

        Each internal, non-root node induces a split = its leaf subset. Trivial
        splits (size <= 1 or covering all-but-one leaf) are skipped. If
        `contains_leaf` is given, splits not containing it are normalized to their
        complement so equality is anchor-consistent. Returns a frozenset of
        frozensets, using leaf names when `return_label` else identifiers.
        """
        def get_complement_split(split):
            # Leaves on the other side of the bipartition.
            return frozenset(filter(lambda x: x not in split, leaves))

        leaves = self.get_leaves(return_label=True)
        splits = set()
        for nid in self.get_all_nodes():
            # if it is root or a leaf which means trivial split, pass
            if not self._nodes[nid].is_leaf() and not self._nodes[nid].is_root():
                split = frozenset(
                    [
                        node.name if return_label else node.identifier
                        for node in self._nodes[nid].get_leaves()
                    ]
                )
                if 1 < len(split) < len(leaves) - 1:
                    if contains_leaf and contains_leaf not in split:
                        splits.add(get_complement_split(split))
                    else:
                        splits.add(split)
        return frozenset(splits)

    def to_dict(self):
        # return a dict for the whole tree.
        pass

    def output(
        self,
        output_format="newick_sorted",
        branch_length_func=None,
        confidence_func=None,
    ):
        """Serialize the tree to a Newick string.

        Args:
            output_format: "newick_sorted" (children sorted for a canonical string)
                or "newick" (children in stored order).
            branch_length_func: optional fn(node) -> branch length, appended as ":b".
            confidence_func: optional fn(node) -> confidence/support annotation.
        Returns the Newick string terminated by ';'.
        """
        # Recursively build Newick; a leaf is its own name, an internal node is
        # "(child,child,...)" with optional confidence and branch-length suffixes.
        def _newick_unsorted(node, branch_length_func, confidence_func):
            if node.is_leaf():
                if branch_length_func and confidence_func:
                    b = branch_length_func(node)
                    c = confidence_func(node)
                    return f"({node.name}){c}:{b}"
                elif branch_length_func:
                    b = branch_length_func(node)
                    return f"{node.name}:{b}"
                elif confidence_func:
                    c = confidence_func(node)
                    return f"({node.name}{c})"
                return node.name
            fstr = (
                "("
                + ",".join(
                    [
                        _newick_unsorted(child, branch_length_func, confidence_func)
                        for child in node.get_children()
                    ]
                )
                + ")"
                + (f"{confidence_func(node)}" if confidence_func else "")
                + (f":{branch_length_func(node)}" if branch_length_func else "")
            )
            return fstr

        # Same recursion as above, but children substrings are sorted so the
        # output is canonical (independent of child insertion order).
        def _newick_sorted(node, branch_length_func, confidence_func):
            if node.is_leaf():
                if branch_length_func and confidence_func:
                    b = branch_length_func(node)
                    c = confidence_func(node)
                    return f"({node.name}){c}:{b}"
                elif branch_length_func:
                    b = branch_length_func(node)
                    return f"{node.name}:{b}"
                elif confidence_func:
                    c = confidence_func(node)
                    return f"({node.name}{c})"
                return node.name
            newick_children = []
            for child in node.get_children():
                newick_children.append(
                    _newick_sorted(child, branch_length_func, confidence_func)
                )
            fstr = (
                "("
                + ",".join(sorted(newick_children))
                + ")"
                + (f"{confidence_func(node)}" if confidence_func else "")
                + (f":{branch_length_func(node)}" if branch_length_func else "")
            )
            return fstr

        def newick():
            return (
                _newick_unsorted(self.root, branch_length_func, confidence_func) + ";"
            )

        def newick_sorted():
            return _newick_sorted(self.root, branch_length_func, confidence_func) + ";"

        funcs = {"newick": newick, "newick_sorted": newick_sorted}
        return funcs[output_format]()

    def draw(self, attr=None, **kwargs):
        """Pretty-print the tree to the console using pptree.

        If `attr` is given, that node attribute is stringified and used as the
        printed label; extra kwargs are forwarded to pptree.print_tree.
        """
        if attr is not None:
            for node in self.get_all_nodes():
                self[node].__setattr__(attr, str(self[node].__getattribute__(attr)))
            pptree.print_tree(self.root, "_children", nameattr=attr, **kwargs)
        else:
            pptree.print_tree(self.root, "_children", **kwargs)

    def copy(self):
        """Full deep copy of the tree, INCLUDING attached GPU/numpy arrays.

        Uses pickle-based deepcopy, so every per-node array (U, Q, etc.) is
        duplicated. Correct but expensive; prefer struct_copy for NNI candidates.
        """
        return deepcopy(self)

    def struct_copy(self):
        """Lightweight copy: only tree structure (nodes, parent-child links, name, identifier).
        Does NOT copy CuPy/numpy arrays attached to nodes (U, U_, _U, Q, etc.).
        Much faster than deepcopy for trees that will be used for evaluation.

        This speed difference is what makes generating many NNI candidate trees
        cheap: only the topology is cloned, and the large GPU arrays are recomputed
        on demand rather than copied."""
        return struct_copy_tree(self)


def struct_copy_tree(tree):
    """Standalone lightweight tree copy that works with any BaseTree-like object.
    Only copies tree structure (nodes, parent-child links, name, identifier, branch).
    Does NOT copy CuPy/numpy arrays attached to nodes.

    Why: NNI search clones a tree's topology many times to enumerate neighbor
    trees. Pickle-deepcopy would also copy the large per-node GPU arrays (U/Q/...),
    which is slow and unnecessary since those are recomputed for each candidate.
    This builds bare Node objects via __new__ (bypassing __init__) and rebuilds
    links in two passes, leaving array attributes absent."""
    old_to_new = {}
    node_items = tree._nodes.items()
    _Node_new = Node.__new__

    # First pass: create all new nodes (minimal attribute setup)
    for nid, old_node in node_items:
        new_node = _Node_new(Node)
        new_node._identifier = old_node._identifier
        new_node.name = old_node.name
        new_node._branch = old_node._branch
        new_node.parent = None
        new_node.children = OrderedDict()
        new_node._children = []
        old_to_new[nid] = new_node

    # Second pass: now that every node exists, wire up parent and child links by
    # mapping each old node's neighbors to their freshly created counterparts.
    _otn_get = old_to_new.get
    for nid, old_node in node_items:
        new_node = old_to_new[nid]
        par = old_node.parent
        if par is not None and hasattr(par, 'identifier'):
            new_parent = _otn_get(par.identifier)
            if new_parent is not None:
                new_node.parent = new_parent
        for child in old_node._children:
            cid = child.identifier
            new_child = _otn_get(cid)
            if new_child is not None:
                new_node.children[cid] = new_child
                new_node._children.append(new_child)

    # Build new tree directly (bypass __init__/_update since _nodes is ready).
    new_tree = BaseTree.__new__(BaseTree)
    new_tree.root = old_to_new[tree.root.identifier]
    new_tree._nodes = old_to_new
    return new_tree
