from . import util


def batch_topological_sort(trees, order="up"):
    """Merge per-tree layers across many trees (the "FING" algorithm).

    Layer-sorts each tree independently, then concatenates the i-th layer of
    every tree into one combined layer. Because nodes from different candidate
    trees are mutually independent, this lets layer i of all trees be processed
    together in a single batched GPU pass. Returns the list of merged layers.
    """
    sorted_trees = [topological_sort(tree, order=order) for tree in trees]
    merged_layers = []
    remaining = len(sorted_trees)  # trees not yet fully consumed
    layer_idx = 0
    while remaining > 0:
        layer = []
        for st in sorted_trees:
            if layer_idx < len(st):
                # This tree still has a layer at this depth; pool its nodes.
                layer += st[layer_idx]
            elif layer_idx == len(st):
                # Tree exhausted exactly once; count it off (only at == len).
                remaining -= 1
        layer_idx += 1
        if layer:
            merged_layers.append(layer)
    return merged_layers


def topological_sort(tree, order="up"):
    """Group a tree's nodes into dependency-respecting layers.

    order="up": leaves -> root (a parent appears only after both children);
    order="down": root -> leaves. Each returned layer contains nodes whose
    dependencies all lie in earlier layers, so a layer can be processed in one
    batched pass. Returns a list of layers (lists of nodes).
    """
    valid_orders = ["up", "down"]
    assert order in valid_orders, "order is not supported."
    layers = []
    recorded = set()  # identifiers already placed in some layer
    # Seed with the leaves (going up) or the root (going down).
    layer = [tree[leaf] for leaf in tree.get_leaves()] if order == "up" else [tree.root]
    while layer:
        layers.append(layer)
        recorded.update([n.identifier for n in layer])
        next_layer = []
        for node in layer:
            # Move toward the destination: up to parents, or down to children.
            next_nodes = [node.parent] if order == "up" else node.children.values()
            for next_node in next_nodes:
                if next_node and next_node.identifier not in recorded:
                    if_add = True
                    # Going up, a parent is ready only once ALL its children
                    # have been recorded in earlier layers.
                    if order == "up":
                        for child in next_node.children:
                            if child not in recorded:
                                if_add = False
                                break
                    if if_add:
                        next_layer.append(next_node)
                        recorded.add(next_node.identifier)
        layer = next_layer
    return layers


if __name__ == "__main__":
    tree = util.get_random_binary_tree(5)
    tree2 = util.get_random_binary_tree(5)
    tree3 = util.get_random_binary_tree(5)
    tree.draw()
    tree2.draw()
    tree3.draw()
    # sorted = topological_sort(tree, order='up')
    sorted = batch_topological_sort([tree, tree2, tree3], order="down")
    print([[n.name for n in li] for li in sorted])