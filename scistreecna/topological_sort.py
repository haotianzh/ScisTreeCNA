from . import util


def batch_topological_sort(trees, order='up'):
    sorted_trees = []
    for tree in trees:
        sorted_trees.append(topological_sort(tree, order=order))
    merged_layers = []
    while sorted_trees:
        layer = []
        for tree in sorted_trees:
            if tree:
                layer += tree.pop(0)
            else:
                sorted_trees.remove(tree)
        if layer:
            merged_layers.append(layer)
    return merged_layers


def topological_sort(tree, order='up'):
    valid_orders = ['up', 'down']
    assert order in valid_orders, "order is not supported."
    layers = []
    recorded = set()
    layer = [tree[leaf] for leaf in tree.get_leaves()] if order == 'up' else [tree.root]
    while layer:
        layers.append(layer)
        recorded.update([n.identifier for n in layer])
        next_layer = []
        for node in layer:
            next_nodes = [node.parent] if order == 'up' else node.children.values() 
            for next_node in next_nodes:
                if next_node and next_node.identifier not in recorded:
                    if_add = True
                    if order == 'up':
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
    sorted = batch_topological_sort([tree, tree2, tree3], order='down')
    print([[n.name for n in li] for li in sorted])