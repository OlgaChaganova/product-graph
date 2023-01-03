from ogb.nodeproppred import PygNodePropPredDataset


if __name__ == '__main__':
    dataset = PygNodePropPredDataset(name='ogbn-products', root='../data') 
