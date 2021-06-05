import torch as th
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GraphConv, RGCNConv
from argparse import Namespace

def get_activation(name: str, leaky_relu: float = .5):
    if name == 'leaky_relu':
        return nn.LeakyReLU(leaky_relu)
    elif name == 'rrelu':
        return nn.RReLU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'selu':
        return nn.SELU()
    else:
        raise Exception('Unknown activation')


def get_gnn_conv(name: str):
    if name == 'gcn':
        return GCNConv
    elif name == 'gat':
        return GATConv
    elif name == 'graph_conv':
        return GraphConv
    elif name == 'rcgn':
        return RGCNConv
    else:
        raise Exception('Unknown GNN layer')


def get_initialiser(name: str):
    if name == 'orthogonal':
        return nn.init.orthogonal_
    elif name == 'xavier':
        return nn.init.xavier_uniform_
    elif name == 'kaiming':
        return nn.init.kaiming_uniform_
    elif name == 'none':
        pass
    else:
        raise Exception('Unknown init method')


def get_optimiser(args: Namespace, params, net=None):
    weight_decay = args.weight_decay
    lr = args.lr
    if net == 'propensity':
        weight_decay = args.pro_weight_decay
        lr = args.pro_lr
    elif net == 'como':
        lr = args.como_lr
        weight_decay = args.como_weight_decay
    elif net == 'gnn':
        lr = args.como_lr
        weight_decay = args.gnn_weight_decay

    optimiser = None
    if args.optimiser == 'sgd':
        optimiser = th.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif args.optimiser == 'adam':
        optimiser = th.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif args.optimiser == 'amsgrad':
        optimiser = th.optim.Adam(params, lr=lr, amsgrad=True, weight_decay=weight_decay)
    return optimiser


def get_lr_scheduler(args: Namespace, optimizer):
    if args.lr_scheduler == 'exponential':
        return th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        return th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
    elif args.lr_scheduler == 'cycle':
        return th.optim.lr_scheduler.CyclicLR(optimizer, 0, max_lr=args.lr, step_size_up=20, cycle_momentum=False)
    elif args.lr_scheduler == 'none':
        return NoneScheduler()


class NoneScheduler:
    def step(self):
        pass


def get_optimiser_scheduler(args: Namespace, model, net=None):
    optimiser = get_optimiser(args, model.parameters(), net)
    lr_scheduler = get_lr_scheduler(args, optimiser)
    return optimiser, lr_scheduler
