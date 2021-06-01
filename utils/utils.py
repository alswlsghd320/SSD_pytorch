import torch.optim as optim

def get_optimizer(net, optim_name ='Adam', lr=1e-3, decay=5e-4, momentum=0.99):
    optim_name = optim_name.lower()
    if optim_name == 'adam':
        return optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
    elif optim_name == 'sgd':
        return optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    else:
        raise ValueError('There is no optimizer in this projects.')