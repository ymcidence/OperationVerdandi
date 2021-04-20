def get_dataset_stat(dataset):
    if dataset == 'cifar10':
        image_size = 32
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        n_class = 10
    elif dataset == 'cifar100' or dataset == 'cifar20':
        image_size = 32
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        if dataset == 'cifar100':
            n_class = 100
        else:
            n_class = 20
    elif dataset == 'stl10':
        image_size = 96
        mean = [0.4409, 0.4279, 0.3868]
        std = [0.2683, 0.2610, 0.2687]
        n_class = 10
    else:
        raise NotImplementedError

    return image_size, mean, std, n_class
