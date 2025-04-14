from torch.optim import Adam, SGD


def get_optimizer(args, model):
    print(f"Initialize optimizer {args.optim}")

    if args.optim == "sgd":
        return SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    if args.optim == "adam":
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def get_optimizers(args, model):
    print(f"Initialize optimizer(s) {args.optim}")

    optimizers = [get_optimizer(args, model)]

    return optimizers
