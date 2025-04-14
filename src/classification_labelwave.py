import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import src.labelwave.labelwave_cifar_dataloader as dataloader
from src.utils import set_seeds


def build_model(num_classes: int = 10):
    from src.labelwave.resnet import resnet32
    model = resnet32(num_classes)
    print('============ use resnet32 ')
    model = model.cuda()
    return model


def main(seeds):
    device = torch.device("cuda")
    # REMEMBER TO DOWNLOAD CIFAR10/100 AND THEN UNZIP THEM!
    epochs = 250
    pre_epoch = 0
    batch_size = 100
    lr = 0.1
    r = 0.4
    labelwave_patience = 10
    noise_mode = 'sym'
    dataset = "cifar-10"
    if dataset == 'cifar-10':
        num_classes = 10
    else:
        num_classes = 100
    root_dir = f'./{dataset}'

    best_metrics = []
    for argsseed in range(seeds):
        file_name = f"Results_labelwave_{noise_mode}{r}_{dataset}_ResNet32_lr{lr}_bs{batch_size}_seed{argsseed}"
        file_name = str(file_name)
        print(file_name)

        csvfile = file_name + ".csv"
        with open(csvfile, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'LabelWave', 'ValAcc', 'TestAcc'])

        path = os.getcwd()
        file_path = path + '/' + file_name
        folder = os.path.exists(file_path)
        if not folder:
            os.mkdir(file_path)
        set_seeds(argsseed)

        loader = dataloader.cifar_dataloader(dataset=dataset,
                                             r=r,
                                             noise_mode=noise_mode,
                                             batch_size=batch_size,
                                             num_workers=12,
                                             file_name=file_name,
                                             root_dir=root_dir,
                                             random_seed=int(argsseed),
                                             noise_file='%s/%.1f_%s.json' % (root_dir, r, noise_mode))

        test_loader = loader.run('test')
        train_loader = loader.run('train')
        # val_loader = loader.run('val')

        print("Start Training!")
        net = build_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        min_labelwave = float("inf")
        current_patience = 0
        # best_val_acc = 0
        best_test_acc = 0
        best_epoch = 0

        previous_predictions = {}
        for epoch in range(pre_epoch, epochs):
            start_epoch_time = time.time()
            labelwave = 0
            total_loss = 0
            total_batches = 0
            current_epoch_predictions = {}
            print("Training")
            net.train()
            for batch_idx, (inputs, labels, paths) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=True):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                total_loss += loss.item()
                total_batches += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                _, current_predictions = torch.max(outputs.data, 1)
                for idx, path_tensor in enumerate(paths):
                    path = str(path_tensor.item())
                    current_epoch_predictions[path] = current_predictions[idx].item()
            labelwave = 0
            for path, prediction in current_epoch_predictions.items():
                if path in previous_predictions and previous_predictions[path] != prediction:
                    labelwave += 1
            previous_predictions = current_epoch_predictions.copy()
            if epoch > 1:
                if labelwave < min_labelwave:
                    min_labelwave = labelwave
                    current_patience = 0
                else:
                    current_patience += 1
            print(f"Epoch {epoch + 1}, LabelWave: {labelwave}")
            scheduler.step()

            with torch.no_grad():
                correct = 0
                total = 0
                print("Validating")
                # net.eval()
                # for batch_idx, (inputs, labels) in enumerate(val_loader):
                #    images, labels = inputs, labels
                #    images, labels = images.cuda(), labels.cuda()
                #    outputs = net(images)
                #    #loss = criterion(outputs, labels)
                #    _, predicted = torch.max(outputs.data, 1)
                #    total += labels.size(0)
                #    correct += (predicted == labels).sum()
                #    val_acc = 100 * correct / total
                #    val_acc = val_acc.item()

                correct = 0
                total = 0
                print("Testing")
                net.eval()
                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    images, labels = inputs, labels
                    images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)
                    # loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    test_acc = 100 * correct / total
                    test_acc = test_acc.item()
            end_epoch_time = time.time()
            # print('ValAcc：%.3f%%' % val_acc, 'TestAcc：%.3f%%' % test_acc)
            print('TestAcc：%.3f%%' % test_acc)
            print(f"Epoch completed in {end_epoch_time - start_epoch_time:.2f} seconds")
            with open(csvfile, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, labelwave, -0.0, test_acc])

            if current_patience == 0:
                best_epoch = epoch
                # best_val_acc = val_acc
                best_test_acc = test_acc
            if current_patience >= labelwave_patience:
                print(f"EARLY-STOP - found performances: (epoch: {best_epoch})")
                # print('ValAcc：%.3f%%' % best_val_acc, 'TestAcc：%.3f%%' % best_test_acc)
                print('TestAcc：%.3f%%' % best_test_acc)
                best_metrics.append([best_epoch, best_test_acc])
                break
    for idx, metric in enumerate(best_metrics):
        print(f"{idx}: {metric}")


if __name__ == "__main__":
    main(5)
