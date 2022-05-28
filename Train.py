from copy import deepcopy
import torch.nn
import numpy as np
import tqdm
from Model import AlexNet
from DataLoaderDogCat import DogCats
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from torch.autograd import Variable
import matplotlib.pyplot as plt


def simulate_self(model_, memories_, lr_, optimizer_):
    loss_ = torch.nn.NLLLoss()  # why? why not
    loss_ = loss_.cuda()
    lr__ = deepcopy(lr_)

    # create copies of state dicts of model and optimizer
    model_copy = AlexNet(num_classes=2)
    sd_model = model_.state_dict()
    model_copy.load_state_dict(sd_model)
    model_copy.cuda()

    optimizer_copy = SGD(model_copy.parameters(), lr=optimizer_.param_groups[0]['lr'])
    sd_opt = optimizer_.state_dict()
    optimizer_copy.load_state_dict(sd_opt)
    optimizer_copy.param_groups[0]['lr'] = lr__

    x = torch.cat([xx for xx in map(lambda l: l[0], memories_)], 0)
    y = torch.cat([yy for yy in map(lambda l: l[1], memories_)], 0)

    y_output = model_copy(x)
    err_ = loss_(y_output, y)

    err_.backward()
    optimizer_copy.step()

    y_output_ = model_copy(x)
    y_output_probs = torch.exp(y_output_)
    y_output_labels = np.argmax(y_output_probs.cpu().data.numpy(), axis=1)
    correct_ones = 0
    for y_real, y_pred in zip(y, y_output_labels):
        y_real = y_real.cpu().data.numpy().flatten()[0]
        if y_real == y_pred:
            correct_ones += 1

    return correct_ones / len(y)


BATCH_SIZE = 1
LEARNING_RATE = 0.001
USE_CUDA = torch.cuda.is_available()
N_EPOCHS = 100
IMAGE_SIZE = (224, 224)

model = AlexNet(num_classes=2)
print(model)

data_loader = DataLoader(DogCats(IMAGE_SIZE, augment=False), batch_size=BATCH_SIZE, shuffle=True)

optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
loss_class = torch.nn.NLLLoss()

if USE_CUDA:
    model = model.cuda()
    loss_class = loss_class.cuda()

for p in model.parameters():
    p.requires_grad = True

memories = []
lr = LEARNING_RATE
f = 100
lr_stepsize = lr / 10

for epoch in range(N_EPOCHS + 1):
    data_iter = iter(data_loader)
    i = 0
    epoch_losses = []
    with tqdm.tqdm(total=len(data_loader)) as pbar:
        while i < len(data_loader):
            sample = next(data_iter)
            s_image, s_label = sample['image'], sample['class']
            s_label = s_label.long()

            model.zero_grad()
            if USE_CUDA:
                s_image = s_image.cuda()
                s_label = s_label.cuda()

            s_image_v = Variable(s_image)
            s_label_v = Variable(s_label)

            memories.append([s_image_v, s_label_v])
            if len(memories) > f:
                memories.pop(0)

            s_class_output = model(s_image_v)

            s_class_output_probs = torch.exp(s_class_output)
            label_y = np.argmax(s_class_output_probs[0].cpu().data.numpy())
            if i > 0 and i % f == 0:
                if label_y == 0:
                    # I love dogs!
                    simulated_lr = lr + lr_stepsize
                else:
                    # Cats are my enemies
                    simulated_lr = lr - lr_stepsize
                ingest_accuracy = simulate_self(model, memories, simulated_lr, optimizer)
                reject_accuracy = simulate_self(model, memories, lr, optimizer)
                if ingest_accuracy > reject_accuracy:
                    lr = simulated_lr

            err = loss_class(s_class_output, s_label_v)

            optimizer.param_groups[0]['lr'] = lr
            err.backward()
            optimizer.step()

            i += 1

            epoch_losses.append(err.cpu().data.numpy())
            pbar.set_description(f"Iter: {i}/{len(data_loader)}, [Loss: {np.mean(epoch_losses)}] [Learning Rate: {lr}]")
            pbar.update()

    print(f'[Epoch: {epoch}/{N_EPOCHS}, [Loss: {np.mean(epoch_losses)}]')
    torch.save(model, f'./checkpoints/homeostatic_model_{epoch}.pth')
