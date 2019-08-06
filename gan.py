
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, in_size, out_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = torch.sigmoid(self.fc3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, in_size):
        super(Discriminator, self).__init__()
        self.drop1 = nn.Dropout(0.8)
        self.fc1 = nn.Linear(in_size, 1024)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.fc1(self.drop1(x)), inplace=True)
        x = F.relu(self.fc2(self.drop2(x)), inplace=True)
        x = torch.sigmoid(self.fc3(self.drop3(x)))
        return x


def train(dataloader, epoch, generator=None, discriminator=None):
    data, _ = next(iter(dataloader))
    height = data.size()[2]
    width = data.size()[3]

    # parameters
    in_size = 100
    out_size = height * width

    # models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if generator is None:
        g = Generator(in_size, out_size).to(device)
    else:
        g = generator.to(device)
    if discriminator is None:
        d = Discriminator(out_size).to(device)
    else:
        d = discriminator.to(device)

    # optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(g.parameters(), lr=0.0001)
    d_optimizer = optim.Adam(d.parameters(), lr=0.0001)

    # training
    g_losses = []
    d_losses = []
    for i in range(epoch):
        g_running_loss = 0
        d_running_loss = 0
        for _, (images, _) in enumerate(dataloader):
            n = images.size()[0]

            ### train generator ###
            g_optimizer.zero_grad()

            # generate data
            fake_images = g(torch.rand((n, in_size), device=device))
            fake_labels = torch.full((n, ), 1, device=device)

            # backpropagation
            g_loss = criterion(d(fake_images).view(-1), fake_labels)
            g_running_loss += g_loss.item()
            g_loss.backward()
            g_optimizer.step()


            ### train discriminator ###
            d_optimizer.zero_grad()

            # generate data
            fake_images = g(torch.rand((n, in_size), device=device)).detach()
            fake_labels = torch.full((n, ), 0, device=device)
            real_labels = torch.full((n, ), 1, device=device)

            # optimize
            d_fake_loss = criterion(d(fake_images).view(-1), fake_labels)
            d_real_loss = criterion(d(images.view(n, -1).to(device)).view(-1), real_labels)
            d_loss = d_fake_loss + d_real_loss
            d_running_loss += d_loss.item()
            d_loss.backward()
            d_optimizer.step()

        g_losses.append(g_running_loss)
        d_losses.append(d_running_loss)
        print("Generator: %f, Discriminator: %f" % (g_running_loss, d_running_loss))

    return g, d, np.array(g_losses), np.array(d_losses)
