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
    self.drop1 = nn.Dropout(0.2)
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


class GAN:
  def __init__(self, dataloader):
    self.dataloader = dataloader
    data, _ = next(iter(dataloader))
    self.channel = data.size()[1]
    self.height = data.size()[2]
    self.width = data.size()[3]
    self.in_size = 100
    self.out_size = self.channel * self.height * self.width
    self.g = Generator(self.in_size, self.out_size)
    self.d = Discriminator(self.out_size)
    self.g_optimizer = optim.Adam(self.g.parameters())
    self.d_optimizer = optim.Adam(self.d.parameters())
    self.g_losses = []
    self.d_losses = []

  def train(self, epoch):

    # models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.g = self.g.to(device)
    self.d = self.d.to(device)

    # optimizers
    criterion = nn.BCELoss()

    # training
    for i in range(epoch):
      g_running_loss = 0
      d_running_loss = 0
      for _, (images, _) in enumerate(self.dataloader):
        n = images.size()[0]

        ### train generator ###
        self.g_optimizer.zero_grad()

        # generate data
        fake_images = self.g(torch.rand((n, self.in_size), device=device))
        fake_labels = torch.full((n, ), 1, device=device)

        # backpropagation
        g_loss = criterion(self.d(fake_images).view(-1), fake_labels)
        g_running_loss += g_loss.item()
        g_loss.backward()
        self.g_optimizer.step()


        ### train discriminator ###
        self.d_optimizer.zero_grad()

        # generate data
        fake_images = self.g(torch.rand((n, self.in_size), device=device)).detach()
        fake_labels = torch.full((n, ), 0, device=device)
        real_labels = torch.full((n, ), 1, device=device)

        # optimize
        d_fake_loss = criterion(self.d(fake_images).view(-1), fake_labels)
        d_real_loss = criterion(self.d(images.view(n, -1).to(device)).view(-1), real_labels)
        d_loss = 0.5 * (d_fake_loss + d_real_loss)
        d_running_loss += d_loss.item()
        d_loss.backward()
        self.d_optimizer.step()

      g_running_loss /= len(self.dataloader)
      d_running_loss /= len(self.dataloader)
      self.g_losses.append(g_running_loss)
      self.d_losses.append(d_running_loss)
      print("[%d] Generator: %f, Discriminator: %f" % (i, g_running_loss, d_running_loss))

    return
