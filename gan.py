import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):
  def __init__(self, noise_size, out_size):
    # The number of layers and hiddlen nodes are unclear.
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(noise_size, 1024)
    self.fc2 = nn.Linear(1024, 1024)
    self.fc3 = nn.Linear(1024, out_size)

  def forward(self, x):
    x = F.relu(self.fc1(x), inplace=True)
    x = F.relu(self.fc2(x), inplace=True)
    x = torch.sigmoid(self.fc3(x))
    return x


class Discriminator(nn.Module):
  def __init__(self, in_size):
    # The number of layers and hidden nodes and dropout rates are unclear.
    super(Discriminator, self).__init__()
    self.drop1 = nn.Dropout(0.2)
    self.fc1 = nn.Linear(in_size, 1024)
    self.drop2 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(1024, 1024)
    self.drop3 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(1024, 1)

  def forward(self, x):
    # Authors use maxout layers.
    x = F.relu(self.fc1(self.drop1(x)), inplace=True)
    x = F.relu(self.fc2(self.drop2(x)), inplace=True)
    x = torch.sigmoid(self.fc3(self.drop3(x)))
    return x


class GAN:
  def __init__(self, dataloader,
               noise_size=100,
               noise_generator=None,
               generator=None,
               discriminator=None,
               g_optimizer=None,
               d_optimizer=None
              ):
    self.dataloader = dataloader
    data, _ = next(iter(dataloader))
    self.channel = data.size()[1]
    self.height = data.size()[2]
    self.width = data.size()[3]

    self.noise_size = noise_size
    if noise_generator is None:
      self.noise_generator = lambda n: torch.rand(n, self.noise_size)
    else:
      self.noise_generator = noise_generator

    self.out_size = self.channel * self.height * self.width

    if generator is None:
      self.generator = Generator(self.noise_size, self.out_size)
    else:
      self.generator = generator

    if discriminator is None:
      self.discriminator = Discriminator(self.out_size)
    else:
      self.discriminator = discriminator

    # Authors use SGD with momentum.
    if g_optimizer is None:
      self.g_optimizer = optim.Adam(self.generator.parameters())
    else:
      self.g_optimizer = g_optimizer

    if d_optimizer is None:
      self.d_optimizer = optim.Adam(self.discriminator.parameters())
    else:
      self.d_optimizer = d_optimizer

    self.g_losses = []
    self.d_losses = []

  def train(self, epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.generator = self.generator.to(device)
    self.discriminator = self.discriminator.to(device)
    criterion = nn.BCELoss()

    for i in range(epoch):
      g_running_loss = 0
      d_running_loss = 0

      for _, (images, _) in enumerate(self.dataloader):
        n = images.size()[0]
        images = images.view(n, -1).to(device)

        ### train generator ###
        self.g_optimizer.zero_grad()
        noise = self.noise_generator(n).to(device)
        fake_images = self.generator(noise)
        fake_labels = torch.ones((n, ), device=device)

        g_loss = criterion(self.discriminator(fake_images).view(-1), fake_labels)
        g_running_loss += g_loss.item()
        g_loss.backward()
        self.g_optimizer.step()


        ### train discriminator ###
        self.d_optimizer.zero_grad()
        noise = self.noise_generator(n).to(device)
        fake_images = self.generator(noise).detach()
        fake_labels = torch.zeros((n, ), device=device)
        real_labels = torch.ones((n, ), device=device)

        d_fake_loss = criterion(self.discriminator(fake_images).view(-1), fake_labels)
        d_real_loss = criterion(self.discriminator(images).view(-1), real_labels)
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
