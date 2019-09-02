import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):
  def __init__(self, noise_size, label_size, out_size):
    super(Generator, self).__init__()
    self.fc_noise = nn.Linear(noise_size, 200)
    self.fc_label = nn.Linear(label_size, 1000)
    self.drop1 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(1200, 1200)
    self.drop2 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(1200, out_size)

  def forward(self, noise, label):
    # It is unclear whether Dropout is used for inputs or not.
    noise = F.relu(self.fc_noise(noise), inplace=True)
    label = F.relu(self.fc_label(label), inplace=True)
    x = torch.cat((noise, label), dim=1)
    x = F.relu(self.fc1(self.drop1(x)), inplace=True)
    x = torch.sigmoid(self.fc2(self.drop2(x)))
    return x


class Discriminator(nn.Module):
  def __init__(self, in_size, label_size):
    super(Discriminator, self).__init__()
    self.fc_data = nn.Linear(in_size, 240)
    self.fc_label = nn.Linear(label_size, 50)
    self.drop1 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(290, 240)
    self.drop2 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(240, 1)

  def forward(self, data, label):
    # It is unclear whether Dropout is used for inputs or not.

    # Authors use a maxout layer with 240 units and 5 pieces.
    data = F.relu(self.fc_data(data), inplace=True)

    # Authors use a maxout layer with 50 units and 5 pieces.
    label = F.relu(self.fc_label(label), inplace=True)
    x = torch.cat((data, label), dim=1)

    # Authors use a maxout layer with 240 units and 4 pieces.
    x = F.relu(self.fc1(self.drop1(x)), inplace=True)
    x = torch.sigmoid(self.fc2(self.drop2(x)))
    return x


class CGAN:
  def __init__(self, dataloader, label_size,
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
    self.label_size = label_size

    self.noise_size = noise_size
    if noise_generator is None:
      self.noise_generator = lambda n: torch.rand(n, self.noise_size)
    else:
      self.noise_generator = noise_generator

    self.out_size = self.channel * self.height * self.width

    if generator is None:
      self.generator = Generator(self.noise_size, self.label_size, self.out_size)
    else:
      self.generator = generator

    if discriminator is None:
      self.discriminator = Discriminator(self.out_size, self.label_size)
    else:
      self.discriminator = discriminator

    # Authors use SGD with initial leraning rate 0.1 and momentum 0.5.
    # Learning rate is exponentially decreased down to 1e-6 with decay factor of 1.00004.
    # Momentum is increased up to 0.7.
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

      for _, (images, labels) in enumerate(self.dataloader):
        n = images.size()[0]
        images = images.view(n, -1).to(device)
        labels = torch.stack([torch.eye(self.label_size)[label] for label in labels]).to(device)

        ### train generator ###
        self.g_optimizer.zero_grad()

        noise = self.noise_generator(n).to(device)
        fake_images = self.generator(noise, labels)
        fake_labels = torch.ones((n, ), device=device)

        g_loss = criterion(self.discriminator(fake_images, labels).view(-1), fake_labels)
        g_running_loss += g_loss.item()
        g_loss.backward()
        self.g_optimizer.step()


        ### train discriminator ###
        self.d_optimizer.zero_grad()

        noise = self.noise_generator(n).to(device)
        fake_images = self.generator(noise, labels).detach()
        fake_labels = torch.zeros((n, ), device=device)
        real_labels = torch.ones((n, ), device=device)

        d_fake_loss = criterion(self.discriminator(fake_images, labels).view(-1), fake_labels)
        d_real_loss = criterion(self.discriminator(images, labels).view(-1), real_labels)
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
