import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
  def __init__(self, noise_size, out_size):
    super(Generator, self).__init__()
    # The number of layers and hidden nodes are roughly chosen.
    self.model = nn.Sequential(
        nn.Linear(noise_size, 1024),
        nn.ReLU(),

        nn.Linear(1024, 1024),
        nn.ReLU(),

        nn.Linear(1024, out_size),
        nn.Sigmoid())

  def forward(self, x):
    return self.model(x)


class Discriminator(nn.Module):
  def __init__(self, in_size):
    super(Discriminator, self).__init__()
    # The number of layers and hidden nodes and dropout rates are roughly chosen.
    # Maxout layers are used in the original paper instead of ReLU activation functions.　
    self.model = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_size, 1024),
        nn.ReLU(),

        nn.Dropout(0.5),
        nn.Linear(1024, 1024),
        nn.ReLU(),

        nn.Dropout(0.5),
        nn.Linear(1024, 1),
        nn.Sigmoid())

  def forward(self, x):
    return self.model(x)


class GAN:
  def __init__(self, data_size,
               noise_generator=None,
               generator=None,
               discriminator=None,
               g_optimizer=None,
               d_optimizer=None,
               device=None):
    self.device = device

    # The dimensionality of noise is roughly chosen.
    if noise_generator:
      self.noise_generator = lambda n: noise_generator(n).to(device)
    else:
      self.noise_generator = lambda n: torch.rand(n, 100, device=device)

    if generator:
      self.generator = generator.to(device)
    else:
      self.generator = Generator(self.noise_generator(1).size()[1], data_size).to(device)
    if discriminator:
      self.discriminator = discriminator.to(device)
    else:
      self.discriminator = Discriminator(data_size).to(device)

    # The learning rates and momentums are roughly chosen.
    if g_optimizer:
      self.g_optimizer = g_optimizer
    else:
      self.g_optimizer = optim.SGD(self.generator.parameters(), lr=0.01, momentum=0.9)
    if d_optimizer:
      self.d_optimizer = d_optimizer
    else:
      self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=0.01, momentum=0.9)

  def train(self, data):
    criterion = nn.BCELoss()

    n = data.size()[0]

    ### train discriminator ###
    self.d_optimizer.zero_grad()
    fake_data = self.generator(self.noise_generator(n))
    fake_labels = torch.zeros((n, ), device=self.device)
    real_labels = torch.ones((n, ), device=self.device)

    d_fake_loss = criterion(self.discriminator(fake_data).view(-1), fake_labels)
    d_real_loss = criterion(self.discriminator(data).view(-1), real_labels)
    d_loss = d_fake_loss + d_real_loss
    d_loss.backward()
    self.d_optimizer.step()

    ### train generator ###
    self.g_optimizer.zero_grad()
    fake_data = self.generator(self.noise_generator(n))
    fake_labels = torch.ones((n, ), device=self.device)

    g_loss = criterion(self.discriminator(fake_data).view(-1), fake_labels)
    g_loss.backward()
    self.g_optimizer.step()
    return g_loss.item(), d_loss.item()

  def sample(self, n):
    return self.generator(self.noise_generator(n))

  def set_state(self, state):
    self.generator.load_state_dict(state["generator_state_dict"])
    self.discriminator.load_state_dict(state["discriminator_state_dict"])
    self.g_optimizer.load_state_dict(state["g_optimizer_state_dict"])
    self.d_optimizer.load_state_dict(state["d_optimizer_state_dict"])
    return

  def get_state(self):
    state = {"generator_state_dict": self.generator.state_dict(),
             "discriminator_state_dict": self.discriminator.state_dict(),
             "g_optimizer_state_dict": self.g_optimizer.state_dict(),
             "d_optimizer_state_dict": self.d_optimizer.state_dict()}
    return state
