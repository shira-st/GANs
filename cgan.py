import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
  def __init__(self, noise_size, condition_size, out_size):
    super(Generator, self).__init__()
    # Dropout is applied to all layers for now.
    self.noise_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(noise_size, 200),
        nn.ReLU())
    self.condition_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(condition_size, 1000),
        nn.ReLU())
    self.concat_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1200, 1200),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1200, out_size),
        nn.Sigmoid())

  def forward(self, noise, condition):
    noise = self.noise_layer(noise)
    condition = self.condition_layer(condition)
    x = self.concat_layer(torch.cat((noise, condition), dim=1))
    return x


class Discriminator(nn.Module):
  def __init__(self, in_size, condition_size):
    # Dropout is applied to all layers for now.
    super(Discriminator, self).__init__()
    # Maxout layer with 240 units and 5 pieces is used in the original paper.
    self.data_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_size, 240),
        nn.ReLU())
    # Maxout layer with 50 units and 5 pieces is used in the original paper.
    self.condition_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(condition_size, 50),
        nn.ReLU())
    # Maxout layer with 240 units and 4 pieces is used in the original paper.
    self.concat_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(290, 240),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(240, 1),
        nn.Sigmoid())

  def forward(self, data, condition):
    data = self.data_layer(data)
    condition = self.condition_layer(condition)
    x = self.concat_layer(torch.cat((data, condition), dim=1))
    return x


class CGAN:
  def __init__(self, data_size, condition_size,
               noise_generator=None,
               generator=None,
               discriminator=None,
               g_optimizer=None,
               d_optimizer=None,
               g_scheduler=None,
               d_scheduler=None,
               device=None):
    self.device = device

    if noise_generator:
      self.noise_generator = lambda n: noise_generator(n).to(device)
    else:
      self.noise_generator = lambda n: torch.rand(n, 100, device=device)

    if generator:
      self.generator = generator.to(device)
    else:
      self.generator = Generator(self.noise_generator(1).size()[1], condition_size, data_size).to(device)
    if discriminator:
      self.discriminator = discriminator.to(device)
    else:
      self.discriminator = Discriminator(data_size, condition_size).to(device)

    if g_optimizer:
      self.g_optimizer = g_optimizer
    else:
      self.g_optimizer = optim.SGD(self.generator.parameters(), lr=0.1, momentum=0.5)
    if d_optimizer:
      self.d_optimizer = d_optimizer
    else:
      self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=0.1, momentum=0.5)

    # Momentum is increased from 0.5 up to 0.7 in the original paper.
    lr_lambda = lambda epoch: max(0.99996 ** epoch, 0.00001)
    if g_scheduler:
      self.g_scheduler = g_scheduler
    else:
      self.g_scheduler = optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda)
    if d_scheduler:
      self.d_scheduler = d_scheduler
    else:
      self.d_scheduler = optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda)

  def train(self, data, condition):
    criterion = nn.BCELoss()

    n = data.size()[0]

    ### train discriminator ###
    self.d_optimizer.zero_grad()
    fake_data = self.generator(self.noise_generator(n), condition)
    fake_labels = torch.zeros((n, ), device=self.device)
    real_labels = torch.ones((n, ), device=self.device)

    d_fake_loss = criterion(self.discriminator(fake_data, condition).view(-1), fake_labels)
    d_real_loss = criterion(self.discriminator(data, condition).view(-1), real_labels)
    d_loss = d_fake_loss + d_real_loss
    d_loss.backward()
    self.d_optimizer.step()

    ### train generator ###
    self.g_optimizer.zero_grad()
    fake_data = self.generator(self.noise_generator(n), condition)
    fake_labels = torch.ones((n, ), device=self.device)

    g_loss = criterion(self.discriminator(fake_data, condition).view(-1), fake_labels)
    g_loss.backward()
    self.g_optimizer.step()
    return g_loss.item(), d_loss.item()

  def scheduler_step(self):
    if self.g_scheduler:
      self.g_scheduler.step()
    if self.d_scheduler:
      self.d_scheduler.step()
    return

  def sample(self, n, condition):
    noise = self.noise_generator(n)
    return self.generator(noise, condition)

  def set_state(self, state):
    self.generator.load_state_dict(state["generator_state_dict"])
    self.discriminator.load_state_dict(state["discriminator_state_dict"])
    self.g_optimizer.load_state_dict(state["g_optimizer_state_dict"])
    self.d_optimizer.load_state_dict(state["d_optimizer_state_dict"])
    if self.g_scheduler:
      self.g_scheduler.load_state_dict(state["g_scheduler_state_dict"])
    if self.d_scheduler:
      self.d_scheduler.load_state_dict(state["d_scheduler_state_dict"])
    return

  def get_state(self):
    state = {"generator_state_dict": self.generator.state_dict(),
             "discriminator_state_dict": self.discriminator.state_dict(),
             "g_optimizer_state_dict": self.g_optimizer.state_dict(),
             "d_optimizer_state_dict": self.d_optimizer.state_dict()}
    if self.g_scheduler:
      state["g_scheduler_state_dict"] = self.g_scheduler.state_dict()
    if self.d_scheduler:
      state["d_scheduler_state_dict"] = self.d_scheduler.state_dict()
    return state
