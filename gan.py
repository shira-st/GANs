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
  def __init__(self, img_size, noise_size=100):
    self.noise_size = noise_size

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.generator = Generator(noise_size, img_size).to(self.device)
    self.discriminator = Discriminator(img_size).to(self.device)

    # The learning rates and momentums are roughly chosen.
    self.g_optimizer = optim.SGD(self.generator.parameters(), lr=0.01, momentum=0.9)
    self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=0.01, momentum=0.9)
    self.g_losses = []
    self.d_losses = []

  def train(self, dataloader, epoch):
    criterion = nn.BCELoss()

    for i in range(epoch):
      g_running_loss = 0
      d_running_loss = 0

      for _, (images, _) in enumerate(dataloader):
        n = images.size()[0]

        ### train discriminator ###
        self.d_optimizer.zero_grad()
        fake_images = self.generator(torch.rand(n, self.noise_size, device=self.device))
        real_images = images.view(n, -1).to(self.device)
        fake_labels = torch.zeros((n, ), device=self.device)
        real_labels = torch.ones((n, ), device=self.device)

        d_fake_loss = criterion(self.discriminator(fake_images).view(-1), fake_labels)
        d_real_loss = criterion(self.discriminator(real_images).view(-1), real_labels)
        d_loss = d_fake_loss + d_real_loss
        d_running_loss += d_loss.item()
        d_loss.backward()
        self.d_optimizer.step()

        ### train generator ###
        self.g_optimizer.zero_grad()
        fake_images = self.generator(torch.rand(n, self.noise_size, device=self.device))
        fake_labels = torch.ones((n, ), device=self.device)

        g_loss = criterion(self.discriminator(fake_images).view(-1), fake_labels)
        g_running_loss += g_loss.item()
        g_loss.backward()
        self.g_optimizer.step()

      g_running_loss /= len(dataloader)
      d_running_loss /= len(dataloader)
      self.g_losses.append(g_running_loss)
      self.d_losses.append(d_running_loss)
      print("[%d] Generator: %f, Discriminator: %f" % (i + 1, g_running_loss, d_running_loss))
    return

  def set_state(self, state):
    self.generator.load_state_dict(state["generator_state_dict"])
    self.discriminator.load_state_dict(state["discriminator_state_dict"])
    self.g_optimizer.load_state_dict(state["g_optimizer_state_dict"])
    self.d_optimizer.load_state_dict(state["d_optimizer_state_dict"])
    self.g_losses.extend(state["g_losses"])
    self.d_losses.extend(state["d_losses"])
    return

  def get_state(self):
    state = {"generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
            "d_optimizer_state_dict": self.d_optimizer.state_dict(),
            "g_losses": self.g_losses,
            "d_losses": self.d_losses}
    return state
