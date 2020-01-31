import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GANs.gan import GAN
from GANs.cgan import CGAN


class Generator_init(nn.Module):
  def __init__(self, noise_size, out_size):
    super(Generator_init, self).__init__()
    # The activation function is not used in the output layer for now.
    self.model = nn.Sequential(
        nn.Linear(noise_size, 1200),
        nn.ReLU(),

        nn.Linear(1200, 1200),
        nn.ReLU(),

        nn.Linear(1200, out_size))

  def forward(self, x):
    x = self.model(x)
    return x


class Discriminator_init(nn.Module):
  def __init__(self, in_size):
    super(Discriminator_init, self).__init__()
    # The dropout rates are roughly chosen.
    # The sigmoid function is used in the output layer for now.
    self.model = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_size, 600),
        nn.ReLU(),

        nn.Dropout(0.5),
        nn.Linear(600, 600),
        nn.ReLU(),

        nn.Dropout(0.5),
        nn.Linear(600, 1),
        nn.Sigmoid())

  def forward(self, x):
    x = self.model(x)
    return x


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    # The size of the images are maintained by padding for now.
    # The Sigmoid function is used in the output layer for the MNIST dataset.
    self.model = nn.Sequential(
        nn.Conv2d(2, 128, 7, padding=3),
        nn.ReLU(),

        nn.Conv2d(128, 128, 7, padding=3),
        nn.ReLU(),

        nn.Conv2d(128, 1, 5, padding=2),
        nn.Sigmoid())

  def forward(self, noise, condition):
    x = torch.cat([noise, condition], dim=1)
    x = self.model(x)
    return x


class Discriminator(nn.Module):
  def __init__(self, in_size):
    super(Discriminator, self).__init__()
    # The size of the images are maintained by padding for now.
    self.conv = nn.Sequential(
        nn.Conv2d(1, 128, 5, padding=2),
        nn.ReLU(),

        nn.Conv2d(128, 128, 5, padding=2),
        nn.ReLU())

    self.fc = nn.Sequential(
        nn.Linear(128 * in_size, 1),
        nn.Sigmoid())

  def forward(self, data, condition):
    x = data + condition
    x = self.conv(x)
    x = x.view(x.size()[0], -1)
    x = self.fc(x)
    return x


class LAPGAN:
  def __init__(self, image_sizes, noise_size=100):
    self.image_sizes = image_sizes

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Momentum is increased from 0.5 up to 0.8 by the factor of 0.0008 in the original paper.
    generator_init = Generator_init(noise_size, image_sizes[-1] ** 2)
    discriminator_init = Discriminator_init(image_sizes[-1] ** 2)
    g_optimizer_init = optim.SGD(generator_init.parameters(), lr=0.02, momentum=0.5)
    d_optimizer_init = optim.SGD(discriminator_init.parameters(), lr=0.02, momentum=0.5)
    self.gan_init = GAN(image_sizes[-1] ** 2,
                        noise_generator=lambda n: torch.rand(n, noise_size).uniform_(-1, 1),
                        generator=generator_init,
                        discriminator=discriminator_init,
                        g_optimizer=g_optimizer_init,
                        d_optimizer=d_optimizer_init,
                        device=self.device)

    generators = [Generator() for _ in image_sizes[: -1]]
    discriminators = [Discriminator(image_size ** 2) for image_size in image_sizes[: -1]]
    g_optimizers = [optim.SGD(generator.parameters(), lr=0.02, momentum=0.5) for generator in generators]
    d_optimizers = [optim.SGD(discriminator.parameters(), lr=0.02, momentum=0.5) for discriminator in discriminators]
    self.gans = []
    for k in range(len(image_sizes) - 1):
      gan = CGAN(None, None,
                 noise_generator=lambda n, k=k: torch.rand(n, 1, image_sizes[k], image_sizes[k]).uniform_(-1, 1),
                 generator=generators[k],
                 discriminator=discriminators[k],
                 g_optimizer=g_optimizers[k],
                 d_optimizer=d_optimizers[k],
                 device=self.device)
      self.gans.append(gan)

    gamma = 0.9996
    self.g_scheduler_init = optim.lr_scheduler.ExponentialLR(g_optimizer_init, gamma)
    self.d_scheduler_init = optim.lr_scheduler.ExponentialLR(d_optimizer_init, gamma)
    self.g_schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma) for optimizer in g_optimizers]
    self.d_schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma) for optimizer in d_optimizers]

    self.g_losses = [[] for _ in range(len(image_sizes))]
    self.d_losses = [[] for _ in range(len(image_sizes))]

  def train(self, dataloader, epoch):
    criterion = nn.BCELoss()

    for i in range(epoch):

      ### train initial GAN ###
      g_running_loss = 0
      d_running_loss = 0
      for _, (images, _) in enumerate(dataloader):
        n = images.size()[0]
        images = images.to(self.device)

        for l in range(len(self.image_sizes) - 1):
          images = F.interpolate(images, size=(self.image_sizes[l + 1], self.image_sizes[l + 1]))

        g_loss, d_loss = self.gan_init.train(images.view(n, -1))
        g_running_loss += g_loss
        d_running_loss += d_loss

      self.g_scheduler_init.step()
      self.d_scheduler_init.step()
      g_running_loss /= len(dataloader)
      d_running_loss /= len(dataloader)
      self.g_losses[-1].append(g_running_loss)
      self.d_losses[-1].append(d_running_loss)
      print("[{0[epoch]}] initial Generator: {0[g_loss]}, initial Discriminator: {0[d_loss]}".format({"epoch": i + 1, "g_loss": g_running_loss, "d_loss": d_running_loss}))

      ### train k-th CGAN ###
      for k in range(len(self.image_sizes) - 1):
        g_running_loss = 0
        d_running_loss = 0

        for _, (images, _) in enumerate(dataloader):
          n = images.size()[0]
          images = images.to(self.device)

          for l in range(k):
            images = F.interpolate(images, size=(self.image_sizes[l + 1], self.image_sizes[l + 1]))
          coarse = F.interpolate(F.interpolate(images, size=(self.image_sizes[k + 1], self.image_sizes[k + 1])), size=(self.image_sizes[k], self.image_sizes[k]))

          g_loss, d_loss = self.gans[k].train(images - coarse, coarse)
          g_running_loss += g_loss
          d_running_loss += d_loss

        self.g_schedulers[k].step()
        self.d_schedulers[k].step()
        g_running_loss /= len(dataloader)
        d_running_loss /= len(dataloader)
        self.g_losses[k].append(g_running_loss)
        self.d_losses[k].append(d_running_loss)
        print("[{0[epoch]}] Generator{0[layer]}: {0[g_loss]}, Discriminator{0[layer]}: {0[d_loss]}".format({"epoch": i + 1, "layer": k, "g_loss": g_running_loss, "d_loss": d_running_loss}))
    return

  def sample(self, n):
    images = self.gan_init.sample(n).view(n, 1, self.image_sizes[-1], self.image_sizes[-1])
    for k in reversed(range(len(self.image_sizes) - 1)):
      images = F.interpolate(images, size=(self.image_sizes[k], self.image_sizes[k]))
      images = images + self.gans[k].sample(n, images)
    return images

  def set_state(self, state):
    self.gan_init.set_state(state["gan_init"])
    for k in range(len(self.gans)):
      self.gans[k].set_state(state["gans"][k])
    self.g_scheduler_init.load_state_dict(state["g_scheduler_init_state_dict"])
    self.d_scheduler_init.load_state_dict(state["d_scheduler_init_state_dict"])
    for k in range(len(self.g_schedulers)):
      self.g_schedulers[k].load_state_dict(state["g_schedulers_state_dict"][k])
    for k in range(len(self.d_schedulers)):
      self.d_schedulers[k].load_state_dict(state["d_schedulers_state_dict"][k])
    for k in range(len(self.image_sizes)):
      self.g_losses[k].extend(state["g_losses"][k])
      self.d_losses[k].extend(state["d_losses"][k])
    return

  def get_state(self):
    state = {"gan_init": self.gan_init.get_state(),
              "gans": [gan.get_state() for gan in self.gans],
              "g_scheduler_init_state_dict": self.g_scheduler_init.state_dict(),
              "d_scheduler_init_state_dict": self.d_scheduler_init.state_dict(),
              "g_schedulers_state_dict": [scheduler.state_dict() for scheduler in self.g_schedulers],
              "d_schedulers_state_dict": [scheduler.state_dict() for scheduler in self.d_schedulers],
              "g_losses": self.g_losses,
              "d_losses": self.d_losses}
    return state
