import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):
  def __init__(self, noise_size, label_size, out_size):
    super(Generator, self).__init__()
    # Dropout is applied to all layers for now.
    self.noise_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(noise_size, 200),
        nn.ReLU())
    self.label_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(label_size, 1000),
        nn.ReLU())
    self.concat_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1200, 1200),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1200, out_size),
        nn.Sigmoid())

  def forward(self, noise, label):
    noise = self.noise_layer(noise)
    label = self.label_layer(label)
    x = self.concat_layer(torch.cat((noise, label), dim=1))
    return x


class Discriminator(nn.Module):
  def __init__(self, in_size, label_size):
    # Dropout is applied to all layers for now.
    super(Discriminator, self).__init__()
    # Maxout layer with 240 units and 5 pieces is used in the original paper.
    self.data_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_size, 240),
        nn.ReLU())
    # Maxout layer with 50 units and 5 pieces is used in the original paper.
    self.label_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(label_size, 50),
        nn.ReLU())
    # Maxout layer with 240 units and 4 pieces is used in the original paper.
    self.concat_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(290, 240),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(240, 1),
        nn.Sigmoid())

  def forward(self, data, label):
    data = self.data_layer(data)
    label = self.label_layer(label)
    x = self.concat_layer(torch.cat((data, label), dim=1))
    return x


class CGAN:
  def __init__(self, img_size, label_size, noise_size=100):
    self.noise_size = noise_size
    self.label_size = label_size

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.generator = Generator(noise_size, label_size, img_size).to(self.device)
    self.discriminator = Discriminator(img_size, label_size).to(self.device)

    # Momentum is increased from 0.5 up to 0.7 in the original paper.
    self.g_optimizer = optim.SGD(self.generator.parameters(), lr=0.1, momentum=0.5)
    self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=0.1, momentum=0.5)
    self.lr_lambda = lambda epoch: max(0.99996 ** epoch, 0.00001)
    self.g_scheduler = optim.lr_scheduler.LambdaLR(self.g_optimizer, self.lr_lambda)
    self.d_scheduler = optim.lr_scheduler.LambdaLR(self.d_optimizer, self.lr_lambda)
    self.g_losses = []
    self.d_losses = []

  def train(self, dataloader, epoch):
    criterion = nn.BCELoss()

    for i in range(epoch):
      g_running_loss = 0
      d_running_loss = 0

      for _, (images, labels) in enumerate(dataloader):
        n = images.size()[0]
        labels = torch.stack([torch.eye(self.label_size)[label] for label in labels]).to(self.device)

        ### train discriminator ###
        self.d_optimizer.zero_grad()
        fake_images = self.generator(torch.rand(n, self.noise_size, device=self.device), labels)
        real_images = images.view(n, -1).to(self.device)
        fake_labels = torch.zeros((n, ), device=self.device)
        real_labels = torch.ones((n, ), device=self.device)

        d_fake_loss = criterion(self.discriminator(fake_images, labels).view(-1), fake_labels)
        d_real_loss = criterion(self.discriminator(real_images, labels).view(-1), real_labels)
        d_loss = d_fake_loss + d_real_loss
        d_running_loss += d_loss.item()
        d_loss.backward()
        self.d_optimizer.step()

        ### train generator ###
        self.g_optimizer.zero_grad()
        fake_images = self.generator(torch.rand(n, self.noise_size, device=self.device), labels)
        fake_labels = torch.ones((n, ), device=self.device)

        g_loss = criterion(self.discriminator(fake_images, labels).view(-1), fake_labels)
        g_running_loss += g_loss.item()
        g_loss.backward()
        self.g_optimizer.step()

      self.g_scheduler.step()
      self.d_scheduler.step()
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
    self.g_scheduler.load_state_dict(state["g_scheduler_state_dict"])
    self.d_scheduler.load_state_dict(state["d_scheduler_state_dict"])
    self.g_losses.extend(state["g_losses"])
    self.d_losses.extend(state["d_losses"])
    return

  def get_state(self):
    state = {"generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
            "d_optimizer_state_dict": self.d_optimizer.state_dict(),
            "g_scheduler_state_dict": self.g_scheduler.state_dict(),
            "d_scheduler_state_dict": self.d_scheduler.state_dict(),
            "g_losses": self.g_losses,
            "d_losses": self.d_losses}
    return state
