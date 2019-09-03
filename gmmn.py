import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):
  def __init__(self, noise_size, out_size):
    # The number of hidden nodes are unclear.
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(noise_size, 1024)
    self.fc2 = nn.Linear(1024, 1024)
    self.fc3 = nn.Linear(1024, 1024)
    self.fc4 = nn.Linear(1024, 1024)
    self.fc5 = nn.Linear(1024, out_size)

  def forward(self, x):
    x = F.relu(self.fc1(x), inplace=True)
    x = F.relu(self.fc2(x), inplace=True)
    x = F.relu(self.fc3(x), inplace=True)
    x = F.relu(self.fc4(x), inplace=True)
    x = torch.sigmoid(self.fc5(x))
    return x


class MMDs(nn.Module):
  def __init__(self, sigmas):
    super(MMDs, self).__init__()
    self.sigmas = sigmas

  def dist(self, X, Y=None):
    if Y is None:
      Y = X
    X_norm = (X ** 2).sum(1).view(-1, 1)
    Y_norm = (Y ** 2).sum(1).view(1, -1)
    dist = X_norm + Y_norm - 2 * torch.mm(X, Y.t())
    return dist

  def mmd(self, X, Y, sigma):
    N = len(X)
    M = len(Y)
    xx = self.dist(X)
    yy = self.dist(Y)
    xy = self.dist(X, Y)
    xx = torch.exp(- xx / (2 * sigma)).sum() / (N ** 2)
    yy = torch.exp(- yy / (2 * sigma)).sum() / (M ** 2)
    xy = torch.exp(- xy / (2 * sigma)).sum() / (N * M)
    mmd = torch.sqrt(xx + yy - 2 * xy)
    return mmd

  def forward(self, X, Y):
    mmds = 0
    for sigma in self.sigmas:
      mmds += self.mmd(X, Y, sigma)
    return mmds


class GMMN:
  def __init__(self, dataloader,
               noise_size=100,
               noise_generator=None,
               generator=None,
               optimizer=None
              ):
    self.dataloader = dataloader
    data, _ = next(iter(dataloader))
    self.channel = data.size()[1]
    self.height = data.size()[2]
    self.width = data.size()[3]

    self.noise_size = noise_size
    if noise_generator is None:
      self.noise_generator = lambda n: torch.rand(n, self.noise_size).uniform_(-1, 1)
    else:
      self.noise_generator = noise_generator

    self.out_size = self.channel * self.height * self.width

    if generator is None:
      self.generator = Generator(self.noise_size, self.out_size)
    else:
      self.generator = generator

    # Authors use SGD with momentum.
    if optimizer is None:
      self.optimizer = optim.Adam(self.generator.parameters())
    else:
      self.optimizer = optimizer

    self.losses = []

  def train(self, epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.generator = self.generator.to(device)

    # The bandwidths are unclear.
    criterion = MMDs([1, 2, 5, 10, 20, 50])

    for i in range(epoch):
      running_loss = 0

      for _, (images, _) in enumerate(self.dataloader):
        n = images.size()[0]
        images = images.view(n, -1).to(device)

        self.optimizer.zero_grad()
        noise = self.noise_generator(n).to(device)
        fake_images = self.generator(noise)

        loss = criterion(images, fake_images)
        running_loss += loss.item()
        loss.backward()
        self.optimizer.step()

      running_loss /= len(self.dataloader)
      self.losses.append(running_loss)
      print("[%d] loss: %f" % (i, running_loss))

    return
