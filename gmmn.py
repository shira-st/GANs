import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):
  def __init__(self, in_size, out_size):
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(in_size, 1024)
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


class MMD(nn.Module):
  def __init__(self, sigmas):
    super(MMD, self).__init__()
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
    mmds = sum([self.mmd(X, Y, sigma) for sigma in self.sigmas])
    return mmds


class GMMN:
  def __init__(self, dataloader):
    self.dataloader = dataloader
    data, _ = next(iter(dataloader))
    self.channel = data.size()[1]
    self.height = data.size()[2]
    self.width = data.size()[3]
    self.in_size = 100
    self.out_size = self.channel * self.height * self.width
    self.f = Generator(self.in_size, self.out_size)
    self.optimizer = optim.Adam(self.f.parameters())
    self.losses = []

  def train(self, epoch):

    # models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.f = self.f.to(device)

    # optimizers
    criterion = MMD([1, 2, 5, 10, 20, 50])

    # training
    for i in range(epoch):
      running_loss = 0
      for _, (images, _) in enumerate(self.dataloader):
        n = images.size()[0]

        self.optimizer.zero_grad()

        # generate data
        fake_images = self.f(torch.rand(n, self.in_size, device=device).uniform_(-1, 1))

        # backpropagation
        loss = criterion(images.view((n, -1)).to(device), fake_images)
        running_loss += loss.item()
        loss.backward()
        self.optimizer.step()

      running_loss /= len(self.dataloader)
      self.losses.append(running_loss)
      print("[%d] loss: %f" % (i, running_loss))

    return
