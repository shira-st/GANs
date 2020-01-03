import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
  def __init__(self, noise_size, out_size):
    super(Generator, self).__init__()
    # The number of hidden nodes are roughly chosen.
    self.model = nn.Sequential(
        nn.Linear(noise_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, out_size),
        nn.Sigmoid())

  def forward(self, x):
    return self.model(x)


def distance(X, Y):
  X_norm = (X ** 2).sum(1).view(-1, 1)
  Y_norm = (Y ** 2).sum(1).view(1, -1)
  dist = X_norm + Y_norm - 2 * torch.mm(X, Y.t())
  return dist


def MMD(X, Y, sigma):
  N = X.shape[0]
  M = Y.shape[0]
  xx = torch.exp(- distance(X, X) / (2 * sigma)).sum() / (N ** 2)
  yy = torch.exp(- distance(Y, Y) / (2 * sigma)).sum() / (M ** 2)
  xy = torch.exp(- distance(X, Y) / (2 * sigma)).sum() / (N * M)
  mmd = torch.sqrt(xx + yy - 2 * xy)
  return mmd


class GMMN:
  # The dimensionality of noise and bandwidths are roughly chosen.
  def __init__(self, img_size, noise_size=100, sigmas=(1, 2, 5, 10, 20, 50)):
    self.noise_size = noise_size
    self.sigmas = sigmas

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.generator = Generator(noise_size, img_size).to(self.device)

    # The learning rate and momentum are roughly chosen
    self.optimizer = optim.SGD(self.generator.parameters(), lr=0.01, momentum=0.9)
    self.losses = []

  def train(self, dataloader, epoch):
    criterion = lambda X, Y: torch.stack([MMD(X, Y, sigma) for sigma in self.sigmas]).sum()

    for i in range(epoch):
      running_loss = 0

      for _, (images, _) in enumerate(dataloader):
        n = images.size()[0]

        self.optimizer.zero_grad()
        fake_images = self.generator(torch.rand(n, self.noise_size, device=self.device).uniform_(-1, 1))
        real_images = images.view(n, -1).to(self.device)

        loss = criterion(real_images, fake_images)
        running_loss += loss.item()
        loss.backward()
        self.optimizer.step()

      running_loss /= len(dataloader)
      self.losses.append(running_loss)
      print("[%d] loss: %f" % (i + 1, running_loss))
    return

  def set_state(self, state):
    self.generator.load_state_dict(state["generator_state_dict"])
    self.optimizer.load_state_dict(state["optimizer_state_dict"])
    self.losses.extend(state["losses"])
    return

  def get_state(self):
    state = {"generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses}
    return state
