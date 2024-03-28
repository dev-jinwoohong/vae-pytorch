import os

import torch
from torch import nn, optim
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import image_annotate, image_to_gif

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=False, num_workers=4, pin_memory=True)


class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2_mu = nn.Linear(h_dim, latent_dim)
        self.fc2_var = nn.Linear(h_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        out = F.relu(self.fc1(x))
        return self.fc2_mu(out), self.fc2_var(out)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x):
        out = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(out))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD


def train(epochs):
    model.train()
    train_loss = 0
    for idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, log_var = model(data)
        loss = loss_function(recon, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epochs, train_loss / len(train_loader.dataset)))


def test(epochs, save_path):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon, mu, log_var = model(data)
            test_loss += loss_function(recon, data, mu, log_var).item()
            if idx == 0:
                n = min(data.size(0), 10)
                comparison = torch.cat([data[:n],
                                        recon.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           os.path.join(save_path, 'epoch_' + str(epochs) + '.png'), nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    sample_path = './samples_test'
    result_path = './results_test'

    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    s = torch.randn(64, 20).to(device)
    for epoch in range(0, 20):
        train(epoch)
        test(epoch, result_path)
        if epoch % 10 == 0:
            sample = model.decode(s).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       os.path.join(sample_path, 'sample_{}_result.png'.format(epoch)))

    image_annotate(image_path=result_path, font_size=30, font_type='Roboto-Black.ttf')
    image_to_gif(image_path=result_path)
