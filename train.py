import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chess_net import ChessNet

# Assume `ChessDataset` is your custom Dataset class
from chess_dataset import ChessDataset


def train():
    net = ChessNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    # Load the dataset from a PGN file
    chess_dataset = ChessDataset('games.pgn')
    trainloader = DataLoader(chess_dataset, batch_size=4, shuffle=True)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), 'chess_net.pth')


if __name__ == "__main__":
    train()
