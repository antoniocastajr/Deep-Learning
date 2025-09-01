# this is a code fragment
# not meant to be ran independently

class LeNet5(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # Convolutional layers (Conv2d + AvgPool2d)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(2)

        # Fully-connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Convolutional (conv) component of the network
        x = self.conv1(x)
        x = ag.relu(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = ag.relu(x)
        x = self.avgpool2(x)

        # Flatten the output from conv layers to feed into fully connected layers
        x = x.reshape(x.shape[0], -1)

        # Fully-connected (fc) component of the network
        x = self.fc1(x)
        x = ag.relu(x)

        x = self.fc2(x)
        x = ag.relu(x)

        x = self.fc3(x)

        return x
