import torch
import os
import sys
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np




class bbr_model(nn.Module):
    def __init__(self, computational_device, mode=21):
        super(bbr_model, self).__init__()
        self.mode = mode

        #convolutions
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=computational_device)


        # atencja
        self.query = torch.nn.Parameter(torch.empty(1, 1, 64))
        torch.nn.init.xavier_normal_(self.query)

        # linear fc head
        self.fc1 = nn.Linear(64, 16, device=computational_device)
        self.fc2 = nn.Linear(16, 4, device=computational_device)

        # activation/misc
        self.pool22 = nn.MaxPool2d(2, stride=2)
        self.dropc = torch.nn.Dropout(p=0.15)
        self.dropl = torch.nn.Dropout(p=0.3)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten()

    def forward(self, img_data):
        # convolutions
        x = self.conv1(img_data)
        if self.mode in [218, 170, 114, 70, 40]:
            x = self.pool22(x)
        else:
            x = self.relu(x)
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        if self.mode in [218, 170, 114, 70]:
            x = self.pool22(x)
        else:
            x = self.relu(x)
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        if self.mode in [218, 170, 114]:
            x = self.pool22(x)
        else:
            x = self.relu(x)
        x = self.relu(self.conv6(x))
        x = self.conv7(x)
        if self.mode in [218, 170]:
            x = self.pool22(x)
        else:
            x = self.relu(x)
        x = self.relu(self.conv8(x))
        x = self.conv9(x)
        if self.mode in [218]:
            x = self.pool22(x)
        else:
            x = self.relu(x)
        x = self.conv10(x)

        features = torch.flatten(x, 2)  # batch x 64 x dim^2

        # attention
        if self.mode == 218:
            response = torch.flatten(features, 1)
        else:
            energy = self.query @ features  # batch x 1 x dim^2
            weights = torch.nn.functional.softmax(energy, 2)  # batch x 1 x dim^2
            response = features @ weights.transpose(1, 2)  # batch x 64 x 1
            response = torch.flatten(response, 1)  # batch x 64


        output = self.fc1(response)
        output = self.fc2(self.relu(output))

        return output


def main():
    model_name = sys.argv[1]

    if torch.cuda.is_available():
        computational_device = 'cuda'
    else:
        computational_device = 'cpu'

    save_path = "./models/" + model_name

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Initialize model
    model = bbr_model(computational_device)

    weights = {
        'epoch': 0,
        'model': model.state_dict(),
        'optimizer': 0}
    torch.save(weights, save_path + '/model_scripted.pt')


if __name__ == "__main__":
    main()
