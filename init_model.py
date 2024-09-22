import torch
import os
import sys
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np

class attention_module(nn.Module):
    def __init__(self, comp_device, kernel_size=7, mode=0):
        super(attention_module, self).__init__()
        self.conv_out = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, device=comp_device)
        self.mode = mode

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)

        combined = torch.cat([max, avg], dim=1)
        attention_map = torch.sigmoid(self.conv_out(combined))
        if self.mode != 0:
            image = attention_map.detach()
            save_image(image, 'attention_map.png')
        return x * attention_map




class bbr_model(nn.Module):
    def __init__(self, computational_device, mode=0, mode_2=0):
        super(bbr_model, self).__init__()
        self.mode = mode
        self.mode_2 = mode_2

        #conwolucje
        self.conv1 = nn.Conv2d(1, 6, kernel_size=11, stride=2, padding=0, device=computational_device)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=9, stride=1, padding=0, device=computational_device)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=7, stride=1, padding=0, device=computational_device)
        self.conv4 = nn.Conv2d(24, 36, kernel_size=5, stride=1, padding=0, device=computational_device)
        self.conv5 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv6 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv7 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv8 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv9 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv10 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv11 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv12 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.attention = attention_module(comp_device=computational_device, kernel_size=9, mode=self.mode)
        self.conv_last = nn.Conv2d(36, 1, kernel_size=3, stride=1, padding=0, device=computational_device)

        #druga gałąź konwolucji
        self.conv5_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv6_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv7_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv8_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv9_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv10_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv11_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.conv12_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0, device=computational_device)
        self.attention_2 = attention_module(comp_device=computational_device, kernel_size=9, mode=self.mode_2)
        self.conv_last_2 = nn.Conv2d(36, 1, kernel_size=3, stride=1, padding=0, device=computational_device)


        # activation/misc

        self.pool22 = nn.MaxPool2d(2, stride=2)
        self.dropc = torch.nn.Dropout(p=0.15)
        self.dropl = torch.nn.Dropout(p=0.3)
        self.lrelu = torch.nn.LeakyReLU()
        self.flatten = torch.nn.Flatten()

        # linear fc head
        self.fc1 = nn.Linear(60*60*1 + 21*21*1, 1024, device=computational_device)
        self.fc2 = nn.Linear(1024, 512, device=computational_device)
        self.fc3 = nn.Linear(512, 128, device=computational_device)
        self.fc4 = nn.Linear(128, 30, device=computational_device)
        self.fc5 = nn.Linear(32, 8, device=computational_device)
        self.fc6 = nn.Linear(8, 4, device=computational_device)

    def forward(self, img_data,  location_data):

        if location_data.dim() == 1:
            location_data = location_data.unsqueeze(0)

        # konwolucje 1
        x = self.lrelu(self.conv1(img_data))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x2 = self.pool22(x)

        x = self.dropc(x)
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = self.dropc(x)
        x = self.lrelu(self.conv10(x))
        x = self.lrelu(self.conv11(x))
        x = self.lrelu(self.conv12(x))
        x = self.attention(x)
        x = self.conv_last(x)
        x = torch.flatten(x, 1, 3)

        # konwolucje 2
        x2 = self.dropc(x2)
        x2 = self.lrelu(self.conv5_2(x2))
        x2 = self.lrelu(self.conv6_2(x2))
        x2 = self.lrelu(self.conv7_2(x2))
        x2 = self.lrelu(self.conv8_2(x2))
        x2 = self.lrelu(self.conv9_2(x2))
        x2 = self.dropc(x2)
        x2 = self.lrelu(self.conv10_2(x2))
        x2 = self.lrelu(self.conv11_2(x2))
        x2 = self.lrelu(self.conv12_2(x2))
        x2 = self.attention_2(x2)
        x2 = self.conv_last_2(x2)
        x2 = torch.flatten(x2, 1, 3)



        #head
        l_input = torch.cat([x, x2], dim=1)
        output = self.lrelu(self.fc1(l_input))
        output = self.lrelu(self.fc2(output))
        output = self.dropl(output)
        output = self.lrelu(self.fc3(output))
        output = self.lrelu(self.fc4(output))

        # wprowadzenie lokalizacji obiektu
        output = torch.cat([output, location_data], dim=1)
        output = self.lrelu(self.fc5(output))
        output = self.fc6(output)



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
