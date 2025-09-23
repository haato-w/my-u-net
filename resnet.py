import os
import sys
import time
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imageio

torch.cuda.reset_peak_memory_stats() # For memory stats

"""
neural texture & ResNet(Convolution) renderer
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class DNRResNet(nn.Module):
    def __init__(self, c_in, base_ch=64, layers=[2,2,2,2], num_classes=3):
        super().__init__()
        self.in_ch = base_ch

        # ここを弱める: stride=1、maxpoolを外す
        self.conv1 = nn.Conv2d(c_in, base_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()  # ← MaxPool2d を使わない

        # ResNet stages（layer2/3/4 の先頭だけ stride=2 → 合計 /8）
        self.layer1 = self._make_layer(base_ch, layers[0], stride=1)
        self.layer2 = self._make_layer(base_ch*2, layers[1], stride=2)  # /2
        self.layer3 = self._make_layer(base_ch*4, layers[2], stride=2)  # /4
        self.layer4 = self._make_layer(base_ch*8, layers[3], stride=2)  # /8

        # /8 → ×8 で元解像度へ（3 回）
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch*2, base_ch,   kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def _make_layer(self, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [ResidualBlock(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        H_in, W_in = x.shape[-2], x.shape[-1]  # 入力解像度を保持

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.up(x)

        # 最後に絶対に入力サイズへ揃える（奇数・非倍数でもOK）
        x = F.interpolate(x, size=(H_in, W_in), mode='bilinear', align_corners=False)

        rgb = torch.sigmoid(self.out_conv(x))
        return rgb


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps') # for MAC
print("device: ", device)


"""
Config
"""
EPOCH_NUM = 1000
video_interval = 50
# H, W = 512, 512
H, W = 890, 1600
CHANEL = 3
gt_image_path = 'resources/checkered_boots_cropped.png'
result_dir = 'result'

"""
Preparating directory
"""
if 1 < len(sys.argv):
    experiment_name = sys.argv[1]
else:
    experiment_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    print(f"No experiment name is given. Using datatime {experiment_name} as experiment name.")

output_dir = os.path.join(result_dir, experiment_name)
os.makedirs(output_dir)
in_training_imgs_dir = os.path.join(output_dir, 'in_training_imgs')
os.makedirs(in_training_imgs_dir)

"""
Loading and processing GT image
"""
gt_image = Image.open(gt_image_path)
gt_image = gt_image.resize((W, H)) # fit to kernel_size
gt_image = gt_image.convert('RGB')
gt_image_array = np.array(gt_image)
gt_image_array = gt_image_array / 255.0
# Creating tensor of ground-truth image
target_tensor = torch.tensor(gt_image_array, dtype=torch.float32, device=device)
# Saving processed gt image
plt.imsave(os.path.join(output_dir, "processed_gt_image.png"), target_tensor.detach().cpu().numpy())
target_tensor = target_tensor.permute(2,0,1).unsqueeze(0)   # [1,3,H,W]

"""
Preparing model input
"""
B, C_in = 1, 16
Fscr = nn.Parameter(torch.randn(B, C_in, H, W, device=device), requires_grad=True)


# netA = DNRUNet(c_in=C_in, base_ch=64, sh_mode="broadcast", aux_ch=0)
# netB = DNRUNet(c_in=C_in, base_ch=64, sh_mode="coeff", sh_out=32, sh_reduce="sum").to(device)
model = DNRResNet(c_in=C_in, base_ch=64).to(device)
optimizer = optim.Adam([*model.parameters(), Fscr], lr=1e-3)

progress_bar = tqdm(range(1, EPOCH_NUM + 1))
progress_bar.set_description("[train]")
loss_records = []

forward_time_list = []
backward_time_list = []

for epoch in progress_bar:
    optimizer.zero_grad()

    torch.cuda.synchronize()
    start = time.perf_counter()
    pred = model(Fscr)
    torch.cuda.synchronize()
    forward_time_list.append(time.perf_counter() - start)

    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = F.mse_loss(pred, target_tensor, reduction='sum')
    loss.backward()
    torch.cuda.synchronize()
    backward_time_list.append(time.perf_counter() - start)

    optimizer.step()

    with torch.no_grad():
        loss_records.append(loss.item())

        # Displaying Loss value
        if epoch % 10 == 0:
            loss_value = {'Loss': f"{loss.item():.{5}f}"}
            progress_bar.set_postfix(loss_value)
        # Saving rendered image for training video
        if epoch % video_interval == 0:
            rendered_image = pred[0].permute(1, 2, 0).contiguous()
            output_img_array = rendered_image.cpu().detach().numpy()
            plt.imsave(os.path.join(in_training_imgs_dir, f'rendered_output_image_{epoch}.png'), output_img_array)

print('Training complete')

# Showing training metrics
print(f"Forward average time: {np.mean(forward_time_list):.6f} Sec")
print(f"Backward average time: {np.mean(backward_time_list):.6f} Sec")
max_mem = torch.cuda.max_memory_allocated()
print(f"Max GPU memory: {max_mem / 1024 ** 2:.2f} MB")

"""
Postprocessing
"""
print('Postprocessing...')
# Visulizing trained image
rendered_image = pred[0].permute(1, 2, 0).contiguous()
output_img_array = rendered_image.cpu().detach().numpy()
plt.imsave(os.path.join(output_dir, f'final_rendered_output_image_{epoch}.png'), output_img_array)

# save loss figure
loss_fname = "loss.png"
plt.plot(range(EPOCH_NUM), loss_records)
plt.title('Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(output_dir, loss_fname), bbox_inches='tight')

# Creating video
video_writer = imageio.get_writer(os.path.join(output_dir, 'training.mp4'), fps=2)

fnames = os.listdir(in_training_imgs_dir)
fnames.sort(key = lambda x: int(x.split('.')[0].split('_')[-1]))
for training_img_fname in fnames:
  training_img_path = os.path.join(in_training_imgs_dir, training_img_fname)
  training_img = imageio.v3.imread(training_img_path)
  video_writer.append_data(training_img)

video_writer.close()

# Removing temporary traing images
shutil.rmtree(in_training_imgs_dir)

print('Done')

