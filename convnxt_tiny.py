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
neural texture & U-Net(Convolution) renderer
"""
# ================= ConvNeXt-Tiny backbone + FPN decoder =================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """LayerNorm over channel dim for NCHW tensors."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
        # x: [B,C,H,W] -> LN over C
        return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, ls_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)  # depthwise
        self.ln = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4*dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4*dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(ls_init * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        # layer scale
        x = self.gamma.view(1, -1, 1, 1) * x
        return x + shortcut

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ln = LayerNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x):
        return self.conv(self.ln(x))

class ConvNeXtTinyBackbone(nn.Module):
    """ConvNeXt-T: C=(96,192,384,768), B=(3,3,9,3)"""
    def __init__(self, in_ch):
        super().__init__()
        dims = [96, 192, 384, 768]
        blocks = [3, 3, 9, 3]

        # stem: patchify (4x4, s=4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # stages
        self.stage2_down = Downsample(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(dims[1]) for _ in range(blocks[1])])

        self.stage3_down = Downsample(dims[1], dims[2])
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(dims[2]) for _ in range(blocks[2])])

        self.stage4_down = Downsample(dims[2], dims[3])
        self.stage4 = nn.Sequential(*[ConvNeXtBlock(dims[3]) for _ in range(blocks[3])])

        # extra blocks on stage1 (after stem)
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(blocks[0])])

        self.out_dims = dims  # for decoder

    def forward(self, x):
        # returns multi-scale features: C2(1/4), C3(1/8), C4(1/16), C5(1/32)
        c2 = self.stage1(self.stem(x))        # [B,  96, H/4,  W/4]
        c3 = self.stage2(self.stage2_down(c2))# [B, 192, H/8,  W/8]
        c4 = self.stage3(self.stage3_down(c3))# [B, 384, H/16, W/16]
        c5 = self.stage4(self.stage4_down(c4))# [B, 768, H/32, W/32]
        return c2, c3, c4, c5

class FPNDecoder(nn.Module):
    """Top-down FPN-style decoder to recover full resolution."""
    def __init__(self, in_dims, mid_ch=128, out_ch=3):
        super().__init__()
        c2,c3,c4,c5 = in_dims

        self.l5 = nn.Conv2d(c5, mid_ch, 1)
        self.l4 = nn.Conv2d(c4, mid_ch, 1)
        self.l3 = nn.Conv2d(c3, mid_ch, 1)
        self.l2 = nn.Conv2d(c2, mid_ch, 1)

        self.smooth4 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.smooth3 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.smooth2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)

        self.head = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 1),
        )

    def forward(self, feats, out_size):
        c2, c3, c4, c5 = feats
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.smooth4(p4)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.smooth3(p3)
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.smooth2(p2)
        x = self.head(p2)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return x

class DNRConvNeXtTiny(nn.Module):
    """
    Drop-in replacement for DNRUNet:
      - accepts Fscr [B, C_in, H, W]
      - returns rgb [B, 3, H, W] with sigmoid
    """
    def __init__(self, c_in, out_ch=3):
        super().__init__()
        self.backbone = ConvNeXtTinyBackbone(in_ch=c_in)
        self.decoder  = FPNDecoder(self.backbone.out_dims, mid_ch=128, out_ch=out_ch)

    def forward(self, Fscr):
        H, W = Fscr.shape[-2:]
        feats = self.backbone(Fscr)
        rgb  = self.decoder(feats, out_size=(H, W))
        return torch.sigmoid(rgb)
# ================= end of ConvNeXt-Tiny model =================


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
model = DNRConvNeXtTiny(c_in=C_in).to(device)
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

