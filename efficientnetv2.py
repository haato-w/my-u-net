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
# ===== EfficientNetV2 blocks & model (drop-in replacement for DNRUNet) =====
def _gn(num_channels, groups=8):
    return nn.GroupNorm(num_groups=min(groups, num_channels), num_channels=num_channels)

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        mid = max(1, int(in_ch * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, in_ch, 1)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        s = self.pool(x)
        s = self.act(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class MBConv(nn.Module):
    """MBConv (expansion -> dwconv -> SE -> proj), k=3, with GroupNorm + SiLU."""
    def __init__(self, in_ch, out_ch, stride=1, expand=4, k=3, se_ratio=0.25):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        mid = in_ch * expand
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            _gn(mid), nn.SiLU(inplace=True)
        ) if expand != 1 else nn.Identity()
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, k, stride=stride, padding=k//2, groups=mid, bias=False),
            _gn(mid), nn.SiLU(inplace=True)
        )
        self.se = SqueezeExcite(mid, se_ratio=se_ratio)
        self.project = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            _gn(out_ch)
        )
    def forward(self, x):
        h = x
        x = self.expand(x)
        x = self.dw(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_res:
            x = x + h
        return x

class FusedMBConv(nn.Module):
    """Fused-MBConv (regular conv k3 + optional expand) per EfficientNetV2."""
    def __init__(self, in_ch, out_ch, stride=1, expand=4, k=3):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        mid = in_ch * expand
        # fused conv (like expand+dw fused into a single conv)
        self.fused = nn.Sequential(
            nn.Conv2d(in_ch, mid, k, stride=stride, padding=k//2, bias=False),
            _gn(mid), nn.SiLU(inplace=True)
        ) if expand != 1 else nn.Identity()
        self.stem = None
        if expand == 1:
            # when expand=1, do a single conv from in_ch -> out_ch (k3)
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k//2, bias=False),
                _gn(out_ch), nn.SiLU(inplace=True)
            )
        self.project = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            _gn(out_ch)
        ) if expand != 1 else nn.Identity()
    def forward(self, x):
        if self.stem is not None:
            h = x
            x = self.stem(x)
            if self.use_res:
                x = x + h
            return x
        h = x
        x = self.fused(x)
        x = self.project(x)
        if self.use_res:
            x = x + h
        return x

class UpBlock(nn.Module):
    """Upsample by 2, align with skip, then fuse and refine."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.block = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.SiLU(inplace=True),
        )
    def forward(self, x, skip):
        x = self.up(x)
        # pad to match spatial
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, (0, dx, 0, dy))
        skip = self.skip_proj(skip)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class DNREfficientNetV2(nn.Module):
    """
    EfficientNetV2-S encoder + lightweight decoder for image-to-image.
    Input: [B, c_in, H, W]  -> Output: [B, 3, H, W]
    """
    def __init__(self, c_in, num_classes=3, se_ratio=0.25):
        super().__init__()
        act = nn.SiLU(inplace=True)
        # stem: let it accept arbitrary c_in
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 24, kernel_size=3, stride=2, padding=1, bias=False),
            _gn(24), act
        )
        # stage definitions follow Table 4 (EfficientNetV2-S)
        def make_stage(op, in_ch, out_ch, stride, layers, expand, use_se):
            blocks = []
            for i in range(layers):
                s = stride if i == 0 else 1
                if op == "fused":
                    blocks.append(FusedMBConv(in_ch if i == 0 else out_ch, out_ch, stride=s, expand=expand, k=3))
                else:
                    blocks.append(MBConv(in_ch if i == 0 else out_ch, out_ch, stride=s, expand=expand, k=3,
                                         se_ratio=se_ratio if use_se else 0.0))
                in_ch = out_ch
            return nn.Sequential(*blocks)

        # Stages (channels per Table 4)
        self.stage1 = make_stage("fused", 24, 24, stride=1, layers=2, expand=1, use_se=False)
        self.stage2 = make_stage("fused", 24, 48, stride=2, layers=4, expand=4, use_se=False)
        self.stage3 = make_stage("fused", 48, 64, stride=2, layers=4, expand=4, use_se=False)
        self.stage4 = make_stage("mbconv", 64, 128, stride=2, layers=6, expand=4, use_se=True)
        self.stage5 = make_stage("mbconv", 128, 160, stride=1, layers=9, expand=6, use_se=True)
        self.stage6 = make_stage("mbconv", 160, 256, stride=2, layers=15, expand=6, use_se=True)

        # Decoder (skip from stage6->5->4->3->2)
        self.up5 = UpBlock(in_ch=256, skip_ch=160, out_ch=160)
        self.up4 = UpBlock(in_ch=160, skip_ch=128, out_ch=128)
        self.up3 = UpBlock(in_ch=128, skip_ch=64,  out_ch=64)
        self.up2 = UpBlock(in_ch=64,  skip_ch=48,  out_ch=48)
        # last up to reach near input scale (stem/stride=2 & stage2/3/4/6 downsampled -> total /32)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            _gn(24), nn.SiLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1, bias=False),
            _gn(24), nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.stem(x)         # /2
        x1 = self.stage1(x0)      # /2
        x2 = self.stage2(x1)      # /4
        x3 = self.stage3(x2)      # /8
        x4 = self.stage4(x3)      # /16
        x5 = self.stage5(x4)      # /16
        x6 = self.stage6(x5)      # /32

        # Decoder with skips
        d5 = self.up5(x6, x5)     # /16
        d4 = self.up4(d5, x4)     # /8
        d3 = self.up3(d4, x3)     # /4
        d2 = self.up2(d3, x2)     # /2
        d1 = self.up1(d2)         # ~ /1 (align to input by pad if needed)

        # # align to original H,W precisely
        # dy, dx = x.size(2) - d1.size(2), x.size(3) - d1.size(3)
        # if dy != 0 or dx != 0:
        #     d1 = F.pad(d1, (0, dx, 0, dy))

        # rgb = torch.sigmoid(self.head(d1))
        # return rgb

        # 入力と同じ空間サイズへ“補間”で合わせる（ゼロパッドしない）
        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        rgb = torch.sigmoid(self.head(d1))
        return rgb

# ===== end of EfficientNetV2 model =====


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
model = DNREfficientNetV2(c_in=C_in).to(device)
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

