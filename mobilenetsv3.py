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
# ===== MobileNetV3 blocks =====
class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class SE(nn.Module):
    def __init__(self, ch, se_ratio=0.25):
        super().__init__()
        hidden = max(1, int(ch * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=True),
            nn.Hardsigmoid(inplace=True) if hasattr(nn, "Hardsigmoid") else nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, act):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1)
        self.act = act
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class InvertedResidualV3(nn.Module):
    """
    MobileNetV3 inverted residual block with (optional) SE and h-swish.
    """
    def __init__(self, in_ch, out_ch, k, s, expand, use_se, use_hswish):
        super().__init__()
        act = HSwish() if use_hswish else nn.ReLU(inplace=True)
        mid = int(round(in_ch * expand))
        self.use_res = (s == 1 and in_ch == out_ch)

        layers = []
        if expand != 1.0:
            layers.append(ConvBNAct(in_ch, mid, 1, 1, 0, act))
        # depthwise
        layers.append(nn.Conv2d(mid, mid, k, s, k//2, groups=mid, bias=False))
        layers.append(nn.BatchNorm2d(mid, eps=1e-3, momentum=0.1))
        layers.append(act)
        if use_se:
            layers.append(SE(mid))
        # project (linear)
        layers.append(nn.Conv2d(mid, out_ch, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.1))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y

# ===== Backbone (V3-Large-ish, 変更: 任意in_ch対応) =====
class MobileNetV3Backbone(nn.Module):
    """
    Returns multi-scale features at strides 4, 8, 16, 32.
    Layout is adapted from MobileNetV3-Large spec; some channels slightly adjusted to keep code concise.
    """
    def __init__(self, in_ch=16, width=1.0):
        super().__init__()
        w = lambda c: int(round(c * width))

        self.stem = ConvBNAct(in_ch, w(16), 3, 2, 1, HSwish())   # /2
        # stage 2 (output stride 4)
        self.b2 = nn.Sequential(
            InvertedResidualV3(w(16),  w(16), 3, 1, 1.0, False, False),
            InvertedResidualV3(w(16),  w(24), 3, 2, 4.0, False, False),  # /4
            InvertedResidualV3(w(24),  w(24), 3, 1, 3.0, False, False),
        )
        # stage 3 (OS 8)
        self.b3 = nn.Sequential(
            InvertedResidualV3(w(24),  w(40), 5, 2, 3.0, True,  False),  # /8
            InvertedResidualV3(w(40),  w(40), 5, 1, 3.0, True,  False),
            InvertedResidualV3(w(40),  w(40), 5, 1, 3.0, True,  False),
        )
        # stage 4 (OS 16)
        self.b4 = nn.Sequential(
            InvertedResidualV3(w(40),  w(80), 3, 2, 6.0, False, True),   # /16
            InvertedResidualV3(w(80),  w(80), 3, 1, 2.5, False, True),
            InvertedResidualV3(w(80),  w(80), 3, 1, 2.3, False, True),
            InvertedResidualV3(w(80),  w(112),3, 1, 6.0, True,  True),
            InvertedResidualV3(w(112), w(112),3, 1, 6.0, True,  True),
        )
        # stage 5 (OS 32)
        self.b5 = nn.Sequential(
            InvertedResidualV3(w(112), w(160),5, 2, 6.0, True,  True),   # /32
            InvertedResidualV3(w(160), w(160),5, 1, 6.0, True,  True),
            InvertedResidualV3(w(160), w(160),5, 1, 6.0, True,  True),
        )

        self.out_dims = (w(24), w(40), w(112), w(160))  # C4,C8,C16,C32

    def forward(self, x):
        x = self.stem(x)                # /2
        c4 = self.b2(x)                 # /4
        c8 = self.b3(c4)                # /8
        c16 = self.b4(c8)               # /16
        c32 = self.b5(c16)              # /32
        return c4, c8, c16, c32

# ===== Lightweight Decoder (LR-ASPP-ish + progressive upsample) =====
# class LiteDecoder(nn.Module):
#     def __init__(self, c4, c8, c16, c32, out_ch=3):
#         super().__init__()
#         self.act = HSwish()

#         def proj(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.1),
#                 nn.ReLU(inplace=True)
#             )

#         self.p32 = proj(c32, 128)
#         self.p16 = proj(c16, 128)
#         self.p8  = proj(c8,  64)
#         self.p4  = proj(c4,  32)

#         # depthwise separable conv after fusion (軽量)
#         def dws(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False),
#                 nn.BatchNorm2d(in_c, eps=1e-3, momentum=0.1),
#                 self.act,
#                 nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.1),
#                 self.act,
#             )
#         self.f16 = dws(128+128, 128)
#         self.f8  = dws(128+64,   96)
#         self.f4  = dws(96+32,    64)

#         self.head = nn.Sequential(
#             nn.Conv2d(64, 32, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(32, eps=1e-3, momentum=0.1),
#             self.act,
#             nn.Conv2d(32, out_ch, 1, 1, 0)
#         )

#     def forward(self, c4, c8, c16, c32):
#         x32 = self.p32(c32)                               # /32
#         x16 = self.p16(c16)                               # /16
#         x  = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
#         x  = self.f16(torch.cat([x, x16], dim=1))         # /16

#         x8 = self.p8(c8)                                  # /8
#         x  = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
#         x  = self.f8(torch.cat([x, x8], dim=1))           # /8

#         x4 = self.p4(c4)                                  # /4
#         x  = F.interpolate(x, size=x4.shape[-2:], mode='bilinear', align_corners=False)
#         x  = self.f4(torch.cat([x, x4], dim=1))           # /4

#         # 最終的に入力解像度へ
#         H = c4.size(2) * 4
#         W = c4.size(3) * 4
#         x  = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
#         x  = self.head(x)
#         return torch.sigmoid(x)
# --- 変更1: LiteDecoder を出力サイズ指定に対応 ---
class LiteDecoder(nn.Module):
    def __init__(self, c4, c8, c16, c32, out_ch=3):
        super().__init__()
        self.act = HSwish()
        def proj(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.1),
                nn.ReLU(inplace=True)
            )
        self.p32 = proj(c32, 128)
        self.p16 = proj(c16, 128)
        self.p8  = proj(c8,  64)
        self.p4  = proj(c4,  32)

        def dws(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c, eps=1e-3, momentum=0.1),
                self.act,
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.1),
                self.act,
            )
        self.f16 = dws(128+128, 128)
        self.f8  = dws(128+64,   96)
        self.f4  = dws(96+32,    64)

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.1),
            self.act,
            nn.Conv2d(32, out_ch, 1, 1, 0)
        )

    # ここを変更: output_size を受け取り、最終アップサンプルに使う
    def forward(self, c4, c8, c16, c32, output_size):
        x32 = self.p32(c32)                               # /32
        x16 = self.p16(c16)                               # /16
        x  = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
        x  = self.f16(torch.cat([x, x16], dim=1))         # /16

        x8 = self.p8(c8)                                  # /8
        x  = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x  = self.f8(torch.cat([x, x8], dim=1))           # /8

        x4 = self.p4(c4)                                  # /4
        x  = F.interpolate(x, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x  = self.f4(torch.cat([x, x4], dim=1))           # /4

        # 最終は入力と同じサイズに合わせる（ここがポイント）
        x  = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        x  = self.head(x)
        return torch.sigmoid(x)

# ===== Full model =====
# class DNRMobileNetV3(nn.Module):
#     def __init__(self, c_in=16, width=1.0, out_ch=3):
#         super().__init__()
#         self.backbone = MobileNetV3Backbone(in_ch=c_in, width=width)
#         c4, c8, c16, c32 = self.backbone.out_dims
#         self.decoder = LiteDecoder(c4, c8, c16, c32, out_ch=out_ch)
#     def forward(self, x):
#         c4, c8, c16, c32 = self.backbone(x)
#         return self.decoder(c4, c8, c16, c32)
# --- 変更2: DNRMobileNetV3.forward で入力サイズを渡す ---
class DNRMobileNetV3(nn.Module):
    def __init__(self, c_in=16, width=1.0, out_ch=3):
        super().__init__()
        self.backbone = MobileNetV3Backbone(in_ch=c_in, width=width)
        c4, c8, c16, c32 = self.backbone.out_dims
        self.decoder = LiteDecoder(c4, c8, c16, c32, out_ch=out_ch)

    def forward(self, x):
        H_in, W_in = x.size(2), x.size(3)
        c4, c8, c16, c32 = self.backbone(x)
        return self.decoder(c4, c8, c16, c32, output_size=(H_in, W_in))


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
model = DNRMobileNetV3(c_in=C_in).to(device)
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

