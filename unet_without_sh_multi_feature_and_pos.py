import os
import sys
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


"""
Convolution層のセット
Convolution > Norm > ReLU
Defaultだとconvolutionによるサイズの縮小はないので元のU-Netと少し異なる
"""
def conv_block(
  in_ch: int, # 入力チャネル数 (チャネル方向にも畳み込みがされる)
  out_ch: int, # 出力チャネル数
  k: int=3, # kernel size
  s: int=1, # stride size
  p: int=1, # padding amount
  groups: int=8 # group-wise convolution (これ論文にある？)
) -> nn.Sequential:
  return nn.Sequential(
    nn.Conv2d(in_ch, out_ch, k, s, p, bias=False), 
    nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch), 
    nn.ReLU(inplace=True)
  )

"""
Convolution層2つとMaxPooling層のセット
U-Netの前半部分に使用される
"""
class Down(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.block = nn.Sequential(
      conv_block(in_ch, out_ch), 
      conv_block(out_ch, out_ch), 
    )
    self.pool = nn.MaxPool2d(2)
  
  def forward(self, x):
    x = self.block(x)
    skip = x
    x = self.pool(x)
    return x, skip

"""
UpSamplingとConvolution層2つのセット
U-Netの後半部分に使用される
"""
class Up(nn.Module):
  def __init__(self, in_ch, skip_ch, out_ch):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
    self.block = nn.Sequential(
      conv_block(out_ch * 2, out_ch), 
      conv_block(out_ch, out_ch), 
    )
  
  def forward(self, x, skip):
    x = self.up(x)
    diffY = skip.size(2) - x.size(2)
    diffX = skip.size(3) - x.size(3)
    if diffY != 0 or diffX != 0:
      x = F.pad(x, (0, diffX, 0, diffY)) # skipとxのサイズを合わせるためのpadding
    skip = self.skip_proj(skip)
    x = torch.cat([skip, x], dim=1)
    x = self.block(x)
    return x

class DNRUNet(nn.Module):
  def __init__(
    self, 
    c_in, 
    base_ch=64, 
    sh_mode="broadcast", 
    sh_out=None, 
    sh_reduce="sum", 
    aux_ch=0
  ):
    super().__init__()
    self.sh_mode = sh_mode
    
    # if sh_mode == "broadcast":
    #   enc_in = 9 * c_in + aux_ch
    #   self.sh_gate = SHGate(c_in, sh_mode="broadcast")
    # else:
    #   assert sh_out is not None
    #   gated_out = (sh_out if sh_reduce == "sum" else 9 * sh_out)
    #   enc_in = gated_out + aux_ch
    #   self.sh_gate = SHGate(c_in, out_ch=sh_out, sh_mode="coeff", reduce=sh_reduce)
    
    # Encoder (5 steps)
    self.in_conv = conv_block(c_in, base_ch)
    self.down1 = Down(base_ch, base_ch * 2)
    self.down2 = Down(base_ch * 2, base_ch * 4)
    self.down3 = Down(base_ch * 4, base_ch * 8)
    self.down4 = Down(base_ch * 8, base_ch * 16)
    
    # Bottleneck
    self.bot = nn.Sequential(
      conv_block(base_ch * 16, base_ch * 16), 
      conv_block(base_ch * 16, base_ch * 16), 
    )

    # Decoder
    self.up4 = Up(in_ch=base_ch * 16, skip_ch=base_ch * 16, out_ch=base_ch * 8)
    self.up3 = Up(in_ch=base_ch * 8, skip_ch=base_ch * 8, out_ch=base_ch * 4)
    self.up2 = Up(in_ch=base_ch * 4, skip_ch=base_ch * 4, out_ch=base_ch * 2)
    self.up1 = Up(in_ch=base_ch * 2, skip_ch=base_ch * 2, out_ch=base_ch)

    self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)
  
  def forward(self, Fscr):
    # gated = self.sh_gate(Fscr, vdirs)
    # x = gated if aux is None else torch.cat([gated, aux], dim=1)

    # U-Net
    x = self.in_conv(Fscr)
    x, s1 = self.down1(x)
    x, s2 = self.down2(x)
    x, s3 = self.down3(x)
    x, s4 = self.down4(x)

    x = self.bot(x)

    x = self.up4(x, s4)
    x = self.up3(x, s3)
    x = self.up2(x, s2)
    x = self.up1(x, s1)
    
    rgb = self.out_conv(x)
    rgb = torch.sigmoid(rgb)
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
EPOCH_NUM = 2000
video_interval = 50
H, W = 512, 512
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
gt_image = gt_image.resize((H, W)) # fit to kernel_size
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
B, FEATURE_DIM = 1, 16
Fscr = nn.Parameter(torch.randn(B, FEATURE_DIM, H, W, device=device), requires_grad=True)



C_in = 32
n_grid = 64  # n x n grid
encoding_dim_x = 10 # should be 1 at least
encoding_dim_y = 10 # should be 1 at least
pos_dim = encoding_dim_x + encoding_dim_y

Fscr = nn.Parameter(torch.randn(n_grid * n_grid, C_in, device=device), requires_grad=True)

xs = torch.arange(W, device=device)
ys = torch.arange(H, device=device)
global_x, global_y = torch.meshgrid(xs, ys, indexing='ij')  # (W, H)

# 各ピクセルのgridインデックス
x_grid_idx = torch.div(global_x, W // n_grid, rounding_mode='floor').clamp(max=n_grid-1)
y_grid_idx = torch.div(global_y, H // n_grid, rounding_mode='floor').clamp(max=n_grid-1)

# 各grid内でのlocal座標 (0 ~ grid_size-1)
local_x = global_x - x_grid_idx * (W // n_grid)
local_y = global_y - y_grid_idx * (H // n_grid)

# local座標を-1〜1に正規化
grid_size_x = W // n_grid
grid_size_y = H // n_grid
local_x_norm = 2 * (local_x.float() / (grid_size_x - 1)) - 1
local_y_norm = 2 * (local_y.float() / (grid_size_y - 1)) - 1

input_stack = [local_x_norm]
for i in range(encoding_dim_x - 1):
    input_stack.append(torch.sin(2 ** i * np.pi * local_x_norm))
input_stack.append(local_y_norm)
for i in range(encoding_dim_y - 1):
    input_stack.append(torch.sin(2 ** i * np.pi * local_y_norm))

coords = torch.stack(input_stack, dim=-1)
coords = coords.reshape(-1, encoding_dim_x + encoding_dim_y)
coords = coords.view(H, W, -1).permute(2, 0, 1)

# gridインデックスの計算もlocal_x, local_yの後に修正
grid_idx_map = x_grid_idx * n_grid + y_grid_idx
grid_idx_flat = grid_idx_map.reshape(-1)  # (H*W,)

netB = DNRUNet(c_in=C_in + pos_dim, base_ch=64, sh_mode="coeff", sh_out=32, sh_reduce="sum").to(device)
optimizer = optim.Adam([*netB.parameters(), Fscr], lr=1e-3)

progress_bar = tqdm(range(1, EPOCH_NUM + 1))
progress_bar.set_description("[train]")
loss_records = []

for epoch in progress_bar:
    optimizer.zero_grad()
    # 各ピクセルのgridに対応するfeature vectorを選択
    Fscr_repeated = Fscr[grid_idx_flat]  # (H*W, C_in)
    Fscr_repeated = Fscr_repeated.view(H, W, C_in).permute(2, 0, 1)
    model_input = torch.cat([coords, Fscr_repeated], dim=0).unsqueeze(0)  # (H*W, C_in+encoding_dim_x+encoding_dim_y)
    pred = netB(model_input)
    loss = F.mse_loss(pred, target_tensor, reduction='sum')
    loss.backward()
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

