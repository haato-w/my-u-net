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
multi feature on each grid vector and position (positional encoding) & tiny MLP renderer
"""

class PixelMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims = [64] * 2):
        super().__init__()
        layers = []
        input_dim = input_dim
        output_dim = 3 # R,G,B

        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
            
        self.model = nn.Sequential(*layers)
    
    def forward(self, coords):
        return self.model(coords)


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
flat_target_tensor = target_tensor.reshape(-1, CHANEL)

"""
Preparing model input
"""
C_in = 32
n_grid = 64  # n x n grid
encoding_dim_x = 10 # should be 1 at least
encoding_dim_y = 10 # should be 1 at least

Fscr = nn.Parameter(torch.randn(n_grid * n_grid, C_in, device=device), requires_grad=True)

xs = torch.linspace(-1, 1, steps=W, device=device)
ys = torch.linspace(-1, 1, steps=H, device=device)
grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')

input_stack = [grid_x]
for i in range(encoding_dim_x - 1):
    input_stack.append(torch.sin(2 ** i * np.pi * grid_x))
input_stack.append(grid_y)
for i in range(encoding_dim_y - 1):
    input_stack.append(torch.sin(2 ** i * np.pi * grid_y))

coords = torch.stack(input_stack, dim=-1)
coords = coords.reshape(-1, encoding_dim_x + encoding_dim_y)

# Calculate grid index for each pixel
x_idx = torch.arange(W, device=device)
y_idx = torch.arange(H, device=device)
x_grid_idx = torch.div(x_idx * n_grid, W, rounding_mode='floor').clamp(max=n_grid-1)
y_grid_idx = torch.div(y_idx * n_grid, H, rounding_mode='floor').clamp(max=n_grid-1)
grid_idx_map = x_grid_idx[:, None] * n_grid + y_grid_idx[None, :]
grid_idx_flat = grid_idx_map.reshape(-1)  # (H*W,)

model = PixelMLP(input_dim=C_in + encoding_dim_x + encoding_dim_y).to(device)
optimizer = optim.Adam([*model.parameters(), Fscr], lr=1e-3)

progress_bar = tqdm(range(1, EPOCH_NUM + 1))
progress_bar.set_description("[train]")
loss_records = []

for epoch in progress_bar:
    optimizer.zero_grad()
    # 各ピクセルのgridに対応するfeature vectorを選択
    Fscr_repeated = Fscr[grid_idx_flat]  # (H*W, C_in)
    model_input = torch.cat([coords, Fscr_repeated], dim=1)  # (H*W, C_in+encoding_dim_x+encoding_dim_y)
    pred = model(model_input)
    loss = F.mse_loss(pred, flat_target_tensor, reduction='sum')
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
            rendered_image = pred.reshape(H, W, CHANEL)
            output_img_array = rendered_image.cpu().detach().numpy()
            plt.imsave(os.path.join(in_training_imgs_dir, f'rendered_output_image_{epoch}.png'), output_img_array)

print('Training complete')

"""
Postprocessing
"""
print('Postprocessing...')
# Visulizing trained image
rendered_image = pred.reshape(H, W, CHANEL)
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

