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

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False

torch.cuda.reset_peak_memory_stats() # For memory stats

"""
neural texture & tiny MLP renderer
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


class TCNNSimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=64):
        super().__init__()
        if not TCNN_AVAILABLE:
            raise ImportError("tcnn is not installed. Please install tiny-cuda-nn.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=input_dim,
            n_output_dims=output_dim,
            encoding_config={"otype": "Identity"},
            network_config={
                "otype": "FullyFusedMLP",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 2,
                "activation": "ReLU",
                "output_activation": "Sigmoid"
            }
        )

    def forward(self, x):
        # x: (..., input_dim)
        orig_shape = x.shape
        x = x.reshape(-1, self.input_dim).contiguous()
        x_out = self.net(x)
        x_out = x_out.reshape(*orig_shape[:-1], self.output_dim).float()
        return x_out


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
flat_target_tensor = target_tensor.reshape(-1, CHANEL)

"""
Preparing model input
"""
C_in = 16
Fscr = nn.Parameter(torch.randn(H, W, C_in, device=device).reshape(-1, C_in), requires_grad=True)


# model = PixelMLP(input_dim=C_in).to(device)
model = TCNNSimpleMLP(input_dim=C_in).to(device)
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
    loss = F.mse_loss(pred, flat_target_tensor, reduction='mean') # sumだと勾配爆発してnanになる。
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
            rendered_image = pred.reshape(H, W, CHANEL)
            output_img_array = rendered_image.cpu().detach().numpy()
            plt.imsave(os.path.join(in_training_imgs_dir, f'rendered_output_image_{epoch}.png'), output_img_array)

print('Training complete')

# Showing training metrics
print(f"Forward average time: {np.mean(forward_time_list):.6f} Sec")
print(f"Backward average time: {np.mean(backward_time_list):.6f} Sec")
# max_mem = torch.cuda.max_memory_allocated() # This can not measure TCNN GPU Mem. 
# print(f"Max GPU memory: {max_mem / 1024 ** 2:.2f} MB")

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

