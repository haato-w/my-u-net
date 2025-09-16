import torch
import torch.nn as nn
import torch.nn.functional as F

def sh9(viewdirs): #viewdirs: [B, 3, H, W], unit vectors
  x, y, z = viewdirs[:, 0:1], viewdirs[:, 1:2], viewdirs[:, 2:3]
  c0, c1, c2, c3, c4 = 0.282095, 0.488603, 1.092548, 0.315392, 0.546274
  Y = torch.cat([
    c0 + 0.0 * x, 
    c1 * y, c1 * z, c1 * x, 
    c2 * x * y, c2 * y * z, c3 * (3 * z * z - 1), 
    c2 * x * z, c4 * (x * x - y * y)
  ], dim=1)
  return Y

class SHGate(nn.Module):
  def __init__(self, in_ch, out_ch=None, sh_mode="broadcast", reduce="sum"):
    super().__init__()
    assert sh_mode in ["broadcast", "coeff"]
    self.sh_mode = sh_mode
    self.reduce = reduce
    if sh_mode == "coeff":
      assert out_ch is not None, "coeff mode requires out_sh"
      self.proj = nn.Conv2d(in_ch, 9 * out_ch, kernel_size=1, bias=True)
    
  def forward(self, Fscr, viewdirs):
    Y = sh9(viewdirs)
    if self.sh_mode == "broadcast":
      B, C, H, W = Fscr.shape
      F9 = Fscr.unsqueeze(1) * Y.unsqueeze(2)
      F9 = F9.reshape(B, 9 * C, H, W)
      return F9
    else:
      B, _, H, W = Fscr.shape
      coeff = self.proj(Fscr)
      coeff = coeff.view(B, 9, -1, H, W)
      gated = coeff * Y.unsqueeze(2)
      if self.reduce == "sum":
        return gated.sum(dim=1)
      elif self.reduce == "concat":
        return gated.reshape(B, 9 * coeff.shape[2], H, W)
      else:
        raise ValueError("reduce must be 'sum' or 'concat'")

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
    self.block = nn.Sequetial(
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
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
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
    
    if sh_mode == "broadcast":
      enc_in = 9 * c_in + aux_ch
      self.sh_gate = SHGate(c_in, sh_mode="broadcast")
    else:
      assert sh_out is not None
      gated_out = (sh_out if sh_reduce == "sum" else 9 * sh_out)
      enc_in = gated_out + aux_ch
      self.sh_gate = SHGate(c_in, out_ch=sh_out, sh_mode="coeff", reduce=sh_reduce)
    
    # Encoder (5 steps)
    self.in_conv = conv_block(enc_in, base_ch)
    self.down1 = Down(base_ch, base_ch * 2)
    self.down2 = Down(base_ch * 2, base_ch * 4)
    self.down3 = Down(base_ch * 4, base_ch * 8)
    self.down4 = Down(base_ch * 8, base_ch * 8)
    
    # Bottleneck
    self.bot = nn.Sequential(
      conv_block(base_ch * 8, base_ch * 8), 
      conv_block(base_ch * 8, base_ch * 8), 
    )

    # Decoder
    self.up4 = Up(base_ch * 8, base_ch * 8)
    self.up3 = Up(base_ch * 8, base_ch * 4)
    self.up2 = Up(base_ch * 4, base_ch * 2)
    self.up1 = Up(base_ch * 2, base_ch)

    self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)
  
  def forward(self, Fscr, vdirs, aux=None):
    gated = self.sh_gate(Fscr, vdirs)
    x = gated if aux is None else torch.cat([gated, aux], dim=1)

    # U-Net
    x = self.in_conv(x)
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

B, C_in, H, W = 2, 16, 540, 960
Fscr = torch.randn(B, C_in, H, W)
vdirs = F.normalize(torch.randn(B, 3, H, W), dim=1)

netA = DNRUNet(c_in=C_in, base_ch=64, sh_mode="broadcast", aux_ch=0)
rgbA = netA(Fscr, vdirs)

netB = DNRUNet(c_in=C_in, base_ch=64, sh_mode="coeff", sh_out=32, sh_reduce="sum")
rgbB = netB(Fscr, vdirs)

pred = netA(Fscr, vdirs)
loss_img = (pred, gt_image).abs().mean()
