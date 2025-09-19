# hash_image_fit.py
import argparse, math, os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

# ------------------------------
# 2D Multiresolution Hash Encoder
# ------------------------------
class HashEncoder2D(nn.Module):
    def __init__(self, L=16, F=2, T=2**18, Nmin=16, Nmax=2**17, smoothstep=False, device="cuda"):
        super().__init__()
        self.L = L
        self.F = F
        self.T = T
        self.Nmin = Nmin
        self.Nmax = Nmax
        self.smoothstep = smoothstep
        self.device = device

        # 幾何級数スケール
        self.b = math.exp((math.log(Nmax) - math.log(Nmin)) / (L - 1))
        # 各レベルの解像度
        Ns = [int(math.floor(Nmin * (self.b ** l))) for l in range(L)]
        self.register_buffer("Ns", torch.tensor(Ns, dtype=torch.int32, device=device))

        # 各レベルのテーブル（T x F）
        tables = []
        for _ in range(L):
            table = nn.Parameter(torch.empty(T, F).uniform_(-1e-4, 1e-4))
            tables.append(table)
        self.tables = nn.ParameterList(tables)

        # ハッシュ用定数（x ⊕ (y * prime)）mod T
        self.register_buffer("PI1", torch.tensor(1, dtype=torch.int64, device=device))
        self.register_buffer("PI2", torch.tensor(2654435761, dtype=torch.int64, device=device))

    @staticmethod
    def _smoothstep(w):
        # w in [0,1], apply per-dim smoothstep to weights
        return w * w * (3.0 - 2.0 * w)

    def forward(self, xy):  # xy: (B, 2) in [0,1]
        """
        Returns encoded features y: (B, L*F)
        """
        B = xy.shape[0]
        xy = torch.clamp(xy, 0.0, 1.0)  # 保険

        feats = []
        for l in range(self.L):
            Nl = int(self.Ns[l].item())
            table = self.tables[l]  # (T, F)

            # スケールしてボクセル座標へ
            p = xy * Nl  # (B,2)
            p0 = torch.floor(p).to(torch.int64)            # (B,2) lower corner
            p1 = torch.clamp(p0 + 1, max=Nl)               # 上限ガード（境界外れ防止）
            w = p - p0.to(p.dtype)                          # (B,2) fractional

            if self.smoothstep:
                w_eff = self._smoothstep(w)
            else:
                w_eff = w

            # 4隅
            x0, y0 = p0[:, 0], p0[:, 1]
            x1, y1 = p1[:, 0], p1[:, 1]

            # --- 1:1 マップかハッシュかの切替 ---
            # (Nl+1)^2 <= T なら 1:1（線形インデックス）、そうでなければハッシュ
            dense_ok = (Nl + 1) * (Nl + 1) <= self.T

            if dense_ok:
                # 線形インデックス: idx = x*(Nl+1) + y
                def dense_idx(xi, yi):
                    return (xi * (Nl + 1) + yi).to(torch.int64)
                idx00 = dense_idx(x0, y0)
                idx10 = dense_idx(x1, y0)
                idx01 = dense_idx(x0, y1)
                idx11 = dense_idx(x1, y1)
            else:
                # ハッシュ: h(x,y) = (x*PI1) XOR (y*PI2) mod T
                def hash_xy(xi, yi):
                    h = (xi.to(torch.int64) * self.PI1) ^ (yi.to(torch.int64) * self.PI2)
                    # mod T
                    return torch.remainder(h, self.T)
                idx00 = hash_xy(x0, y0)
                idx10 = hash_xy(x1, y0)
                idx01 = hash_xy(x0, y1)
                idx11 = hash_xy(x1, y1)

            # ルックアップ（B, F）
            f00 = table[idx00]
            f10 = table[idx10]
            f01 = table[idx01]
            f11 = table[idx11]

            # 二線形補間
            wx, wy = w_eff[:, 0:1], w_eff[:, 1:2]  # (B,1)
            f0 = f00 * (1 - wx) + f10 * wx
            f1 = f01 * (1 - wx) + f11 * wx
            f = f0 * (1 - wy) + f1 * wy  # (B,F)

            feats.append(f)

        y = torch.cat(feats, dim=1)  # (B, L*F)
        return y

# ------------------------------
# 小型 MLP
# ------------------------------
class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=3, depth=2):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), nn.ReLU(True)]
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------------
# 学習ループ
# ------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    # 画像読み込み & 正規化
    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    target = torch.from_numpy(np.asarray(img)).float() / 255.0  # (H,W,3)
    target = target.to(device)

    # ピクセル座標（中心サンプル）
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    # 中心補正しつつ [0,1] に正規化
    xy_full = torch.stack([(xx + 0.5) / W, (yy + 0.5) / H], dim=-1).view(-1, 2)  # (H*W,2)
    rgb_full = target.view(-1, 3)  # (H*W,3)

    # モデル
    enc = HashEncoder2D(L=args.L, F=args.F, T=args.T, Nmin=args.Nmin, Nmax=args.Nmax,
                        smoothstep=args.smoothstep, device=device).to(device)
    mlp = TinyMLP(in_dim=args.L * args.F, hidden=args.hidden, out_dim=3, depth=args.depth).to(device)

    params = list(enc.parameters()) + list(mlp.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-15, weight_decay=args.wd)

    N = xy_full.shape[0]
    batch = args.batch
    steps = args.steps

    enc.train(); mlp.train()
    for it in range(1, steps + 1):
        # ランダムサブサンプル
        idx = torch.randint(0, N, (batch,), device=device)
        xy = xy_full[idx]            # (B,2)
        target_rgb = rgb_full[idx]   # (B,3)

        y = enc(xy)                  # (B, L*F)
        pred = mlp(y)                # (B,3); リニア出力
        pred = torch.sigmoid(pred)   # sRGBレンジへ

        loss = F.mse_loss(pred, target_rgb, reduction="mean")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % args.log_every == 0:
            psnr = -10.0 * torch.log10(loss.detach())
            print(f"[{it:6d}/{steps}] loss={loss.item():.6f}  PSNR={psnr.item():.2f} dB")

        if it % args.save_every == 0 or it == steps:
            # フル解像度で推論して保存
            with torch.no_grad():
                BMAX = 262144  # 256k/chunk くらいに分割
                outs = []
                for s in range(0, N, BMAX):
                    yb = enc(xy_full[s:s+BMAX])
                    pb = torch.sigmoid(mlp(yb))
                    outs.append(pb)
                recon = torch.cat(outs, dim=0).view(H, W, 3).permute(2,0,1).contiguous()
                os.makedirs(args.outdir, exist_ok=True)
                save_path = os.path.join(args.outdir, f"recon_it{it}.png")
                save_image(recon.clamp(0,1), save_path)
                # 1回だけGTも保存
                if it == args.save_every:
                    save_image(target.permute(2,0,1), os.path.join(args.outdir, "gt.png"))

    print("done.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--F", type=int, default=2)
    ap.add_argument("--T", type=int, default=2**18)          # 2^18=262,144
    ap.add_argument("--Nmin", type=int, default=16)
    ap.add_argument("--Nmax", type=int, default=2**17)       # 131,072
    ap.add_argument("--smoothstep", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=65536)      # 大きめが速い（GPUメモリと相談）
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--outdir", type=str, default="out_hash_fit")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()
