# Authors' code, github.com/Xlab2024/TSS-LCD commit b156d0a007, unmodified (model code identical in d2d4f29).
import numpy as np
import torch
import torch.nn as nn
# def make_beta_schedule(n_timestep, beta_start=1e-4, beta_end=1e-2):
#     return np.linspace(beta_start, beta_end, n_timestep)

def cosine_beta_schedule(T, s=0.0018):
    steps = T + 1
    x = np.linspace(0, T, steps)
    alpha_bar = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(beta, 1e-8, 0.999)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B]
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * (-np.log(10000.0) / half_dim))
        emb = t[:, None].float() * emb[None, :]  # [B, half_dim]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim]

class EnhancedNoiseNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        input_dim = latent_dim * 2 + 32  # 与原接口一致

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(64))
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.enc2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128))
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU())

        # Decoder
        self.up1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128))

        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(64))

        # Final projection to latent_dim
        self.final = nn.Linear(64 * (input_dim // 1), latent_dim)  # Conv1d doesn't change length if no pooling

    def forward(self, x):  # [B, input_dim]
        B = x.size(0)
        x = x.unsqueeze(1)  # [B, 1, T]

        e1 = self.enc1(x)       # [B, 64, T]
        p1 = self.pool1(e1)     # [B, 64, T//2]
        e2 = self.enc2(p1)      # [B, 128, T//2]
        p2 = self.pool2(e2)     # [B, 128, T//4]

        mid = self.bottleneck(p2)  # [B, 256, T//4]

        u1 = self.up1(mid)  # [B, 128, T//2]
        d1 = self.dec1(torch.cat([u1, e2], dim=1))  # skip connection

        u2 = self.up2(d1)   # [B, 64, T]
        d2 = self.dec2(torch.cat([u2, e1], dim=1))  # skip connection

        out = d2.flatten(start_dim=1)  # [B, 64*T]
        return self.final(out)         # [B, latent_dim]

class DiffusionModel(nn.Module):
    def __init__(self, cond_dim, latent_dim, n_timestep, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_timestep = n_timestep
        self.device = device
        self.cond_proj = nn.Linear(cond_dim, latent_dim)
        self.time_embedding = SinusoidalTimeEmbedding(dim=32)
        self.noise_net = EnhancedNoiseNet(latent_dim)

        betas = cosine_beta_schedule(n_timestep)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)  # (T,)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)  # α_bar_t
        self.one_minus_acp = 1 - self.alpha_cumprod
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]])

    def q_sample(self, z0, t, noise):
        a_bar = self.alpha_cumprod[t].view(-1, 1)
        om = 1 - a_bar
        return torch.sqrt(a_bar) * z0 + torch.sqrt(om) * noise

    def forward(self, zt, cond_z, t):
        t_emb = self.time_embedding(t)  # [B, 32]
        inp = torch.cat([zt, self.cond_proj(cond_z), t_emb], dim=1)
        return self.noise_net(inp)

    def p_sample(self, zt, cond_z, t):
        beta_t = self.betas[t].view(-1, 1)
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alpha_cumprod[t].view(-1, 1)
        alpha_bar_tm1 = self.alpha_cumprod_prev[t].view(-1, 1)

        e_pred = self.forward(zt, cond_z, t)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_alpha_bar_tm1 = torch.sqrt(alpha_bar_tm1)
        denom = 1 - alpha_bar_t

        A_t = (1 / sqrt_alpha_t) * (sqrt_alpha_bar_tm1 * beta_t / denom) + (sqrt_alpha_t * (1 - alpha_bar_tm1) / denom)
        B_t = (torch.sqrt(1 - alpha_bar_t) / sqrt_alpha_t) * (sqrt_alpha_bar_tm1 * beta_t / denom)
        sigma_t = torch.sqrt((1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t)

        noise = torch.randn_like(zt)
        mask = (t > 0).float().view(-1, 1)
        return A_t * zt - B_t * e_pred + mask * sigma_t * noise

    # def p_sample(self, zt, cond_z, t):
    #     beta_t = self.betas[t].view(-1,1)
    #     a_bar = self.alpha_cumprod[t].view(-1,1)
    #     om = self.one_minus_acp[t].view(-1,1)
    #     e_pred = self.forward(zt, cond_z, t)
    #     z0_pred = (zt - torch.sqrt(om)*e_pred)/torch.sqrt(a_bar)
    #     mean = torch.sqrt(a_bar)*z0_pred + torch.sqrt(1-a_bar-beta_t)*e_pred
    #     noise = torch.randn_like(zt)
    #     mask = (t>0).float().view(-1,1)
    #     return mean + mask*torch.sqrt(beta_t)*noise

    def p_sample_loop(self, cond_z):
        B = cond_z.size(0)
        z = torch.randn(B, self.latent_dim, device=self.device)
        for step in reversed(range(self.n_timestep)):
            t = torch.full((B,), step, device=self.device, dtype=torch.long)
            z = self.p_sample(z, cond_z, t)
        return z

# MLP-Future2Cond
# def inference(ae_ctx, ae_fut, diffusion, ctx, device):
#     ae_ctx.eval()
#     ae_fut.eval()
#     diffusion.eval()
#     with torch.no_grad():
#         B = ctx.size(0)
#         _, z_ctx = ae_ctx(ctx.to(device))
#         z_sample = diffusion.p_sample_loop(z_ctx)
#         # y = ae_fut.decoder_input(z_sample)
#         # recon_flat = ae_fut.decoder_blocks(y)
#         # y = ae_fut.from_latent(z_sample)
#         # y = y.view(B, -1, CFG['future_length'], CFG['location'], CFG['frequency'])
#         # # y = y.view(B, -1, self.seq_len, self.L, self.F)
#         # recon = ae_fut.decoder3d(y)
#         # recon = recon.squeeze(1).view(B, CFG['future_length'], -1)
#         y = ae_fut.decoder_input(z_sample)
#         y = ae_fut.decoder_blocks(y)
#         recon_flat = ae_fut.to_output(y)
#         # recon = recon_flat.view(B, CFG['future_length'], CFG['D'])
#
#
#         # y = ae_fut.decoder_input(z_sample)
#         # y = ae_fut.decoder_blocks(y)
#         # recon_flat = ae_fut.to_output(y)
#         # # recon_flat = ae_fut.decoder(z_sample)
#         out = recon_flat.view(-1, CFG['future_length'], CFG['D'])
#     return out.cpu().numpy()

# conv1d-F2Cconv1d
def inference(ae_ctx, ae_fut, diffusion, ctx, device):
    ae_ctx.eval()
    ae_fut.eval()
    diffusion.eval()
    with torch.no_grad():
        B = ctx.size(0)
        _, z_ctx = ae_ctx(ctx.to(device))
        z_sample = diffusion.p_sample_loop(z_ctx)
        y = ae_fut.from_latent(z_sample).view(B, -1, ae_fut.latent_len)
        recon = ae_fut.decoder(y)  # [B, D, T]
        recon = recon.permute(0, 2, 1)  # [B, T, D]
    return recon.cpu().numpy()