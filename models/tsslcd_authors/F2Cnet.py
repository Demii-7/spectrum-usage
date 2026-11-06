# Authors' code, github.com/Xlab2024/TSS-LCD commit b156d0a007, unmodified (model code identical in d2d4f29).
import torch.nn as nn


class FutureAutoencoder(nn.Module):
    def __init__(self, seq_len, D, latent_dim, hidden_dim=256, num_blocks=3):
        super().__init__()
        self.seq_len = seq_len
        self.D = D
        self.hidden_dim = hidden_dim
        self.input_dim = seq_len * D
        self.num_blocks = num_blocks

        # ---------------- Encoder ----------------
        encoder_layers = []
        T = seq_len
        self.intermediate_lengths = [T]

        in_channels = D
        for _ in range(num_blocks):
            T_out = (T + 2*1 - 4) // 2 + 1  # padding=1, kernel=4, stride=2
            self.intermediate_lengths.append(T_out)

            encoder_layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            in_channels = hidden_dim
            T = T_out

        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_len = T  # 最终 T
        self.to_latent = nn.Linear(hidden_dim * self.latent_len, latent_dim)

        # ---------------- Decoder ----------------
        self.from_latent = nn.Linear(latent_dim, hidden_dim * self.latent_len)
        decoder_layers = []

        for i in range(num_blocks):
            T_in = self.intermediate_lengths[-(i+1)]  # 当前层输入长度
            T_target = self.intermediate_lengths[-(i+2)]  # 上一层（解码后）应恢复到的长度

            output_padding = T_target - ((T_in - 1) * 2 - 2 * 1 + 4)
            decoder_layers.append(nn.ConvTranspose1d(
                in_channels, hidden_dim, kernel_size=4, stride=2, padding=1, output_padding=output_padding
            ))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            in_channels = hidden_dim

        decoder_layers.append(nn.Conv1d(hidden_dim, D, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # x: [B, T, D]
        B = x.size(0)
        x = x.permute(0, 2, 1)  # [B, D, T]
        enc = self.encoder(x)   # [B, hidden, T']
        flat = enc.view(B, -1)
        z = self.to_latent(flat)

        y = self.from_latent(z).view(B, -1, self.latent_len)
        recon = self.decoder(y)  # [B, D, T]
        recon = recon.permute(0, 2, 1)  # [B, T, D]
        return recon, z