import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """MODIFIED: Bottleneck block for 1D sequence data."""

    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.block = nn.Sequential(
            # MODIFIED: Conv2d -> Conv1d
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            # MODIFIED: BatchNorm2d -> BatchNorm1d
            nn.BatchNorm1d(in_channels // reduction),
            nn.ReLU(inplace=True),
            # MODIFIED: Conv2d -> Conv1d
            nn.Conv1d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            # MODIFIED: BatchNorm2d -> BatchNorm1d
            nn.BatchNorm1d(in_channels // reduction),
            nn.ReLU(inplace=True),
            # MODIFIED: Conv2d -> Conv1d
            nn.Conv1d(in_channels // reduction, out_channels, 1),
            # MODIFIED: BatchNorm2d -> BatchNorm1d
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x


class NoiseAdapter(nn.Module):
    """MODIFIED: NoiseAdapter for 1D sequence data."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            self.feat = nn.Sequential(
                # MODIFIED: Uses the 1D-adapted Bottleneck
                Bottleneck(channels, channels, reduction=8),
                # MODIFIED: AdaptiveAvgPool2d -> AdaptiveAvgPool1d
                nn.AdaptiveAvgPool1d(1)
            )
        else:
            self.feat = nn.Sequential(
                # MODIFIED: Conv2d -> Conv1d
                nn.Conv1d(channels, channels * 2, 1),
                # MODIFIED: BatchNorm2d -> BatchNorm1d
                nn.BatchNorm1d(channels * 2),
                nn.ReLU(inplace=True),
                # MODIFIED: Conv2d -> Conv1d
                nn.Conv1d(channels * 2, channels, 1),
                # MODIFIED: BatchNorm2d -> BatchNorm1d
                nn.BatchNorm1d(channels),
            )
        self.pred = nn.Linear(channels, 2)

    def forward(self, x):
        # Flatten works correctly on (B, C, 1) output from pooling
        x = self.feat(x).flatten(1)
        x = self.pred(x).softmax(1)[:, 0]
        return x


class DiffusionModel(nn.Module):
    """MODIFIED: DiffusionModel for 1D sequence data."""

    def __init__(self, channels_in, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.time_embedding = nn.Embedding(1280, channels_in)

        if kernel_size == 3:
            self.pred = nn.Sequential(
                # MODIFIED: Uses the 1D-adapted Bottleneck
                Bottleneck(channels_in, channels_in),
                Bottleneck(channels_in, channels_in),
                # MODIFIED: Conv2d -> Conv1d
                nn.Conv1d(channels_in, channels_in, 1),
                # MODIFIED: BatchNorm2d -> BatchNorm1d
                nn.BatchNorm1d(channels_in)
            )
        else:
            self.pred = nn.Sequential(
                nn.Conv1d(channels_in, channels_in * 4, 1),
                nn.BatchNorm1d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels_in * 4, channels_in, 1),
                nn.BatchNorm1d(channels_in),
                nn.Conv1d(channels_in, channels_in * 4, 1),
                nn.BatchNorm1d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels_in * 4, channels_in, 1)
            )

    def forward(self, noisy_sequence, t):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_sequence
        # MODIFIED: Adapt time embedding for 3D tensors (B, C, L)
        # Original: [..., None, None] -> (B, C, 1, 1)
        # New: .unsqueeze(-1) -> (B, C, 1) for broadcasting over length L
        time_emb = self.time_embedding(t).unsqueeze(-1)
        feat = feat + time_emb
        ret = self.pred(feat)
        return ret


class AutoEncoder(nn.Module):
    """MODIFIED: AutoEncoder for 1D sequence data."""

    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            # MODIFIED: Conv2d -> Conv1d
            nn.Conv1d(channels, latent_channels, 1, padding=0),
            # MODIFIED: BatchNorm2d -> BatchNorm1d
            nn.BatchNorm1d(latent_channels)
        )
        self.decoder = nn.Sequential(
            # MODIFIED: Conv2d -> Conv1d
            nn.Conv1d(latent_channels, channels, 1, padding=0),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out

    def forward_encoder(self, x):
        return self.encoder(x)


class DDIMPipeline(nn.Module):
    """UNCHANGED: This class is dimension-agnostic and does not need modification."""

    def __init__(self, model, scheduler, noise_adapter=None, solver='ddim'):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            feat,
            generator=None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            proj=None
    ):
        # The logic here (e.g., creating noise shape, looping) is generic
        # and works as long as the sub-modules (model, scheduler, noise_adapter)
        # handle the tensor shapes correctly.

        # We pass the shape without the batch dimension
        tensor_shape = shape

        if self.noise_adapter is not None:
            noise = torch.randn((batch_size, *tensor_shape), device=device, dtype=dtype)
            # The adapted noise_adapter will correctly process the 3D feat
            timesteps = self.noise_adapter(feat)
            # The scheduler's add_noise is dimension-agnostic
            image = self.scheduler.add_noise_diff2(feat, noise, timesteps)
        else:
            image = feat

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # The adapted model will correctly process the 3D image/feat
            noise_pred = self.model(image, t.to(device))

            # The scheduler's step function is dimension-agnostic
            image = self.scheduler.step(
                noise_pred, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample']

        return image