import torch
from torch import nn
import torch.nn.functional as F

from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from .scheduling_ddim import DDIMScheduler


class DiffKD(nn.Module):
    """
    MODIFIED: DiffKD class adapted for 1D sequence data.
    The class name remains 'DiffKD' for compatibility.
    """

    def __init__(
            self,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps

        # 使用我们适配好的1D AutoEncoder (来自new_diffkd_modules.py)
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels

        # MODIFIED: Conv2d -> Conv1d，用于处理序列数据的通道变换
        self.trans = nn.Conv1d(student_channels, teacher_channels, 1)

        # 使用我们适配好的1D子模块 (来自new_diffkd_modules.py)
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False,
                                       beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)

        # MODIFIED: Conv2d -> Conv1d, BatchNorm2d -> BatchNorm1d
        self.proj = nn.Sequential(
            nn.Conv1d(teacher_channels, teacher_channels, 1),
            nn.BatchNorm1d(teacher_channels)
        )

    def forward(self, student_feat, teacher_feat):
        # 输入的 student_feat 和 teacher_feat 都应为 (B, C, L) 形状

        student_feat = self.trans(student_feat)

        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        # pipeline的调用逻辑保持不变，它能正确处理3D张量的shape
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],  # 对于(B, C, L)，这里能正确获取(C, L)
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
        )
        refined_feat = self.proj(refined_feat)

        # ddim_loss的计算逻辑也与维度无关，保持不变
        ddim_loss = self.ddim_loss(teacher_feat)
        return refined_feat, teacher_feat, ddim_loss, rec_loss

    def ddim_loss(self, gt_feat):
        noise = torch.randn(gt_feat.shape, device=gt_feat.device)
        bs = gt_feat.shape[0]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()

        noisy_features = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_features, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss
