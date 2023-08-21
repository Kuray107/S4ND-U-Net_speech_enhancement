import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DSSM_modules.s4nd import (
    S4ND,
    DownPool2D,
    UpPool2D,
    FFBlock2D,
    ResidualBlock2D,
)
from model.DSSM_modules.components import TransposedLN
from model.base_SE_model import Base_TF_Model

from util.dsp import convert_to_different_features


class backbone(nn.Module):
    def __init__(
        self,
        d_model=32,
        d_state=16,
        n_blocks=2,
        n_layers_per_block=4,
        time_resample_factor=2,
        ff_bottleneck_expand_factor=2,
        bidirectional=True,
        unet=True,
        dropout=0.0,
    ):
        def s4nd_block(dim):
            layer = S4ND(
                d_model=dim,
                d_state=d_state,
                bidirectional=bidirectional,
                dropout=dropout,
                transposed=True,
                return_state=False,
            )
            return ResidualBlock2D(
                d_model=dim,
                layer=layer,
                dropout=dropout,
            )

        def ff_block(dim):
            layer = FFBlock2D(
                d_model=dim,
                ff_bottleneck_expand_factor=ff_bottleneck_expand_factor,
                dropout=dropout,
            )
            return ResidualBlock2D(
                d_model=dim,
                layer=layer,
                dropout=dropout,
            )

        super().__init__()
        self.d_model = current_d_model = d_model
        self.unet = unet

        down_blocks = []
        for block_idx in range(n_blocks):
            block = []
            if unet:
                for layer_idx in range(n_layers_per_block):
                    block.append(s4nd_block(current_d_model))
                    if ff_bottleneck_expand_factor > 0:
                        block.append(ff_block(current_d_model))

            block.append(DownPool2D(current_d_model, time_resample_factor))
            down_blocks.append(nn.ModuleList(block))
            current_d_model *= 2

        center_block = []
        for layer_idx in range(n_layers_per_block):
            center_block.append(s4nd_block(current_d_model))
            if ff_bottleneck_expand_factor > 0:
                center_block.append(ff_block(current_d_model))

        up_blocks = []
        for block_idx in range(n_blocks):
            block = []
            block.append(UpPool2D(current_d_model, time_resample_factor))
            current_d_model //= 2

            for layer_idx in range(n_layers_per_block):
                block.append(s4nd_block(current_d_model))
                if ff_bottleneck_expand_factor > 0:
                    block.append(ff_block(current_d_model))
            up_blocks.append(nn.ModuleList(block))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.center_block = nn.ModuleList(center_block)
        self.up_blocks = nn.ModuleList(up_blocks)

        assert current_d_model == d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input:  (batch, d_model, freq, length)
        output: (batch, d_model, freq, length)
        """

        outputs = []
        outputs.append(x)
        for block in self.down_blocks:
            for layer in block:
                x = layer(x)
                outputs.append(x)

        for layer in self.center_block:
            x = layer(x)
        x = x + outputs.pop()

        for block in self.up_blocks:
            if self.unet:
                for layer in block:
                    x = layer(x)
                    x = x + outputs.pop()
            else:
                for layer in block:
                    x = layer(x)
                    if isinstance(layer, UpPool2D):
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop()

        return x


class S4ND_U_Net(Base_TF_Model):
    def __init__(
        self,
        base_args,
        d_model=32,
        d_state=16,
        n_blocks=2,
        n_layers_per_block=4,
        time_resample_factor=2,
        ff_bottleneck_expand_factor=2,
        is_complex=False,
        masking=True,
    ):
        super().__init__(**base_args)
        self.is_complex = is_complex
        self.masking = masking

        in_channels = out_channels = 2 if self.is_complex else 1
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1), nn.ReLU()
        )

        self.backbone = backbone(
            d_model=d_model,
            d_state=d_state,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            time_resample_factor=time_resample_factor,
            ff_bottleneck_expand_factor=ff_bottleneck_expand_factor,
        )

        self.norm = TransposedLN(d_model)

        self.final_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(d_model, out_channels, kernel_size=1),
        )

    def _forward(self, noisy_wav):
        complex_noisy_spec = self.stft(noisy_wav)
        noisy_features = convert_to_different_features(complex_noisy_spec)
        if self.is_complex:
            x = torch.stack([noisy_features["real"], noisy_features["imag"]], dim=1)
        else:
            x = noisy_features["mag"].unsqueeze(1)

        x = self.init_conv(x)
        x = self.backbone(x)
        x = self.norm(x)
        outputs = self.final_conv(x)

        if self.masking:
            outputs = self.apply_masking(
                outputs, noisy_features, is_complex=self.is_complex
            )
        elif not self.is_complex:
            outputs = F.relu(outputs)

        return outputs, noisy_features

    def forward(self, noisy_wav, clean_wav):
        outputs, _ = self._forward(noisy_wav)
        return self._compute_loss(
            outputs=outputs, clean_wav=clean_wav, is_complex=self.is_complex
        )

    def inference(self, noisy_wav):
        outputs, noisy_features = self._forward(noisy_wav)

        if self.is_complex:
            complex_enhanced_spec = torch.complex(outputs[:, 0], outputs[:, 1])
        else:
            enhanced_real = outputs.squeeze(1) * torch.cos(noisy_features["phase"])
            enhanced_imag = outputs.squeeze(1) * torch.sin(noisy_features["phase"])
            complex_enhanced_spec = torch.complex(
                real=enhanced_real, imag=enhanced_imag
            )

        enhanced_wav = self.istft(complex_enhanced_spec, length=noisy_wav.size(1))
        return enhanced_wav


if __name__ == "__main__":
    model = S4ND_U_Net(
        stft_args={
            "n_fft": 510,
            "win_length": 400,
            "hop_length": 100,
            "transform_type": "exponent",
        }
    ).cuda()
    clean_wav = torch.randn([1, 48000], device="cuda")
    noisy_wav = torch.randn([1, 48000], device="cuda")
    loss = model(noisy_wav, clean_wav)
    enhanced_wav = model.inference(noisy_wav)
    print(loss)
