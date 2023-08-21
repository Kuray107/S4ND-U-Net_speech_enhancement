"""
A Basic Speech Enhancement Model abstract class. This class is intended to be successed 
and thus shouldn't be created directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.dsp import convert_to_different_features
from model.loss import get_TF_domain_loss_function


class Base_TF_Model(nn.Module):
    """
    The parent class for all TF-domain SE models.
    Member Variables:
        n_fft,
        win_length,
        hop_length,
        spec_factor,
        spec_abs_exponent,
        transform_type,
        loss_fn,

    Member Functions:
        stft
        istft
        apply_masking
        _compute_loss
        _resynthesis_from_phase
    """

    def __init__(
        self,
        n_fft=512,
        win_length=512,
        hop_length=256,
        spec_factor=0.15,
        spec_abs_exponent=0.5,
        transform_type="exponent",
        loss_fn_name="TF-MAE",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = torch.hann_window(self.win_length).cuda()
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.transform_type = transform_type

        self.loss_fn = get_TF_domain_loss_function(loss_fn_name)

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args.
            wav: torch.Tensor, shape = [B, T]
        Return:
            complex_spec: torch.Tensor, shape = [B, D, T]
        """
        complex_spec = torch.stft(
            input=wav,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            normalized=False,
            return_complex=True,
        )
        complex_spec = self.spec_forward(complex_spec)

        return complex_spec

    def istft(self, complex_spec, length=None):
        """
        Args.
            complex_spec: torch.Tensor, shape = [B, D, T]
            length: int
        Return:
            wav: torch.Tensor, shape = [B, T]
        """
        complex_spec = self.spec_backward(complex_spec)
        wav = torch.istft(
            input=complex_spec,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            length=length,
        )
        return wav

    def spec_forward(self, complex_spec: torch.Tensor):
        """
        Args.
            complex_spec: torch.Tensor, shape = [B, D, T]
        Return:
            complex_spec: torch.Tensor, shape = [B, D, T]
        """
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                complex_spec = (complex_spec.abs() + 1e-8) ** e * torch.exp(1j * complex_spec.angle())
            complex_spec = complex_spec * self.spec_factor

        elif self.transform_type == "log":
            complex_spec = torch.log(1 + complex_spec.abs()) * torch.exp(1j * complex_spec.angle())
            complex_spec = complex_spec * self.spec_factor
        elif self.transform_type == "none":
            complex_spec = complex_spec
        return complex_spec

    def spec_backward(self, complex_spec: torch.Tensor):
        """
        Args.
            complex_spec: torch.Tensor, shape = [B, D, T]
        Return:
            complex_spec: torch.Tensor, shape = [B, D, T]
        """
        if self.transform_type == "exponent":
            complex_spec = complex_spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                complex_spec = complex_spec.abs() ** (1 / e) * torch.exp(1j * complex_spec.angle())
        elif self.transform_type == "log":
            complex_spec = complex_spec / self.spec_factor
            complex_spec = (torch.exp(complex_spec.abs()) - 1) * torch.exp(1j * complex_spec.angle())
        elif self.transform_type == "none":
            complex_spec = complex_spec
        return complex_spec

    def apply_masking(self, outputs, noisy_features, is_complex=False):
        """
        Args.
            outputs: torch.Tensor [B, C, D, T]
            noisy_features: Dict
        Return.
            torch.Tensor [B, C, D, T]
        """
        if is_complex:
            assert outputs.size(1) == 2
            mask_real = outputs[:, 0]
            mask_imag = outputs[:, 1]
            mask_mags = torch.sqrt(mask_real**2 + mask_imag**2)
            mask_phase = torch.atan2(mask_imag, mask_real)

            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags * noisy_features["mag"]
            est_phase = noisy_features["phase"] + mask_phase
            est_real = est_mags * torch.cos(est_phase)
            est_imag = est_mags * torch.sin(est_phase)
            return torch.stack([est_real, est_imag], dim=1)
        else:
            assert outputs.size(1) == 1
            mask_mag = torch.sigmoid(outputs).squeeze(1)
            enhanced_mag = noisy_features["mag"] * mask_mag
            return enhanced_mag.unsqueeze(1)

    def _compute_loss(self, outputs, clean_wav, is_complex):
        complex_clean_spec = self.stft(clean_wav)
        clean_features = convert_to_different_features(complex_clean_spec)
        if is_complex:
            enhanced_mag_spec = torch.sqrt(
                outputs[:, 0] ** 2 + outputs[:, 1] ** 2 + 1e-8
            )
            phase = torch.angle(torch.complex(real=outputs[:, 0], imag=outputs[:, 1]))

            mag_loss = self.loss_fn(enhanced_mag_spec, clean_features["mag"])
            phase_loss = -1 * torch.mean(
                clean_features["mag"] * torch.cos(phase - clean_features["phase"])
            )
            all_loss = mag_loss + phase_loss

            return all_loss
        else:
            return self.loss_fn(outputs.squeeze(1), clean_features["mag"])

    def phase_evaluation(self, noisy_wav, clean_wav):
        """evaluate how well we could estimate the phase"""

        assert (
            self.is_complex
        )  # Only complex feature model needs to analyze the phase information
        outputs, _ = self._forward(noisy_wav)
        complex_enhanced_spec = torch.complex(outputs[:, 0], outputs[:, 1])
        enhanced_mag_spec = torch.abs(complex_enhanced_spec)

        enhanced_phase_wav = self.istft(complex_enhanced_spec, length=noisy_wav.size(1))

        iter_phase_wav = self._resynthesis_from_phase(
            enhanced_mag_spec, enhanced_phase_wav
        )
        noisy_phase_wav = self._resynthesis_from_phase(enhanced_mag_spec, noisy_wav)
        clean_phase_wav = self._resynthesis_from_phase(enhanced_mag_spec, clean_wav)

        return enhanced_phase_wav, iter_phase_wav, noisy_phase_wav, clean_phase_wav

    def _resynthesis_from_phase(self, mag_spec, org_wav):
        org_cspec = self.stft(org_wav)
        phase = torch.angle(org_cspec)
        est_cspec = torch.complex(
            real=mag_spec * torch.cos(phase), imag=mag_spec * torch.sin(phase)
        )
        return self.istft(est_cspec, length=org_wav.size(1))