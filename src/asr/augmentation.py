"""Аугментація аудіо даних для тренування"""

import numpy as np
import torch
import torchaudio


class AudioAugmentor:
  """Аугментації для покращення robustness моделі"""

  def __init__(self, sample_rate=16000):
      self.sample_rate = sample_rate

  def add_noise(self, waveform, snr_db=20.0):
      """Додати випадковий шум з заданим SNR"""
      noise = torch.randn_like(waveform)
      signal_power = waveform.pow(2).mean()
      noise_power = noise.pow(2).mean()
      snr = 10 ** (snr_db / 10)
      scale = (signal_power / (noise_power * snr)).sqrt()
      return waveform + scale * noise

  def change_speed(self, waveform, factor=1.0):
      """Змінити швидкість відтворення (0.9-1.1)"""
      effects = [["speed", str(factor)], ["rate", str(self.sample_rate)]]
      augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
          waveform, self.sample_rate, effects
      )
      return augmented

  def change_pitch(self, waveform, semitones=0):
      """Змінити висоту тону (±2 semitones)"""
      cents = semitones * 100
      effects = [["pitch", str(cents)], ["rate", str(self.sample_rate)]]
      augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
          waveform, self.sample_rate, effects
      )
      return augmented

  def spec_augment(self, mel_spec, freq_masks=2, time_masks=2,
                   freq_width=10, time_width=20):
      """SpecAugment — маскування частотних і часових смуг"""
      augmented = mel_spec.clone()

      # Частотні маски
      for _ in range(freq_masks):
          f = np.random.randint(0, freq_width)
          f0 = np.random.randint(0, max(1, augmented.shape[-2] - f))
          augmented[..., f0:f0 + f, :] = 0

      # Часові маски
      for _ in range(time_masks):
          t = np.random.randint(0, time_width)
          t0 = np.random.randint(0, max(1, augmented.shape[-1] - t))
          augmented[..., :, t0:t0 + t] = 0

      return augmented

  def augment(self, waveform, level="medium"):
      """Застосувати випадкову комбінацію аугментацій.

      level: "light", "medium", "heavy"
      """
      if isinstance(waveform, np.ndarray):
          waveform = torch.from_numpy(waveform)
      if waveform.dim() == 1:
          waveform = waveform.unsqueeze(0)

      if level == "light":
          snr_range = (25, 35)
      elif level == "medium":
          snr_range = (15, 30)
      else:
          snr_range = (10, 25)

      # Шум
      if np.random.random() < 0.5:
          snr = np.random.uniform(*snr_range)
          waveform = self.add_noise(waveform, snr_db=snr)

      # Speed perturbation (0.9x - 1.1x)
      if np.random.random() < 0.5:
          factor = np.random.uniform(0.9, 1.1)
          waveform = self.speed_perturb(waveform, factor)

      return waveform

  def speed_perturb(self, waveform, factor):
      """Змінити швидкість через інтерполяцію (працює без sox)"""
      orig_len = waveform.shape[-1]
      new_len = int(orig_len / factor)
      waveform = torch.nn.functional.interpolate(
          waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
      ).squeeze(0)
      return waveform