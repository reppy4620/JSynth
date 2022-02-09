JSynth
===

JSynth is Japanese Audio Synthesis Framework powered by PyTorchLightning.

This is made for accelerating my development on TTS.

# Requirements

- PyTorch
- PyTorchLightning
- torchaudio
- OmegaConf
- librosa
- soundfile
- pyworld
- matplotlib

# Models

- Conformer(FastSpeech2)
- Glow-TTS
- Grad-TTS

# Usage

## 0. Choose Model
if you chose model and then

```bash
$ cd scripts/MODEL_DIR
```

## 1. Preprocess

```bash
$ sh preprocess.sh configs/<CONFIG>.yaml
```

## 2. Train

```bash
$ sh train.sh configs/<CONFIG>.yaml
```
