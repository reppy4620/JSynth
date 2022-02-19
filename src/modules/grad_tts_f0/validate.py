from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchaudio
from tqdm import tqdm
from torchaudio.sox_effects import apply_effects_tensor

from .pl import GradTTSWithF0Module
from ..common.tokenizers import Tokenizer
from ..vocoders.hifi_gan import load_hifi_gan

SR = 24000


def validate(args, config):
    output_dir = Path(config.output_dir) / 'validate'
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GradTTSWithF0Module.load_from_checkpoint(args.ckpt_path, params=config)
    vocoder = load_hifi_gan(args.vocoder_path)
    model = model.eval().to(device)
    vocoder = vocoder.eval().to(device)

    data_dir = Path(config.data.data_dir)
    data_list = list(sorted(data_dir.glob('*.pt')))[:config.data.valid_size]

    tokenizer = Tokenizer.from_config(config.data.tokenizer)

    def save_fig(gen, gt, path):
        plt.figure(figsize=(14, 7))
        plt.subplot(211)
        plt.gca().title.set_text('GEN')
        plt.imshow(gen, aspect='auto', origin='lower')
        plt.subplot(212)
        plt.gca().title.set_text('GT')
        plt.imshow(gt, aspect='auto', origin='lower')
        plt.savefig(path)
        plt.close()

    def save_wav(wav, path):
        effects = [
            ['gain', '-n']
        ]
        wav, _ = apply_effects_tensor(wav, SR, effects, channels_first=True)
        torchaudio.save(
            str(path),
            wav,
            SR
        )

    for i, p in tqdm(enumerate(data_list), total=len(data_list)):
        d = output_dir / f'res_{i+1:04d}'
        d.mkdir(exist_ok=True)
        (
            wav,
            mel,
            inputs,
            *_
        ) = torch.load(p)
        *inputs, _ = tokenizer(inputs)
        length = torch.LongTensor([len(inputs[0])])

        inputs = [x.unsqueeze(0).to(device) for x in inputs]
        length = length.to(device)

        with torch.no_grad():
            o = model([*inputs, length])
            w = vocoder(o)
            o = o.squeeze(0).detach().cpu()
            w = w.squeeze(0).detach().cpu()

        save_wav(wav, d / f'gt.wav')
        save_wav(w, d / f'gen.wav')
        save_fig(o, mel, d / f'mel.png')
