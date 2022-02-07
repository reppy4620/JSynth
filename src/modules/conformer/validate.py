import os
from pathlib import Path

import matplotlib.pyplot as plt
import pyworld as pw
import soundfile as sf
import torch
from tqdm import tqdm

from .module import ConformerModule


def validate(args, config):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    img_dir = output_dir / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConformerModule.load_from_checkpoint(args.ckpt_path, params=config)
    model = model.eval().to(device)

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

    def _get_sp(filename, fft_size=1024, frame_period=256 / 24000 * 1000):
        x, fs = sf.read(filename)
        _f0, t = pw.dio(x, fs, frame_period=frame_period)
        f0 = pw.stonemask(x, _f0, t, fs)
        sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
        return sp

    def get_sp(filename):
        sp = _get_sp(filename)
        sp = torch.FloatTensor(sp)
        sp = torch.log(sp)
        sp = sp.transpose(-1, -2)
        sp = sp[:-1, :]
        return sp

    wav_path_list = list(sorted(Path(model.params.data.data_dir).glob('*.wav')))[:model.params.data.valid_size * 2]

    for i, gt_path in tqdm(enumerate(wav_path_list), total=len(wav_path_list)):
        fn = gt_path.name
        gt = get_sp(gt_path)
        with torch.no_grad():
            gen = model(gt.unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()

        save_fig(gen, gt, img_dir / f'sp_{os.path.splitext(fn)[0]}.png')
