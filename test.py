import utils
from utils import imresize
from model import inference
from model import grayscale_model
import time
import numpy as np
import torch
from metrics import mPSNR, mSAM, mSSIM
from degradation import ImgDownscale
import matplotlib.pyplot as plt
from torch import nn
import scipy.io as sio
import tqdm
import os


##################################
# configs
##################################
# device
device = 'cuda'
device_ids = [0, 1]
##################################
# display
show_img = True
pseudocolor_bands = [26, 13, 5]
##################################
# model
scale = 2
pretrained_model_fn = './pretrained/IPT_sr2.pt'
finetuned_trainable_fn = ''
hidden_channels = 64
need_lora = True
lora_r = 4
##################################
# data
src_lr_hsi_folder = ''
ref_hr_hsi_folder = ''
dst_sr_folder = ''
lr_mat_key = 'data'
hr_mat_key = 'data'
sr_mat_key = 'data'
##################################
# inference params
iters = 5
beta = 0.8
batch_size = 2048
eigen_dims = 0.5  # 0 < R < 1: Ratio to the input channels. (float)| R >= 1: No. channels (int)| None: use all channels
step_size = (32, 32)
use_geometric_transform = True
##################################


def load_hsi(hsi_fn, key):
    mat = sio.loadmat(hsi_fn)
    hsi = mat[key]
    hsi = np.array(hsi).astype(np.float32)
    return hsi


def bicubic_numpy(x, **kwargs):
    return imresize.imresize(
        torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device),
        **kwargs
    ).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()


if __name__ == '__main__':

    model = grayscale_model.GrayscaleSR(
        scale=scale,
        hidden_channels=hidden_channels,
        need_lora=need_lora,
        lora_r=lora_r,
    )\
        .load_frozen_body(pretrained_model_fn)\
        .load_head_tail(finetuned_trainable_fn)\
        .load_lora(finetuned_trainable_fn)\
        .to(device)

    model_parallel = False
    if device.lower().find('cuda') >= 0:
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
            model_parallel = True
        else:
            device = f'cuda:{device_ids[0]}'

    lr_hsi_fns = sorted(os.listdir(src_lr_hsi_folder))
    hr_hsi_fns = sorted(os.listdir(ref_hr_hsi_folder)) if ref_hr_hsi_folder else None
    if hr_hsi_fns is not None:
        # check if lr and hr data are matched
        if len(lr_hsi_fns) == len(hr_hsi_fns):
            for i in range(len(lr_hsi_fns)):
                if lr_hsi_fns[i] != hr_hsi_fns[i]:
                    raise FileNotFoundError('LR HSIs must match HR reference HSIs.')
        else:
            raise FileNotFoundError('LR HSIs must match HR reference HSIs.')

    print('Start inference...')
    max_len = max([len(_) for _ in lr_hsi_fns])
    print((max_len - len('HSI')) * ' ' + 'HSI', end=': ')
    print('Time (s)', end=', ')
    if ref_hr_hsi_folder:
        print(' PSNR,', '  SSIM,', '  SAM,', end='')
    print()

    metrics = {'time': [], 'psnr': [], 'ssim': [], 'sam': []}
    for i in range(len(lr_hsi_fns)):
        lr_hsi_fn = lr_hsi_fns[i]
        print((max_len - len(lr_hsi_fn)) * ' ' + lr_hsi_fn, end=': ')
        lr_hsi = load_hsi(os.path.join(src_lr_hsi_folder, lr_hsi_fn), key=lr_mat_key)
        lH, lW, lC = lr_hsi.shape

        start = time.time()
        sr_hsi = inference.iterative_sr(
            # iter params
            deg_hsi_batch=torch.from_numpy(lr_hsi).permute(2, 0, 1).unsqueeze(0).to(device),
            iters=iters,
            beta=beta,
            # hsi_inference params
            model=model,
            scale=scale,
            batch_size=batch_size,
            patch_size=(48, 48),
            step_size=step_size,
            eigen_dims=eigen_dims,
            transform=use_geometric_transform
        ).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        end = time.time()
        
        time_s = end - start
        metrics['time'].append(time_s)
        time_str = f'{end - start:.2f}'
        print(f"{max(0, len('Time (s)') - len(time_str)) * ' '}{time_str}", end=', ')

        if ref_hr_hsi_folder:
            hr_hsi_fn = hr_hsi_fns[i]
            hr_hsi = load_hsi(os.path.join(ref_hr_hsi_folder, hr_hsi_fn), key=hr_mat_key)
            hH, hW, hC = hr_hsi.shape
            hr_hsi = bicubic_numpy(hr_hsi, sizes=(lH * scale, lW * scale))

            hr_minmax_obj = utils.MinMax(hr_hsi)
            hr_minmax = hr_minmax_obj.transform(hr_hsi)
            sr_minmax = hr_minmax_obj.transform(sr_hsi)

            psnr = mPSNR(x=sr_minmax, ref_x=hr_minmax)
            ssim = mSSIM(x=sr_minmax, ref_x=hr_minmax)
            sam = mSAM(x=sr_minmax, ref_x=hr_minmax)
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['sam'].append(sam)
            print(f'{psnr:.2f}', end=', ')
            print(f'{ssim:.4f}', end=', ')
            print(f'{sam:.3f}', end=', ')
        print()

        if dst_sr_folder:
            if not os.path.exists(dst_sr_folder):
                os.makedirs(dst_sr_folder)
            sio.savemat(os.path.join(dst_sr_folder, lr_hsi_fn), {sr_mat_key: sr_hsi})

        if show_img:
            fig = plt.figure()
            if ref_hr_hsi_folder:
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)
            else:
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)

            bicubic_hsi = bicubic_numpy(lr_hsi, scale=scale)

            ax1.imshow(utils.minmax(bicubic_hsi[..., pseudocolor_bands]))
            ax1.set_title(f'Bicubic x{scale}\n{lr_hsi_fn}')
            ax1.axis('off')

            ax2.imshow(utils.minmax(sr_hsi[..., pseudocolor_bands]))
            ax2.set_title(f'EigenSR x{scale}\n{lr_hsi_fn}')
            ax2.axis('off')

            if ref_hr_hsi_folder:
                ax3.imshow(utils.minmax(hr_hsi[..., pseudocolor_bands]))
                ax3.set_title(f'HR reference\n{lr_hsi_fn}')
                ax3.axis('off')

            plt.tight_layout()
            plt.show()

    time_str = f"{np.mean(metrics['time']):.2f}"
    print((max_len - len('Avg')) * ' ' + 'Avg', end=': ')
    print(f"{max(0, len('Time (s)') - len(time_str)) * ' '}{time_str}", end=', ')
    print(f"{np.nanmean(metrics['psnr']):.2f}", end=', ')
    print(f"{np.nanmean(metrics['ssim']):.4f}", end=', ')
    print(f"{np.nanmean(metrics['sam']):.3f}", end=', ')
    print()


