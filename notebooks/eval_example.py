from typing import Dict, Any, Optional
import os
import gc
import sys
import copy
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
sys.path.append('../src')
sys.path.append('../src/pgm')
import multiprocessing
import numpy as np
from tqdm import tqdm

from pgm.train_cf import cf_epoch
from pgm.train_pgm import setup_dataloaders, preprocess
from pgm.flow_pgm import ChestPGM

class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)
predictor_path = '/workspace/causal-gen/checkpoints/f_a_s_r/aux_60k-aux/checkpoint.pt'
print(f'\nLoading predictor checkpoint: {predictor_path}')
predictor_checkpoint = torch.load(predictor_path)
predictor_args = Hparams()
predictor_args.update(predictor_checkpoint['hparams'])
assert predictor_args.dataset == 'mimic'
predictor = ChestPGM(predictor_args).cuda()
predictor.load_state_dict(predictor_checkpoint['ema_model_state_dict'])
pgm_path = '/workspace/causal-gen/checkpoints/f_a_s_r/pgm_60k-pgmg/checkpoint.pt'
print(f'\nLoading PGM checkpoint: {pgm_path}')
pgm_checkpoint = torch.load(pgm_path)
pgm_args = Hparams()
pgm_args.update(pgm_checkpoint['hparams'])
assert pgm_args.dataset == 'mimic'
pgm = ChestPGM(pgm_args).cuda()
pgm.load_state_dict(pgm_checkpoint['ema_model_state_dict'])

def load_vae(vae_path):
    print(f'\nLoading VAE checkpoint: {vae_path}')
    vae_checkpoint = torch.load(vae_path)
    vae_args = Hparams()
    vae_args.update(vae_checkpoint['hparams'])
    vae_args.data_dir = '/workspace/pneumonia_without_drug/mimic'

    # init model
    assert vae_args.hps == 'mimic192'
    if not hasattr(vae_args, 'vae'):
        vae_args.vae = 'simple'

    if vae_args.vae == 'hierarchical':
        from vae import HVAE
        vae = HVAE(vae_args).cuda()
    elif vae_args.vae == 'simple':
        from simple_vae import VAE
        vae = VAE(vae_args).cuda()
    else:
        NotImplementedError
    vae.load_state_dict(vae_checkpoint['ema_model_state_dict'])
    return vae, vae_args

vae_path = '/workspace/causal-gen/checkpoints/f_a_s_r/mimic192/checkpoint.pt'
vae, vae_args = load_vae(vae_path)
import matplotlib.pyplot as plt

def visualize_counterfactuals(original_images, cf_images, original_pa, cf_pa, num_samples=8):
    """
    Visualize original images vs their counterfactuals using TRUE / SCM-generated parent labels.

    Rows:
      1. Original image with disease (finding)
      2. Counterfactual image with intervened disease (finding)
      3. Direct Effect (Original - Counterfactual) heatmap (diverging colormap)
    """
    n = min(num_samples, len(original_images), len(cf_images))
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 6.0))

    # Ensure axes is 2D even if n == 1
    if n == 1:
        axes = axes.reshape(3, 1)

    # Precompute raw difference in float (keep signed range)
    # original_images and cf_images are in [0,255] after preprocessing
    diff = cf_images[:n].float() - original_images[:n].float()  # shape (n,1,H,W) or (n,H,W)
    
    # Use 75th percentile of absolute difference to set color scale (handles outliers)
    # This makes the visualization clearer by not letting extreme outliers dominate the scale
    abs_diff = diff.abs()
    percentile_75 = torch.quantile(abs_diff, 0.75).item()
    dmax = percentile_75 + 1e-6
    vmin, vmax = -dmax, dmax

    for i in range(n):
        # Row 0: Original
        orig_img = original_images[i].squeeze().detach().cpu().numpy()
        axes[0, i].imshow(orig_img, cmap='gray', vmin=0, vmax=255)
        # Extract disease label (finding)
        orig_finding_tensor = original_pa["finding"][i]
        # Handle both scalar and one-hot encoded disease labels
        if orig_finding_tensor.dim() > 0 and orig_finding_tensor.numel() > 1:
            orig_disease = orig_finding_tensor.argmax().item()
        else:
            orig_disease = int(orig_finding_tensor.item())
        axes[0, i].set_title(f'Original\nDisease: {orig_disease}', fontsize=9)
        axes[0, i].axis('off')

        # Row 1: Counterfactual
        cf_img = cf_images[i].squeeze().detach().cpu().numpy()
        axes[1, i].imshow(cf_img, cmap='gray', vmin=0, vmax=255)
        cf_finding_tensor = cf_pa["finding"][i]
        # Handle both scalar and one-hot encoded disease labels
        if cf_finding_tensor.dim() > 0 and cf_finding_tensor.numel() > 1:
            cf_disease = cf_finding_tensor.argmax().item()
        else:
            cf_disease = int(cf_finding_tensor.item())
        axes[1, i].set_title(f'Counterfactual\nDisease: {cf_disease}', fontsize=9)
        axes[1, i].axis('off')

        # Row 2: Difference (CF - O)
        diff_img = diff[i].squeeze().detach().cpu().numpy()
        im = axes[2, i].imshow(diff_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2, i].set_title('Direct Effect\n(CF - O)', fontsize=9)
        axes[2, i].axis('off')

    # Add a single colorbar for the difference row
    plt.tight_layout(rect=[0,0,0.92,1])
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.22])
    fig.colorbar(im, cax=cbar_ax, label='Intensity Δ (CF - O) [clipped at p75]')
    plt.savefig('/workspace/causal-gen/notebooks/disease_counterfactuals.png')
    plt.show()
    
def vae_preprocess(pa: Dict[str, Tensor], parents_order: Optional[list] = None, input_res: int = 192, device: Optional[torch.device] = None) -> Tensor:
    """Construct VAE conditioning tensor.
    Args:
        pa: dict of parent tensors (each (B,C) or (B,)).
        parents_order: explicit ordering list; if None uses pa.keys() insertion order.
        input_res: spatial size to tile to.
        device: target device.
    Returns:
        Tensor of shape (B, sum(C_i), input_res, input_res)
    """
    order = parents_order if parents_order is not None else list(pa.keys())
    feats = []
    for k in order:
        if k not in pa:
            raise KeyError(f"Parent '{k}' missing from provided dict. Keys={list(pa.keys())}")
        v = pa[k]
        if v is None:
            raise ValueError(f"Parent {k} is None; ensure batch was preprocessed (split='l').")
        if v.dim() == 1:
            v = v.unsqueeze(-1)  # (B,1)
        feats.append(v.float())
    cat = torch.cat(feats, dim=1)
    cat = cat[..., None, None].repeat(1, 1, *(input_res,) * 2)
    if device is not None:
        cat = cat.to(device)
    return cat

def test_cf_disease_visualization(vae, pgm, predictor, dataloaders, num_samples=9, te_cf: bool = False):
    """
    Generate and visualize counterfactuals on MIMIC by intervening on disease_label (finding).

    Selection:
      - Choose ~num_samples/2 from each disease class (0,1) if available.
      - If not enough of one class, fill remaining from other class.
    Intervention:
      - Flip disease label: 0 -> 1, 1 -> 0

    Visualization:
      - Use TRUE original and counterfactual parent labels (finding) from SCM / counterfactual generation.
      - Disease (finding) comes from original_pa and cf_pa (PGM outputs), not predictor image-based outputs.
    """
    vae.eval(); pgm.eval(); predictor.eval()

    assert num_samples >= 2, "num_samples should be >= 2 for balancing across 2 disease classes"

    # ----------------- Prepare batch -----------------
    full_batch = next(iter(dataloaders['test']))
    # Move entire batch to GPU (temporarily) to filter
    tmp = {k: v.clone() for k,v in full_batch.items()}
    tmp = preprocess(tmp)  # now on CUDA

    finding = tmp['finding']  # (B, 1) binary label
    finding_labels = finding.squeeze(-1).long()  # (B,) get class indices (0 or 1)
    class_indices = [(finding_labels == c).nonzero(as_tuple=True)[0] for c in range(2)]

    per_class = max(1, num_samples // 2)
    remainder = num_samples - per_class * 2

    sel_parts = []
    for c in range(2):
        idxs = class_indices[c][:per_class]
        sel_parts.append(idxs)
    # Distribute remainder starting from class 0
    if remainder > 0:
        for c in range(2):
            if remainder == 0: break
            extra_pool = class_indices[c][per_class:per_class+1]
            if extra_pool.numel() > 0:
                sel_parts[c] = torch.cat([sel_parts[c], extra_pool])
                remainder -= 1
    sel_idx = torch.cat(sel_parts)
    sel_idx = sel_idx[:num_samples]

    batch = {k: full_batch[k][sel_idx.cpu()] for k in full_batch.keys()}
    batch = preprocess(batch)  # final selected minibatch to CUDA

    device = batch['x'].device
    parents_order = getattr(vae, 'args', getattr(pgm, 'args', None))
    parents_order = getattr(parents_order, 'parents_x', ['finding','age','sex','race'])  # fallback

    original_x = batch['x']
    original_pa = {k: v for k, v in batch.items() if k != 'x'}

    # ----------------- Build intervention (flip disease label) -----------------
    if 'finding' not in original_pa:
        raise KeyError("'finding' not found in batch parents for MIMIC dataset.")
    
    finding = original_pa['finding']  # (B, 1) binary label
    # Flip the disease label: 0 -> 1, 1 -> 0
    finding_cf = 1 - finding
    
    do = {'finding': finding_cf}
    do = {k: v.clone() for k, v in do.items()}  # ensure shape
    do = preprocess(do)  # ensure shape/device

    # ----------------- Generate counterfactual parents via SCM -----------------
    cf_pa = pgm.counterfactual(obs=original_pa, intervention=do, num_particles=1)

    # Ensure cf_pa contains flipped finding (overwrite if SCM kept same due to design)
    cf_pa['finding'] = finding_cf

    # ----------------- Prepare VAE parent tensors -----------------
    _pa_dict = {k: original_pa[k].clone() for k in original_pa}
    _cf_pa_dict = {k: cf_pa[k].clone() for k in cf_pa}
    input_res = getattr(getattr(vae, 'args', None), 'input_res', 192)
    _pa = vae_preprocess(_pa_dict, parents_order=parents_order, input_res=input_res, device=device)
    _cf_pa = vae_preprocess(_cf_pa_dict, parents_order=parents_order, input_res=input_res, device=device)

    # Sanity checks
    assert _pa.shape[0] == original_x.shape[0] == _cf_pa.shape[0], "Batch size mismatch in parent conditioning tensors"

    # ----------------- Latent abduction -----------------
    t_z = t_u = 0.1  # sampling temps
    z = vae.abduct(original_x, parents=_pa, t=t_z)
    if hasattr(vae, 'cond_prior') and vae.cond_prior:
        # For (H)VAE with conditional prior, z is list of dicts -> extract latent tensors
        z = [z[i]['z'] for i in range(len(z))]

    rec_loc, rec_scale = vae.forward_latents(z, parents=_pa)
    u = (original_x - rec_loc) / rec_scale.clamp(min=1e-12)

    if hasattr(vae, 'cond_prior') and vae.cond_prior and te_cf:
        cf_z = vae.abduct(x=original_x, parents=_pa, cf_parents=_cf_pa, alpha=0.65)
        cf_loc, cf_scale = vae.forward_latents(cf_z, parents=_cf_pa)
    else:
        cf_loc, cf_scale = vae.forward_latents(z, parents=_cf_pa)
    cf_scale = cf_scale * t_u
    cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1, max=1)

    # Displays
    orig_display = ((original_x + 1) * 127.5).cpu()
    cf_display = ((cf_x + 1) * 127.5).cpu()

    visualize_counterfactuals(orig_display, cf_display, original_pa, cf_pa, num_samples)

    return orig_display, cf_display, original_pa, cf_pa


dataloaders = setup_dataloaders(pgm_args)
_ = test_cf_disease_visualization(vae, pgm, predictor, dataloaders, num_samples=9, te_cf=True)

if __name__ == '__main__':
    # The notebook likely executed top-level code already.
    # If functions were defined above, you can call an entry point here.
    pass