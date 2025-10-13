#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------- IO ----------
def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def read_mask01(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8)

def to_bgr(img_gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

def overlay_vis_three(img_gray: np.ndarray, ref_mask: np.ndarray, pred_mask: np.ndarray, drop_mask: np.ndarray, out_path: str):
    base = to_bgr(img_gray).astype(np.float32)
    if ref_mask is not None:
        blue = np.zeros_like(base); blue[...,0]=255
        base = np.where(ref_mask[...,None].astype(bool), base*0.75+blue*0.25, base)
    if pred_mask is not None:
        red = np.zeros_like(base); red[...,2]=255
        base = np.where(pred_mask[...,None].astype(bool), base*0.7+red*0.3, base)
    if drop_mask is not None:
        yellow = np.zeros_like(base); yellow[...,0]=255; yellow[...,2]=255
        base = np.where(drop_mask[...,None].astype(bool), base*0.6+yellow*0.4, base)
    base = np.clip(base,0,255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, base)

# ---------- Data ----------
def list_set(root: str, limit: int = 10) -> List[Dict[str,str]]:
    img_dir = os.path.join(root, 'image')
    msk_dir = os.path.join(root, 'mask')
    ovl_dir = os.path.join(root, 'overlay')  # optional
    paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))[:limit]
    rows = []
    for ip in paths:
        base = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(msk_dir, base+'.png')
        ov = os.path.join(ovl_dir, base+'.png')
        if not os.path.exists(mp):
            continue
        rows.append(dict(image=ip, mask=mp, overlay=ov if os.path.exists(ov) else '', id=base))
    return rows

# ---------- Model Decoder ----------
class HRDecoder(nn.Module):
    def __init__(self, in_ch: int = 256):
        super().__init__()
        def up(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        self.up1 = up(in_ch, 128)  # 64->128
        self.up2 = up(128, 64)     # 128->256
        self.up3 = up(64, 32)      # 256->512
        self.up4 = up(32, 16)      # 512->1024
        self.out = nn.Conv2d(16, 1, 1)
    def forward(self, z):          # z: [B,256,64,64]
        x = self.up1(z); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        return self.out(x)         # [B,1,1024,1024]

# ---------- Teacher-Student Utilities ----------
def soften_binary_mask(mask01_hw: torch.Tensor, ring_width: int = 3) -> torch.Tensor:
    """mask01_hw: [1,1,H,W] float/byte in {0,1} -> soft labels with 0.5 ring."""
    device = mask01_hw.device
    m = (mask01_hw > 0.5).float()
    try:
        import kornia.morphology as km
        ker = torch.ones(1,1,2*ring_width+1,2*ring_width+1, device=device)
        dil = km.dilation(m, ker); ero = km.erosion(m, ker)
    except Exception:
        # Fallback with max/min pooling
        dil = F.max_pool2d(m, kernel_size=2*ring_width+1, stride=1, padding=ring_width)
        ero = -F.max_pool2d(-m, kernel_size=2*ring_width+1, stride=1, padding=ring_width)
    ring = (dil - ero).clamp(min=0.0)
    soft = m.clone()
    soft[ring>0] = 0.5
    return soft

def _sample_points_from_mask(mask_s: np.ndarray, num_pos: int = 8, num_neg: int = 8, border: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coords[P,2], labels[P]) in resized mask coords (x,y)."""
    Hs, Ws = mask_s.shape[:2]
    pos_coords = []
    ys, xs = np.where(mask_s > 0)
    if len(xs) > 0:
        idx = np.random.choice(len(xs), size=min(num_pos, len(xs)), replace=False)
        for i in idx:
            pos_coords.append([xs[i], ys[i]])
    # Negative: ring outside mask via dilation
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*border+1, 2*border+1))
    dil = cv2.dilate((mask_s>0).astype(np.uint8), ker, 1)
    ring = ((dil>0) & (mask_s==0)).astype(np.uint8)
    yn, xn = np.where(ring>0)
    neg_coords = []
    if len(xn) > 0:
        idx = np.random.choice(len(xn), size=min(num_neg, len(xn)), replace=False)
        for i in idx:
            neg_coords.append([xn[i], yn[i]])
    coords = np.array(pos_coords + neg_coords, dtype=np.float32) if (pos_coords or neg_coords) else np.zeros((0,2), np.float32)
    labels = np.array([1]*len(pos_coords) + [0]*len(neg_coords), dtype=np.int64) if (pos_coords or neg_coords) else np.zeros((0,), np.int64)
    return coords, labels

def _bbox_from_mask(mask_s: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_s>0)
    if len(xs)==0:
        return np.zeros((0,), dtype=np.float32)
    x1, x2 = xs.min(), xs.max(); y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)

@torch.no_grad()
def sam_teacher_logits_from_prompts(sam, emb: torch.Tensor, mask_s: np.ndarray, Hs: int, Ws: int, device: str) -> torch.Tensor:
    """Return teacher low-res logits upsampled to [1,1,1024,1024] using point+box prompts from mask_s (Hs,Ws)."""
    coords, labels = _sample_points_from_mask(mask_s, num_pos=8, num_neg=8, border=5)
    box = _bbox_from_mask(mask_s)
    points = None
    if coords.shape[0] > 0:
        pt_coords = torch.from_numpy(coords).unsqueeze(0).to(device)
        pt_labels = torch.from_numpy(labels).unsqueeze(0).to(device)
        points = (pt_coords, pt_labels)
    box_t = None
    if box.shape[0]==4:
        box_t = torch.from_numpy(box).unsqueeze(0).to(device)
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=points,
        boxes=box_t,
        masks=None
    )
    low_res_logits, _ = sam.mask_decoder(
        image_embeddings=emb,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False
    )
    logits_1024 = F.interpolate(low_res_logits, size=(1024,1024), mode='bilinear', align_corners=False)
    return logits_1024

def dice_loss_continuous(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (prob*target).sum(dim=(1,2,3))
    denom = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = 1 - (2*inter + 1.0) / (denom + 1.0 + eps)
    return dice.mean()

def bce_dice_with_soft_targets(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, soft_targets)
    prob = torch.sigmoid(logits)
    dice = dice_loss_continuous(prob, soft_targets)
    return bce + dice

def flip_tensor_lr(t: torch.Tensor) -> torch.Tensor:
    return torch.flip(t, dims=[-1])

# ---------- Train / Eval ----------
def get_sam_embeddings(sam, img_gray: np.ndarray, device: str) -> Tuple[torch.Tensor, int, int]:
    transform = ResizeLongestSide(1024)
    img3 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_sam = transform.apply_image(img3).astype(np.float32)
    Hs, Ws = img_sam.shape[:2]
    image_t = torch.from_numpy(img_sam).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = sam.image_encoder(sam.preprocess(image_t[0]).unsqueeze(0))
    return emb, Hs, Ws

def pad_to_1024(t01_hw: torch.Tensor, Hs: int, Ws: int, device: str) -> torch.Tensor:
    pad_h = 1024 - Hs; pad_w = 1024 - Ws
    pad = (0, pad_w, 0, pad_h)
    t = F.pad(t01_hw.unsqueeze(0).unsqueeze(0), pad, mode='constant', value=0.0)
    return t.to(device)

def dice_bce_boundary_loss(logits_1024: torch.Tensor, target01_1024: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits_1024, target01_1024)
    p = torch.sigmoid(logits_1024)
    inter = (p*target01_1024).sum(dim=(1,2,3))
    dice = 1 - (2*inter+1) / (p.sum(dim=(1,2,3)) + target01_1024.sum(dim=(1,2,3)) + 1)
    # boundary ring
    with torch.no_grad():
        ker = torch.ones(1,1,5,5, device=logits_1024.device)
        try:
            import kornia.morphology as km
            dil = km.dilation(target01_1024, ker); ero = km.erosion(target01_1024, ker)
            ring = (dil - ero).clamp(min=0)
        except Exception:
            ring = torch.zeros_like(target01_1024)
    boundary_l1 = (torch.sigmoid(logits_1024)-target01_1024).abs() * ring
    boundary_l1 = boundary_l1.mean(dim=(1,2,3))
    return bce + dice.mean() + 0.5*boundary_l1.mean()

def train_head(
    sam,
    rows: List[Dict[str,str]],
    device: str,
    epochs: int = 60,
    lr: float = 1e-4,
    alpha: float = 5.0,
    beta: float = 0.0,
    gamma: float = 1.0,
    delta: float = 0.0,
    warmup_epochs: int = 5,
    consistency_pred: bool = True,
    consistency_emb: bool = False,
) -> nn.Module:
    head = HRDecoder(in_ch=256).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad = False
    for epoch in tqdm(range(epochs), desc='few-shot-train', leave=False):
        for r in rows:
            img = read_gray(r['image'])
            gt  = read_mask01(r['mask'])
            emb, Hs, Ws = get_sam_embeddings(sam, img, device)
            gt_s = cv2.resize(gt, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
            tgt_1024 = pad_to_1024(torch.from_numpy(gt_s).float(), Hs, Ws, device)

            # Teacher prediction from prompts (gt-derived)
            try:
                teacher_logits_1024 = sam_teacher_logits_from_prompts(sam, emb, gt_s, Hs, Ws, device)
            except Exception:
                teacher_logits_1024 = None

            opt.zero_grad()
            student_logits_1024 = head(emb)

            # Weak supervision with softened labels
            tgt_1024_soft = soften_binary_mask(tgt_1024)
            l_ws = bce_dice_with_soft_targets(student_logits_1024, tgt_1024_soft)

            # Teacher-student dice loss
            if teacher_logits_1024 is not None:
                student_prob = torch.sigmoid(student_logits_1024)
                teacher_prob = torch.sigmoid(teacher_logits_1024).detach()
                l_ts = dice_loss_continuous(student_prob, teacher_prob)
            else:
                l_ts = torch.tensor(0.0, device=device)

            # Prediction consistency (horizontal flip)
            if consistency_pred:
                img_flipped = cv2.flip(img, 1)
                emb_f, Hsf, Wsf = get_sam_embeddings(sam, img_flipped, device)
                student_logits_f = head(emb_f)
                prob = torch.sigmoid(student_logits_1024)
                prob_f = torch.sigmoid(student_logits_f)
                prob_flip_back = flip_tensor_lr(prob_f)
                l_cons_pred = F.mse_loss(prob_flip_back, prob)
            else:
                l_cons_pred = torch.tensor(0.0, device=device)

            # Embedding consistency (teacher vs student positive-region means on 64x64)
            if consistency_emb and teacher_logits_1024 is not None:
                prob_s_64 = F.interpolate(torch.sigmoid(student_logits_1024), size=(64,64), mode='bilinear', align_corners=False)
                prob_t_64 = F.interpolate(torch.sigmoid(teacher_logits_1024), size=(64,64), mode='bilinear', align_corners=False)
                m_s = (prob_s_64 > 0.5).float()
                m_t = (prob_t_64 > 0.5).float()
                emb_feat = emb  # [1,256,64,64]
                def region_mean(feat, mask):
                    w = mask
                    s = w.sum(dim=(2,3), keepdim=True).clamp(min=1.0)
                    return (feat*w).sum(dim=(2,3), keepdim=True) / s
                mu_s = region_mean(emb_feat, m_s)
                mu_t = region_mean(emb_feat, m_t).detach()
                l_cons_emb = F.mse_loss(mu_s, mu_t)
            else:
                l_cons_emb = torch.tensor(0.0, device=device)

            # Schedule weights
            if epoch < warmup_epochs:
                wa, wb, wg, wd = 0.0, 1.0, 0.0, 0.0
            else:
                wa, wb, wg, wd = alpha, beta, gamma, delta

            loss = wa*l_ts + wb*l_ws + wg*l_cons_pred + wd*l_cons_emb
            loss.backward(); opt.step()
    return head

# ---------- Drop: instance coverage ----------
def _morph(bin_, op, k=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    return cv2.morphologyEx(bin_.astype(np.uint8), op, kernel)

def _remove_small(bin_, min_px):
    nb, comp = cv2.connectedComponents(bin_.astype(np.uint8))
    out = np.zeros_like(bin_, np.uint8)
    for i in range(1, nb):
        if (comp==i).sum() >= min_px: out[comp==i]=1
    return out

def _cc_list(bin01: np.ndarray):
    bin01 = (bin01>0).astype(np.uint8)
    nb, comp = cv2.connectedComponents(bin01)
    masks = []
    for i in range(1, nb):
        m = (comp==i).astype(np.uint8)
        if m.sum()>0: masks.append(m)
    return masks

def _merge_close_by_dilate(bin01: np.ndarray, r_merge: int) -> np.ndarray:
    if r_merge<=0: return (bin01>0).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_merge+1, 2*r_merge+1))
    dil = cv2.dilate((bin01>0).astype(np.uint8), ker, 1)
    return (dil>0).astype(np.uint8)

def _aspect_ratio(mask01: np.uint8) -> float:
    cnts,_ = cv2.findContours((mask01>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 1.0
    c = max(cnts, key=cv2.contourArea)
    (w,h) = cv2.minAreaRect(c)[1]
    if w<=1e-6 or h<=1e-6: return 1.0
    ma, mi = (max(w,h), min(w,h)); return float(ma/mi)

def _solidity(mask01: np.uint8) -> float:
    cnts,_ = cv2.findContours((mask01>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 1.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c); hull = cv2.convexHull(c); harea = cv2.contourArea(hull)
    if harea<=1e-6: return 1.0
    return float(area/harea)

def classify_drop_instances(I_gray: np.ndarray, M_orig: np.ndarray, M_pred: np.ndarray,
                            L: np.ndarray=None, w_dilate_px: int=10, tau_cov: float=0.5,
                            A_min_px: int=None, A_min_pct: float=0.001, r_merge_px: int=6,
                            tau_idr: float=0.2, tau_adr: float=0.12, tau_area_pct_label: float=0.003, u_min: int=1) -> dict:
    H,W = I_gray.shape[:2]
    if L is None: L = np.ones_like(M_pred, np.uint8)
    # 1) clean
    M_orig = ((M_orig>0).astype(np.uint8) & L)
    M_pred = ((M_pred>0).astype(np.uint8) & L)
    A_L = max(int(L.sum()), 1)
    min_area = max(200, int(0.002*A_L))
    for name in ['M_orig','M_pred']:
        X = locals()[name]
        X = _remove_small(X, min_area); X = _morph(X, cv2.MORPH_OPEN, 2); X = _morph(X, cv2.MORPH_CLOSE, 2)
        locals()[name] = X
    # 2) instance merge
    M_pred_merge = _merge_close_by_dilate(M_pred, r_merge_px)
    P_masks = _cc_list(M_pred_merge)
    J = len(P_masks)
    if A_min_px is None: A_min_px = max(300, int(A_min_pct*A_L))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w_dilate_px+1, 2*w_dilate_px+1))
    O_dil = cv2.dilate(M_orig, ker, 1)
    # 3) coverage match
    U = []; per_pred=[]
    for j, Pj in enumerate(P_masks):
        area_j = int(Pj.sum())
        if area_j < A_min_px:
            per_pred.append(dict(j=j, area=area_j, cov=0.0, keep=False)); continue
        cov = float(((Pj & O_dil)>0).sum()) / float(area_j + 1e-6)
        keep = (cov >= tau_cov)
        per_pred.append(dict(j=j, area=area_j, cov=cov, keep=keep))
        if not keep:
            # light shape gating
            ar = _aspect_ratio(Pj); sol = _solidity(Pj)
            if (ar >= 2.0) or (sol <= 0.9): U.append(Pj)
    D_inst = np.zeros_like(M_pred, np.uint8)
    for u in U: D_inst = (D_inst | u).astype(np.uint8)
    ADR = float(D_inst.sum()) / float(max(int(M_pred.sum()),1))
    IDR = float(len(U)) / float(max(J,1))
    # 4) final label
    label = 'NORMAL' if (J==0 or (int(M_pred.sum())/float(A_L) <= tau_area_pct_label)) else ('DROP' if (len(U)>=u_min or IDR>=tau_idr or ADR>=tau_adr) else 'ATEL-COMPLETE')
    return dict(J=J, U=len(U), IDR=IDR, ADR=ADR, D_inst=D_inst, per_pred=per_pred, label=label)

# ---------- Inference (instance-drop 기반) ----------
def infer_set(sam, head, rows: List[Dict[str,str]], device: str, out_dir: str, split: str,
              logit_temp: float=1.0, w_dilate_px:int=10, tau_cov:float=0.5,
              A_min_px:int=None, A_min_pct:float=0.001, r_merge_px:int=6,
              prob_thr: float=0.5, tau_idr: float=0.2, tau_adr: float=0.12, tau_area_pct_label: float=0.003, u_min: int=1) -> Tuple[List[Dict], Dict[str,int]]:
    os.makedirs(out_dir, exist_ok=True)
    mask_save_dir  = os.path.join(out_dir, 'masks', split)
    drop_save_dir  = os.path.join(out_dir, 'drop_masks', split)
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(drop_save_dir, exist_ok=True)

    pred_rows = []
    for r in rows:
        img = read_gray(r['image']); H,W = img.shape[:2]
        gt  = read_mask01(r['mask'])
        emb, Hs, Ws = get_sam_embeddings(sam, img, device)
        with torch.no_grad():
            logits_1024 = head(emb) / max(1e-6, logit_temp)
            prob_1024 = torch.sigmoid(logits_1024)[0,0].cpu().numpy()
        prob_valid = prob_1024[:Hs,:Ws]
        pred = (prob_valid>prob_thr).astype(np.uint8)
        if pred.sum()==0 and prob_valid.max()>0:
            pv_u8 = (prob_valid*255).astype(np.uint8)
            _, ots = cv2.threshold(pv_u8, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            pred = ots.astype(np.uint8)
            if pred.sum()==0:
                thr = np.percentile(prob_valid, 99.0); pred = (prob_valid>=thr).astype(np.uint8)
        pred = cv2.resize(pred, (W,H), interpolation=cv2.INTER_NEAREST)

        # --- instance coverage classification ---
        inst = classify_drop_instances(img, gt, pred,
                                       L=None, w_dilate_px=w_dilate_px, tau_cov=tau_cov,
                                       A_min_px=A_min_px, A_min_pct=A_min_pct, r_merge_px=r_merge_px,
                                       tau_idr=tau_idr, tau_adr=tau_adr, tau_area_pct_label=tau_area_pct_label, u_min=u_min)
        label = inst['label']
        # save masks/vis
        pred_mask_p = os.path.join(mask_save_dir, f"{r['id']}.png")
        drop_mask_p = os.path.join(drop_save_dir, f"{r['id']}.png")
        cv2.imwrite(pred_mask_p, (pred*255).astype(np.uint8))
        cv2.imwrite(drop_mask_p, (inst['D_inst']*255).astype(np.uint8))
        vis_p = os.path.join(out_dir, f"{split}_{r['id']}_vis.png")
        overlay_vis_three(img, gt, pred, None, vis_p)

        row = dict(id=r['id'], image_path=r['image'], mask_path=r['mask'],
                   pred_label=label, split=split, out_mask_path=pred_mask_p,
                   drop_mask_path=drop_mask_p, J=inst['J'], U=inst['U'],
                   IDR=inst['IDR'], ADR=inst['ADR'])
        pred_rows.append(row)
    return pred_rows, {}

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sam_ckpt', required=True)
    ap.add_argument('--vit', default='vit_b')

    ap.add_argument('--train_root',  default='/opt/pxi/projects/label_filtering/medgemma-27b/data/atel_images')
    ap.add_argument('--normal_root', default='/opt/pxi/projects/label_filtering/medgemma-27b/data/normal')
    ap.add_argument('--drop_root',   default='/opt/pxi/projects/label_filtering/medgemma-27b/data/drop_images')

    ap.add_argument('--shots',  type=int,   default=20)
    ap.add_argument('--epochs', type=int,   default=120)
    ap.add_argument('--lr',     type=float, default=1e-4)
    ap.add_argument('--logit_temp', type=float, default=1.0)

    # instance-drop hyperparams
    ap.add_argument('--w_dilate_px', type=int,   default=10)     # label dilation width (8~12 @1024)
    ap.add_argument('--tau_cov',     type=float, default=0.5)    # coverage threshold
    ap.add_argument('--A_min_px',    type=int,   default=0)      # min instance px (if 0 use pct)
    ap.add_argument('--A_min_pct',   type=float, default=0.001)  # 0.1% of lung
    ap.add_argument('--r_merge_px',  type=int,   default=6)      # merge nearby predictions

    ap.add_argument('--prob_thr', type=float, default=0.5)       # probability threshold for binarization
    ap.add_argument('--tau_idr', type=float, default=0.2)        # instance drop rate threshold
    ap.add_argument('--tau_adr', type=float, default=0.12)       # area drop rate threshold
    ap.add_argument('--tau_area_pct_label', type=float, default=0.003)  # overall predicted area ratio to label NORMAL
    ap.add_argument('--u_min', type=int, default=1)              # minimum number of drop instances U to label DROP

    ap.add_argument('--out_dir', default='/opt/pxi/projects/label_filtering/zeroshotSAM/vis/fewshot_sam')
    ap.add_argument('--save_head_path', default='/opt/pxi/projects/label_filtering/zeroshotSAM/checkpoints/fewshot_head_vit_b_atel_hr.pth')
    ap.add_argument('--load_head_path', default='')

    # teacher-student hparams
    ap.add_argument('--alpha', type=float, default=5.0)
    ap.add_argument('--beta',  type=float, default=0.0)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--delta', type=float, default=0.0)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--consistency_pred', action='store_true')
    ap.add_argument('--consistency_emb', action='store_true')

    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry[args.vit](checkpoint=args.sam_ckpt).to(device)
    sam.eval()

    # data
    train_rows  = list_set(args.train_root,  limit=args.shots)
    normal_rows = list_set(args.normal_root, limit=200)
    drop_rows   = list_set(args.drop_root,   limit=200)

    # train / load few-shot head
    if args.load_head_path and os.path.exists(args.load_head_path):
        head = HRDecoder(in_ch=256).to(device)
        head.load_state_dict(torch.load(args.load_head_path, map_location=device))
        print(f"[FEWSHOT] Loaded head: {args.load_head_path}")
    else:
        head = train_head(
            sam, train_rows, device,
            epochs=args.epochs, lr=args.lr,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
            warmup_epochs=args.warmup_epochs,
            consistency_pred=args.consistency_pred,
            consistency_emb=args.consistency_emb,
        )
        if args.save_head_path:
            os.makedirs(os.path.dirname(args.save_head_path), exist_ok=True)
            torch.save(head.state_dict(), args.save_head_path)
            print(f"[FEWSHOT] Saved head: {args.save_head_path}")

    # inference (instance coverage → label)
    preds_all = []
    for split, rows in [('normal', normal_rows), ('atel', train_rows), ('drop', drop_rows)]:
        pr, _ = infer_set(
            sam, head, rows, device, args.out_dir, split,
            logit_temp=args.logit_temp,
            w_dilate_px=args.w_dilate_px,
            tau_cov=args.tau_cov,
            A_min_px=(None if args.A_min_px<=0 else args.A_min_px),
            A_min_pct=args.A_min_pct,
            r_merge_px=args.r_merge_px,
            prob_thr=args.prob_thr,
            tau_idr=args.tau_idr,
            tau_adr=args.tau_adr,
            tau_area_pct_label=args.tau_area_pct_label,
            u_min=args.u_min
        )
        preds_all += pr

    # confusion / acc (split을 GT로 간주)
    classes = ['normal','atel','drop']
    cm = {c:{d:0 for d in classes} for c in classes}
    correct = 0
    for r in preds_all:
        g = r['split']; p = r['pred_label']
        if p not in classes: p='atel'
        cm[g][p]+=1
        if g==p: correct+=1
    acc = correct / max(1,len(preds_all))

    # save csv
    os.makedirs(args.out_dir, exist_ok=True)
    cm_path  = os.path.join(args.out_dir, 'confusion_matrix.csv')
    with open(cm_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['gt\\pred']+classes)
        for g in classes: w.writerow([g]+[cm[g][p] for p in classes])
    res_path = os.path.join(args.out_dir, 'preds_fewshot.csv')
    with open(res_path, 'w', newline='') as f:
        fieldnames = list(preds_all[0].keys()) if preds_all else ['id','image_path','mask_path','pred_label','split']
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader();
        if preds_all: w.writerows(preds_all)
    print('ACC=', acc, 'saved:', cm_path, res_path)

if __name__ == '__main__':
    main()
