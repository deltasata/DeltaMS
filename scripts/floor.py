"""Non-deep-learning floor: difference imaging for new FLAIR lesions.

The astronomy recipe, applied to two co-registered brain scans:
  1. head mask        : where to search (Otsu threshold, largest component, eroded edge)
  2. bias correction  : N4 removes slow brightness drifts across each scan (like flat-fielding)
  3. photometric match: robust z-score of each scan inside the mask, so the two "exposures" agree
  4. difference image : D = z(follow-up) - z(baseline), smoothed, in units of its robust noise (significance)
  5. detection        : significant in D, bright at follow-up, NOT bright at baseline (rejects old lesions)
  6. cleaning         : connected components >= 3 mm^3; confidence = peak significance

Steps 1-4 are computed once per patient (`prepare`); step 5-6 (`detect`) are cheap, so thresholds can be tuned.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

STRUCT26 = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class FloorParams:
    k_diff: float = 4.0       # significance threshold on the difference image (robust sigmas)
    k_bright: float = 2.0     # follow-up z-score must exceed this
    k_base_max: float = 1.5   # baseline z-score must stay below this
    min_mm3: float = 3.0      # minimum lesion volume (as the official evaluation)


def robust_stats(x):
    med = np.median(x)
    return med, 1.4826 * np.median(np.abs(x - med))


def head_mask(img, spacing, erode_mm=3.0):
    m = sitk.GetArrayFromImage(sitk.OtsuThreshold(sitk.GetImageFromArray(img.astype(np.float32)), 0, 1)).astype(bool)
    lab, n = ndimage.label(m)
    if n:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    m = ndimage.binary_fill_holes(m)
    it = max(1, int(round(erode_mm / min(spacing))))
    return ndimage.binary_erosion(m, iterations=it)


def n4(img, mask, spacing):
    im = sitk.GetImageFromArray(img.astype(np.float32))
    im.SetSpacing(tuple(float(s) for s in spacing[::-1]))
    mk = sitk.GetImageFromArray(mask.astype(np.uint8))
    mk.CopyInformation(im)
    ims, mks = sitk.Shrink(im, [3] * 3), sitk.Shrink(mk, [3] * 3)
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations([30, 20, 10])
    corr.Execute(ims, mks)
    return sitk.GetArrayFromImage(im / sitk.Exp(corr.GetLogBiasFieldAsImage(im)))


def prepare(t1_path: Path, t2_path: Path, smooth_mm: float = 1.0):
    i1, i2 = nib.load(str(t1_path)), nib.load(str(t2_path))
    spacing = tuple(float(z) for z in i1.header.get_zooms()[:3])
    a1, a2 = i1.get_fdata(dtype=np.float32), i2.get_fdata(dtype=np.float32)
    m = head_mask(np.maximum(a1, a2), spacing)
    z = []
    for a in (n4(a1, m, spacing), n4(a2, m, spacing)):
        med, mad = robust_stats(a[m])
        z.append(((a - med) / (mad + 1e-6)).astype(np.float32))
    d = ndimage.gaussian_filter(z[1] - z[0], sigma=[smooth_mm / s for s in spacing])
    dmed, dmad = robust_stats(d[m])
    sig = ((d - dmed) / (dmad + 1e-6)).astype(np.float32)
    return dict(z1=z[0], z2=z[1], sig=sig, mask=m, spacing=np.array(spacing), affine=i1.affine)


def detect(prep: dict, p: FloorParams):
    cand = prep["mask"] & (prep["sig"] > p.k_diff) & (prep["z2"] > p.k_bright) & (prep["z1"] < p.k_base_max)
    lab, n = ndimage.label(cand, structure=STRUCT26)
    out = np.zeros(cand.shape, np.uint8)
    lesions = []
    if n:
        vox = float(np.prod(prep["spacing"]))
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum(cand, lab, idx) * vox
        peaks = ndimage.maximum(prep["sig"], lab, idx)
        keep = idx[sizes >= p.min_mm3]
        out[np.isin(lab, keep)] = 1
        lesions = [dict(id=int(i), volume_mm3=float(s), confidence=float(pk))
                   for i, s, pk in zip(idx, sizes, peaks) if s >= p.min_mm3]
    return out, lesions
