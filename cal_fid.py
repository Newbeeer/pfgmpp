from scipy import linalg
import numpy as np

def calculate_fid_from_inception_stats():
    ref = dict(np.load('fid-refs/cifar10-32x32.npz'))
    gen = dict(np.load('training-runs/pfgm_ddpmpp/ckpt_085299/fid-stats.npz'))
    mu_ref = ref['mu']
    sigma_ref = ref['sigma']

    mu = gen['mu']
    sigma = gen['sigma']
    m = np.square(mu - mu_ref).sum()
    s, _ = linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    print("FID:", float(np.real(fid)))
    return float(np.real(fid))