import torch

compute_basis_size = lambda order, mirror: ((order + 1) * (order + 2)) // (1 if mirror else 2)


def herme_vander_torch(z, m):
    He0 = z.new_ones(z.shape)
    if m == 0: return He0[:, None]
    H = [He0, z]
    for n in range(1, m):
        H.append(z * H[-1] - n * H[-2])
    return torch.stack(H, 1)


def gauss_deriv(max_order, device, dtype, kernel_size, sigma=None, include_negations=False, scale_magnitude=True):
    sigma = (kernel_size // 2) / 1.645 if sigma is None else sigma
    if kernel_size % 2 == 0: raise ValueError("ksize must be odd")
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    z = x / sigma
    g = torch.exp(-0.5 * z ** 2) / (sigma * (2.0 * torch.pi) ** 0.5)
    He = herme_vander_torch(z, max_order)
    derivs_1d = [(((-1) ** n) / (sigma ** n) if scale_magnitude else (-1) ** n) * He[:, n] * g for n in
                 range(max_order + 1)]
    bank = []
    for o in range(max_order + 1):
        for i in range(o + 1):
            K = torch.outer(derivs_1d[o - i], derivs_1d[i])
            bank.append(K)
            if include_negations: bank.append(-K)
    return torch.stack(bank, 0)
