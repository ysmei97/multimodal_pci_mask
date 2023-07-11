import torch


def hscore(f, g):
    """Define correlation loss"""
    f0 = f - torch.mean(f, 0)
    g0 = g - torch.mean(g, 0)
    corr = torch.mean(torch.sum(f0 * g0, 1))
    cov_f = torch.matmul(torch.transpose(f0, 0, 1), f0) / float(list(f0.shape)[0] - 1)
    cov_g = torch.matmul(torch.transpose(g0, 0, 1), g0) / float(list(g0.shape)[0] - 1)
    return -corr + torch.sum(cov_f * cov_g)/2


def hscore_A(f, g, A):
    """Correlation loss with the diagonal matrix A between two input features"""
    f0 = f - torch.mean(f, 0)
    g0 = g - torch.mean(g, 0)
    corr = torch.mean(torch.sum(f0 * A * g0, 1))
    s = (torch.matmul((f0 * A), g0.t())**2) / float((f0.size(0)-1)**2)
    return -corr + torch.sum(s)/2.0


def gredient_A(f, g, A, gamma=2.0):
    """Calculate the gradient of matrix for further update"""
    f0 = f - torch.mean(f, 0)
    g0 = g - torch.mean(g, 0)
    fAg = torch.matmul((g0 * A), f0.t())
    d_L = 2.0 * torch.sum(torch.matmul(f0.t(), fAg) * g0.t(), 1)
    A = A - gamma * d_L
    return A


def projection_A(A, c, size):
    """Use project gradient descent to update the matrix A in an unsupervised manner"""
    B = truncate_A(A)
    b1 = 0
    b2 = -b1
    if b1 < 0:
        while torch.abs(torch.sum(B) - c) > c * 0.01:
            r = (b1 + b2)/2
            B = truncate_A(A - r)
            if (torch.sum(B) - c) > 0:
                b1 = r
            else:
                b2 = r
        return B
    elif b1 == 0:
        return B
    else:
        while torch.abs(torch.sum(B) - c) > c * 0.01:
            r = (b1 + b2)/2
            B = truncate_A(A - r)
            if (torch.sum(B) - c) > 0:
                b2 = r
            else:
                b1 = r
        return B


def truncate_A(A):
    """Simultaneously truncate all the elements in matrix A to range (0,1)"""
    for i in range(list(A.shape)[0]):
        if A[i] < 0:  # TEST
            A[i] = 0
        elif A[i] > 1:  # TEST
            A[i] = 1
        else:
            continue
    return A
