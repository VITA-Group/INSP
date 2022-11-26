import torch
from torch.autograd import grad


def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status

def all_2(y, x):
    grad = gradient(y, x)
    grad_x = grad[..., 0]
    grad_y = grad[..., 1]
    grad_xx = torch.autograd.grad(grad_x, x, torch.ones_like(grad_x), create_graph=True)[0][...,0]
    grad_xy = torch.autograd.grad(grad_x, x, torch.ones_like(grad_x), create_graph=True)[0][...,1]
    grad_yx = torch.autograd.grad(grad_y, x, torch.ones_like(grad_y), create_graph=True)[0][...,0]
    grad_yy = torch.autograd.grad(grad_y, x, torch.ones_like(grad_y), create_graph=True)[0][...,1]
    return torch.cat([grad_x, grad_y, grad_xx, grad_xy, grad_yx, grad_yy], dim=0)

def second_order(y, x):
    grad = gradient(y, x)
    grad_x = grad[..., 0]
    grad_y = grad[..., 1]
    grad_xx = torch.autograd.grad(grad_x, x, torch.ones_like(grad_x), create_graph=True)[0][...,0:1]
    grad_xy = torch.autograd.grad(grad_x, x, torch.ones_like(grad_x), create_graph=True)[0][...,1:]
    grad_yx = torch.autograd.grad(grad_y, x, torch.ones_like(grad_y), create_graph=True)[0][...,0:1]
    grad_yy = torch.autograd.grad(grad_y, x, torch.ones_like(grad_y), create_graph=True)[0][...,1:]
    return grad_xx, grad_xy, grad_yx, grad_yy

def third_order(y, x):
    grad = gradient(y, x)
    grad_x = grad[..., 0]
    grad_y = grad[..., 1]
    grad_xy = torch.autograd.grad(grad_x, x, torch.ones_like(grad_x), create_graph=True)[0][...,1]
    grad_xyy = torch.autograd.grad(grad_xy, x, torch.ones_like(grad_x), create_graph=True)[0][...,1]
    grad_yx = torch.autograd.grad(grad_y, x, torch.ones_like(grad_y), create_graph=True)[0][...,0]
    grad_yxx = torch.autograd.grad(grad_yx, x, torch.ones_like(grad_y), create_graph=True)[0][...,0]
    return grad_xyy.unsqueeze(-1), grad_yxx.unsqueeze(-1)

def all_3(y, x):
    grad = gradient(y / 256, x)
    grad_x = grad[..., 0]
    grad_y = grad[..., 1]
    # print(grad_x, grad_y)
    ww = torch.autograd.grad(grad_x / 256, x, torch.ones_like(grad_x), create_graph=True)[0]
    grad_xx = ww[..., 0]
    grad_xy = ww[..., 1]
    ww = torch.autograd.grad(grad_xx / 256, x, torch.ones_like(grad_x), create_graph=True)[0]
    grad_xxx = ww[..., 0]
    grad_xxy = ww[..., 1]
    ww = torch.autograd.grad(grad_xy / 256, x, torch.ones_like(grad_x), create_graph=True)[0]
    grad_xyx = ww[..., 0]
    grad_xyy = ww[..., 1]

    ww = torch.autograd.grad(grad_y / 256, x, torch.ones_like(grad_y), create_graph=True)[0]
    grad_yx = ww[...,0]
    grad_yy = ww[...,1]
    ww = torch.autograd.grad(grad_yx / 256, x, torch.ones_like(grad_y), create_graph=True)[0]
    grad_yxx = ww[...,0]
    grad_yxy = ww[...,1]
    ww = torch.autograd.grad(grad_yy / 256, x, torch.ones_like(grad_y), create_graph=True)[0]
    grad_yyx = ww[..., 0]
    grad_yyy = ww[..., 1]
    res = torch.cat([y.squeeze(-1), grad_x, grad_y, grad_xx, grad_xy, grad_yx, grad_yy, grad_xxx, grad_xxy, grad_xyx, grad_xyy, grad_yxx, grad_yxy, grad_yyx, grad_yyy]).unsqueeze(-1)
    return res

def new_grad(y, x, sz=256, num=31):
    li = [y.squeeze(-1)]
    for i in range(num):
        cur = li[i]
        ww = torch.autograd.grad(cur / sz, x, torch.ones_like(cur), create_graph=True)[0]
        li.append(ww[..., 0])
        li.append(ww[..., 1])
    return torch.cat(li, dim=0).unsqueeze(-1)

def new_grad_lastdim(y, x, sz=256, num=31):
    li = [y.squeeze(-1)]
    for i in range(num):
        cur = li[i]
        ww = torch.autograd.grad(cur / sz, x, torch.ones_like(cur), create_graph=True)[0]
        li.append(ww[..., 0])
        li.append(ww[..., 1])
    return torch.stack(li, dim=-1)

def new_grad_xonly(y, x, sz=256):
    li = [y.squeeze(-1)]
    new = []
    for i in range(11):
        cur = li[i]
        ww = torch.autograd.grad(cur / 256, x, torch.ones_like(cur), create_graph=True)[0]
        li.append(ww[..., 0])
        new.append(ww[..., 1])
    return torch.stack([*li, *new], dim=0).unsqueeze(-1)

def new_grad_audio(y, x, num=3, sz=256):
    li = [y.squeeze(-1)]
    for i in range(num):
        cur = li[i]
        ww = torch.autograd.grad(cur / 256, x, torch.ones_like(cur), create_graph=True)[0]
        li.append(ww[..., 0])
    return torch.cat([*li], dim=0).unsqueeze(-1)


def norm(x):
    return x / x.max()

def grad_1dim_x(y, x, sz=256):
    grad = torch.autograd.grad(y / sz, x, torch.ones_like(y), create_graph=True)[0]
    grad_x = grad[..., 0]
    grad_y = grad[..., 1]
    ww = torch.autograd.grad(grad_x / sz, x, torch.ones_like(grad_x), create_graph=True)[0]
    grad_xx = ww[..., 0]
    grad_xy = ww[..., 1]
    ww = torch.autograd.grad(grad_xx / sz, x, torch.ones_like(grad_x), create_graph=True)[0]
    grad_xxx = ww[..., 0]
    grad_xxy = ww[..., 1]
    ww = torch.autograd.grad(grad_xxx / sz, x, torch.ones_like(grad_x), create_graph=True)[0]
    grad_xxxx = ww[..., 0]
    grad_xxxy = ww[..., 1]
    ww = torch.autograd.grad(grad_xxxx / sz, x, torch.ones_like(grad_x), create_graph=True)[0]
    grad_xxxxx = ww[..., 0]
    grad_xxxxy = ww[..., 1]
    ww = torch.autograd.grad(grad_xxxxx / sz, x, torch.ones_like(grad_x), create_graph=True)[0]
    grad_xxxxxx = ww[..., 0]
    grad_xxxxxy = ww[..., 1]
    ww = torch.autograd.grad(grad_xxxxxx / sz, x, torch.ones_like(grad_x), create_graph=True)[0]
    grad_xxxxxxx = ww[..., 0]
    grad_xxxxxxy = ww[..., 1]
    print((grad_x).max(), (grad_xx).max(), (grad_xxx).max(), (grad_xxxx).max(), (grad_xxxxx).max(), (grad_xxxxxx).max(), (grad_xxxxxxx).max())
    res = torch.cat([(y.squeeze(-1)), (grad_x), (grad_xx), (grad_xxx), (grad_xxxx), (grad_xxxxx), (grad_xxxxxx), (grad_xxxxxxx)]).unsqueeze(-1)
    res2 = torch.cat([(grad_y), (grad_xy), (grad_xxy), (grad_xxxy), (grad_xxxxy), (grad_xxxxxy), (grad_xxxxxxy)]).unsqueeze(-1)
    res = torch.cat((res, res2), dim=0)
    return res

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status




