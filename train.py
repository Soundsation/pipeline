import math
import torch
import torch.distributed as dist
import logging
import os
from collections import OrderedDict


def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:  # real logger
        os.makedirs(logging_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


class Block(torch.nn.Module):
    def __init__(self, ln=True):
        super().__init__()
        self.ln = ln
        self.stratified = False

    def forward(self, model, x, **kwargs):
        b = x.size(0)
        if self.ln:
            if self.stratified:
                # Stratified sampling of normals
                quantiles = torch.linspace(0, 1, b + 1).to(x.device)
                z = quantiles[:-1] + torch.rand((b,)).to(x.device) / b
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                t = torch.sigmoid(z)
            else:
                nt = torch.randn((b,)).to(x.device)
                t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)

        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        zt, t = zt.to(x.dtype), t.to(x.dtype)
        vtheta = model(x=zt, t=t, **kwargs)

        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]

        return batchwise_mse.mean(), {"batchwise_loss": ttloss}

    @torch.no_grad()
    def sample(self, model, z, conds, null_cond=None, sample_steps=50, cfg=2.0, **kwargs):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = model(x=z, t=t, **conds)
            if null_cond is not None:
                vu = model(x=z, t=t, **null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)

        return images

    @torch.no_grad()
    def sample_with_xps(self, model, z, conds, null_cond=None, sample_steps=50, cfg=2.0, **kwargs):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = model(x=z, t=t, **conds)
            if null_cond is not None:
                vu = model(x=z, t=t, **null_cond)
                vc = vu + cfg * (vc - vu)

            x = z - i * dt * vc
            z = z - dt * vc
            images.append(x)

        return images
