import copy
import os

import torch
from torch import distributed as dist
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """
    print(f"Setting up rank {rank} of {total_num_gpus} GPUs")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )


def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()

def generate_samples_ot_fast(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    def ot_sampler_1(x_start, steps):
        traj = []
        # t_span = torch.linspace(0, 1, steps).to(device)
        t_span = torch.linspace(-5, 5, steps).to(device)
        t_span = torch.exp(t_span) / (torch.exp(t_span) + 1)
        # print(t_span)
        t_s = t_span[0]
        t_t = t_span[0]
        for t in t_span:
            t_t = t 
            if t != t_span[0]:
                x0, x1 = model_(t_s, x_start).chunk(2, dim=1)
                t_s = t_s.to(torch.float64)
                t_t = t_t.to(torch.float64)
                if t_s <= 0.5:
                    
                    x1 = x1.reshape((-1, 3*32*32))
                    s = torch.quantile(torch.abs(x1), 0.9, dim=1, keepdim=True)
                    s = torch.clamp(s, min=1.0)
                    x1 = x1.clip(-s, s) / s 
                    x1 = x1.reshape((-1, 3, 32, 32))
                    
                    co1 = (1. - t_t) / (1. - t_s)
                    x_start = co1.to(torch.float32) * x_start + x1 * ((1. - co1).to(torch.float32))
                    # x_start += (x_start - x1) / (t_s - 1.) * (t_t - t_s)
                    # print("x_1", x1)
                    print("t_s", t_s)
                    # print("t_t", t_t)
                    # print("co1", co1.to(torch.float32))
                    # print("co2",(t_t - co1 * t_s).to(torch.float32))
                else:
                    co1 = t_t / t_s
                    x_start = co1.to(torch.float32) * x_start + x0 * ((1. - co1).to(torch.float32))
                    # x_start += (x_start - x0) / t_s * (t_t - t_s)
                    # print("x_0", x0)
                    print("t_s", t_s)
                    # print("t_t", t_t)
                    # print("co1", co1.to(torch.float32))
                    # print("co2",(1 - co1).to(torch.float32))
            t_s = t
            traj.append(x_start)
        return torch.stack(traj)
    
    def ot_sampler_2(x_start, steps):
        traj = []
        traj.append(x_start)
        # t0 = 0.005
        t0 = 0.0
        T = 1 - t0
        # t_span = torch.linspace(t0, T, steps + 1).to(device)
        t_span = torch.linspace(-5., 5., steps+1).to(device)
        t_span = torch.exp(t_span) / (torch.exp(t_span) + 1)
        def lmd_t(t):
            return torch.log(t / (1 - t))
        
        def dy_clip(x):
            x1 = x
            x1 = x1.reshape((-1, 3*32*32))
            s = torch.quantile(torch.abs(x1), 0.9, dim=1, keepdim=True)
            s = torch.clamp(s, min=1.0)
            x1 = x1.clip(-s, s) / s 
            x1 = x1.reshape((-1, 3, 32, 32))
            return x1

        Qi_2 = Qi_1 = model_(t_span[0], x_start)
        for i in range(1, steps+1):
            t_t = t_span[i]
            t_s = t_span[i-1]
            if i == 1:
                
                x1 = Qi_2.chunk(2, dim=1)[1]
                x1 = dy_clip(x1)

                co1 = (1 - t_t) / (1 - t_s)
                x_start = co1 * x_start + x1 * (1. - co1)
            else:
                t_s_1 = t_span[i-2]
                ri = (lmd_t(t_s) - lmd_t(t_s_1)) / (lmd_t(t_t) - lmd_t(t_s))
                # ri = (t_s - t_s_1) / (t_t - t_s)
                if t_s <= 0.5:
                    x1_1 = Qi_1.chunk(2, dim=1)[1]
                    x1_2 = Qi_2.chunk(2, dim=1)[1]

                    x1_1 = dy_clip(x1_1)
                    x1_2 = dy_clip(x1_2)



                    Di = x1_1 + (x1_1 - x1_2) / (2. * ri)
                    co1 = (1 - t_t) / (1 - t_s)
                    x_start = co1 * x_start + Di * (1. - co1)
                else:
                    x0_1 = Qi_1.chunk(2, dim=1)[0]
                    x0_2 = Qi_2.chunk(2, dim=1)[0]
                    Di = x0_1 + (x0_1 - x0_2) / (2. * ri)
                    co1 = t_t / t_s
                    x_start = co1 * x_start + Di * (1. - co1)
                Qi_2 = Qi_1
            Qi_1 = model_(t_t, x_start)
            traj.append(x_start)
        return torch.stack(traj)
    

    class model_ot_(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model 
        
        def forward(self, t, x, *args, **kwargs):
            # pi = torch.pi
            x = x.to(device)
            t = t.to(device)
            x0, x1 = self.model(t, x).chunk(2, dim=1)
            if t <= 0.5:
                # return -0.9998 / (0.9999 - 0.9998 * t) * (x - x1)
                return (x - x1) / (t - 1.)
            else:
                # return (x - x0) * ((0.9998)/(0.9998*t+0.0001))
                return (x - x0) / t

    node_ = NeuralODE(model_ot_(model_), solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 101, device=device),
        )
        # traj = ot_sampler_1(torch.randn(64, 3, 32, 32, device=device),11)
        # traj = ot_sampler_2(torch.randn(64, 3, 32, 32, device=device),100)
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()

def generate_samples_slow(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    def model__(t, x, args=None):
        return 2 * (1 - t) * model_(t, x)
    
    node_ = NeuralODE(model__, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()

def generate_samples_vp_fast(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    class model_vp_(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model 
    
        def forward(self, t, x, *args, **kwargs):
            pi = torch.pi
            x = x.to(device)
            t = t.to(device)
            x0, x1 = self.model(t, x).chunk(2, dim=1)
            if t <= 0.5:
                return (pi * 0.5) * ((x1 - torch.sin(pi * 0.5 * t) * x) / torch.cos(pi * 0.5 * t))
            else:
                return (pi * 0.5) * ((torch.cos(pi * 0.5 * t) * x - x0) / torch.sin(pi * 0.5 * t))

    # class model__(torch.nn.Module):
    #     def __init__(self, model):
    #         super().__init__()
    #         self.model = model

    #     def forward(self, t, x, *args, **kwargs):
    #         s = 0.008
    #         pi = torch.pi
    #         x = x.to(device)
    #         t = t.to(device)
    #         return (pi * 0.5) * ((model_(t, x) - torch.sin(pi * 0.5 * t) * x) / torch.cos(pi * 0.5 * (t/(1+s))))
    
    node_ = NeuralODE(model_vp_(model_), solver="euler", sensitivity="adjoint")

    def vp_sampler_1(x_start, steps):
        pi = torch.pi
        traj = []
        t_span = torch.linspace(0, 1, steps).to(device)
        # t_span = torch.linspace(-5, 5, steps).to(device)
        # t_span = torch.atan(torch.exp(t_span)) / (pi / 2)
        # add 0, 0.999 to t_span
        # t_span = torch.cat([torch.tensor([0.0]).to(device), t_span, torch.tensor([0.999]).to(device)], dim=0)
        t_s = t_span[0]
        t_t = t_span[0]
        for t in t_span:
            # print("start", x_start.shape)
            t_t = t 
            if t != 0:
                co1 = torch.cos(pi * 0.5 * t_t)  / torch.cos(pi * 0.5 * t_s)
                x_start = co1 * x_start + model_(t_s, x_start) * (torch.sin(pi * 0.5 * t_t) - co1 * torch.sin(pi * 0.5 * t_s))
            t_s = t
            traj.append(x_start)
        return torch.stack(traj)

    def vp_sampler_2(x_start, steps):
        traj = []
        pi = torch.pi
        t_span = torch.linspace(0, 1, steps).to(device)
        # t_span = torch.linspace(-5, 5, steps).to(device)
        # t_span = torch.atan(torch.exp(t_span)) / (pi / 2)
        # add 0, 0.999 to t_span
        # t_span = torch.cat([torch.tensor([0.0]).to(device), t_span, torch.tensor([0.999]).to(device)], dim=0)
        # print(t_span)

        t_s = t_span[0]
        t_t = t_span[0]
        for t in t_span:
            t_t = t 
            if t != t_span[0]:
                # rk method 2-order
                hi = t_t - t_s
                si = t_s + 0.5 * hi
                co11 = (torch.cos(pi * 0.5 * si)  / torch.cos(pi * 0.5 * t_s))
                ui = co11 * x_start + model_(t_s, x_start) * (torch.sin(pi * 0.5 * si) - co11 * torch.sin(pi * 0.5 * t_s))
                co1 = torch.cos(pi * 0.5 * t_t)  / torch.cos(pi * 0.5 * t_s)
                x_start = co1 * x_start + model_(si, ui) * (torch.sin(pi * 0.5 * t_t) - co1 * torch.sin(pi * 0.5 * t_s))
                # print("x", x_start[0])
                # print("t", t)
                # print("co1", co1)
                # print("eps", model(torch.cat([x_start, t_s * torch.ones(x_start.shape[0], 1, device=device)], dim=-1)))
                # print("co2", (torch.sin(pi * 0.5 * t_t) - co1 * torch.sin(pi * 0.5 * t_s)))

            t_s = t
            traj.append(x_start)
        return torch.stack(traj)

    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        # traj = vp_sampler_2(torch.randn(64, 3, 32, 32, device=device), 20)
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()



def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x
