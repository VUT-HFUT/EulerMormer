import os
import numpy as np
import torch
import torch.autograd as ag
from utils import L1Loss


def criterion_mag(y_hat, batch_M, texture_AC, motion_BC,  criterion1):
    # One thing deserves mentioning is that the amplified frames given in the dataset are actually perturbed Y(Y'), which I used M to represent.
    loss_y = criterion1(y_hat, batch_M)
    loss_texture_AC = criterion1(*texture_AC)  # v_c, v_a
    loss_motion_BC = criterion1(*motion_BC) # m_c, m_b

    return loss_y, loss_texture_AC, loss_motion_BC