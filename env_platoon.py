import torch
import torch.nn as nn


class EnvPlatoon:
    def __init__(self, num_steps, num_vehicles, tau, h, T, u_min, u_max, a_min, a_max, ):
