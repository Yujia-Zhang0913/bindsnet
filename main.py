from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any,Optional,Callable

import gym
import numpy as np
import torch
from bindsnet.datasets.preprocess import subsample, gray_scale, binary_image, crop
from bindsnet.encoding import Encoder, NullEncoder
import bindsnet.environment as env
import matplotlib.pyplot as plt

from bindsnet.encoding.encodings import bernoulli_RBF, poisson_IO, IO_Current2spikes, Decode_Output
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes,LIF_Train
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_weights
from bindsnet.learning import STDP,IO_Record,PostPre,NoOp
from bindsnet.utils import Error2IO_Current
from bindsnet.encoding import poisson, bernoulli

My_env = env.WholeEnvironment_sim(50, 1, 2)
My_env.step()
My_env.step()