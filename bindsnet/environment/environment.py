from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any,Optional,Callable

import gym
import numpy as np
import torch
from ..datasets.preprocess import subsample, gray_scale, binary_image, crop
from ..encoding import Encoder, NullEncoder
import torch
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
import matlab.engine

class Environment(ABC):
    # language=rst
    """
    Abstract environment class.
    """

    @abstractmethod
    def step(self, a: int) -> Tuple[Any, ...]:
        # language=rst
        """
        Abstract method head for ``step()``.

        :param a: Integer action to take in environment.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        # language=rst
        """
        Abstract method header for ``reset()``.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        # language=rst
        """
        Abstract method header for ``render()``.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        # language=rst
        """
        Abstract method header for ``close()``.
        """
        pass

    @abstractmethod
    def preprocess(self) -> None:
        # language=rst
        """
        Abstract method header for ``preprocess()``.
        """
        pass


class GymEnvironment(Environment):
    # language=rst
    """
    A wrapper around the OpenAI ``gym`` environments.
    """

    def __init__(self, name: str, encoder: Encoder = NullEncoder(), **kwargs) -> None:
        # language=rst
        """
        Initializes the environment wrapper. This class makes the
        assumption that the OpenAI ``gym`` environment will provide an image
        of format HxW or CxHxW as an observation (we will add the C
        dimension to HxW tensors) or a 1D observation in which case no
        dimensions will be added.

        :param name: The name of an OpenAI ``gym`` environment.
        :param encoder: Function to encode observations into spike trains.

        Keyword arguments:

        :param float max_prob: Maximum spiking probability.
        :param bool clip_rewards: Whether or not to use ``np.sign`` of rewards.

        :param int history: Number of observations to keep track of.
        :param int delta: Step size to save observations in history.
        :param bool add_channel_dim: Allows for the adding of the channel dimension in
            2D inputs.
        """
        self.name = name
        self.env = gym.make(name)
        self.action_space = self.env.action_space

        self.encoder = encoder

        # Keyword arguments.
        self.max_prob = kwargs.get("max_prob", 1.0)
        self.clip_rewards = kwargs.get("clip_rewards", True)

        self.history_length = kwargs.get("history_length", None)
        self.delta = kwargs.get("delta", 1)
        self.add_channel_dim = kwargs.get("add_channel_dim", True)

        if self.history_length is not None and self.delta is not None:
            self.history = {
                i: torch.Tensor()
                for i in range(1, self.history_length * self.delta + 1, self.delta)
            }
        else:
            self.history = {}

        self.episode_step_count = 0
        self.history_index = 1

        self.obs = None
        self.reward = None

        assert (
            0.0 < self.max_prob <= 1.0
        ), "Maximum spiking probability must be in (0, 1]."

    def step(self, a: int) -> Tuple[torch.Tensor, float, bool, Dict[Any, Any]]:
        # language=rst
        """
        Wrapper around the OpenAI ``gym`` environment ``step()`` function.

        :param a: Action to take in the environment.
        :return: Observation, reward, done flag, and information dictionary.
        """
        # Call gym's environment step function.
        # Call external engine step function, take action
        self.obs, self.reward, self.done, info = self.env.step(a)

        if self.clip_rewards:
            self.reward = np.sign(self.reward)

        self.preprocess()

        # Add the raw observation from the gym environment into the info
        # for debugging and display.
        info["gym_obs"] = self.obs

        # Store frame of history and encode the inputs.
        if len(self.history) > 0:
            self.update_history()
            self.update_index()
            # Add the delta observation into the info for debugging and display.
            info["delta_obs"] = self.obs

        # The new standard for images is BxTxCxHxW.
        # The gym environment doesn't follow exactly the same protocol.
        #
        # 1D observations will be left as is before the encoder and will become BxTxL.
        # 2D observations are assumed to be mono images will become BxTx1xHxW
        # 3D observations will become BxTxCxHxW
        if self.obs.dim() == 2 and self.add_channel_dim:
            # We want CxHxW, it is currently HxW.
            self.obs = self.obs.unsqueeze(0)

        # The encoder will add time - now Tx...
        if self.encoder is not None:
            self.obs = self.encoder(self.obs)

        # Add the batch - now BxTx...
        self.obs = self.obs.unsqueeze(0)

        self.episode_step_count += 1

        # Return converted observations and other information.
        return self.obs, self.reward, self.done, info

    def reset(self) -> torch.Tensor:
        # language=rst
        """
        Wrapper around the OpenAI ``gym`` environment ``reset()`` function.

        :return: Observation from the environment.
        """
        # Call gym's environment reset function.
        self.obs = self.env.reset()
        self.preprocess()

        self.history = {i: torch.Tensor() for i in self.history}

        self.episode_step_count = 0

        return self.obs

    def render(self) -> None:
        # language=rst
        """
        Wrapper around the OpenAI ``gym`` environment ``render()`` function.
        """
        self.env.render()

    def close(self) -> None:
        # language=rst
        """
        Wrapper around the OpenAI ``gym`` environment ``close()`` function.
        """
        self.env.close()

    def preprocess(self) -> None:
        # language=rst
        """
        Pre-processing step for an observation from a ``gym`` environment.
        """
        if self.name == "SpaceInvaders-v0":
            self.obs = subsample(gray_scale(self.obs), 84, 110)
            self.obs = self.obs[26:104, :]
            self.obs = binary_image(self.obs)
        elif self.name == "BreakoutDeterministic-v4":
            self.obs = subsample(gray_scale(crop(self.obs, 34, 194, 0, 160)), 80, 80)
            self.obs = binary_image(self.obs)
        else:  # Default pre-processing step.
            pass

        self.obs = torch.from_numpy(self.obs).float()

    def update_history(self) -> None:
        # language=rst
        """
        Updates the observations inside history by performing subtraction from most
        recent observation and the sum of previous observations. If there are not enough
        observations to take a difference from, simply store the observation without any
        differencing.
        """
        # Recording initial observations.
        if self.episode_step_count < len(self.history) * self.delta:
            # Store observation based on delta value.
            if self.episode_step_count % self.delta == 0:
                self.history[self.history_index] = self.obs
        else:
            # Take difference between stored frames and current frame.
            temp = torch.clamp(self.obs - sum(self.history.values()), 0, 1)

            # Store observation based on delta value.
            if self.episode_step_count % self.delta == 0:
                self.history[self.history_index] = self.obs

            assert (
                len(self.history) == self.history_length
            ), "History size is out of bounds"
            self.obs = temp

    def update_index(self) -> None:
        # language=rst
        """
        Updates the index to keep track of history. For example: ``history = 4``,
        ``delta = 3`` will produce ``self.history = {1, 4, 7, 10}`` and
        ``self.history_index`` will be updated according to ``self.delta`` and will wrap
        around the history dictionary.
        """
        if self.episode_step_count % self.delta == 0:
            if self.history_index != max(self.history.keys()):
                self.history_index += self.delta
            else:
                # Wrap around the history.
                self.history_index = (self.history_index % max(self.history.keys())) + 1


class NetworkEnvironment_sim(Environment):
    # language=rst
    """
    A wrapper around the OpenAI ``gym`` environments.
    """

    def __init__(self,
                 encoding_time:int,
                 Traj_command:float,
                 supervise:float,
                 **kwargs) -> None:
        # language=rst
        """
           :param encoding_time: ensure the same time of the environments
        """
        self.num_GR = 100
        self.num_PK = 32
        self.num_IO = 32
        self.num_DCN = 100

        self.Traj_result = Traj_command
        self.supervise = supervise

        self.encoding_time = encoding_time
        self.dt = 0.1
        self.Info_network = {"Input":0, "Output": 0, "Output_Anti": 0}
        self.inputs = {}
        self.network = {}
        self.Network_str()

    def Network_str(self):
        # language=rst
        """
            Build the structure of the network
        """
        self.network = Network(self.dt)

        GR_Joint_layer = Input(n=self.num_GR, traces=True)
        PK = LIF_Train(n=self.num_PK, traces=True, refrac=0, thresh=-40)
        PK_Anti = LIF_Train(n=self.num_PK, traces=True, refrac=0, thresh=-40)
        IO = Input(n=self.num_IO, traces=True)
        IO_Anti = Input(n=self.num_IO, traces=True)
        self.DCN = LIFNodes(n=self.num_DCN, thresh=-57, traces=True)
        self.DCN_Anti = LIFNodes(n=self.num_DCN, thresh=-57, trace=True)

        # 输入motor相关
        Parallelfiber = Connection(
            source=GR_Joint_layer,
            target=PK,
            wmin=0,
            wmax=10,
            update_rule=STDP,
            nu=0.1,
            w=0.1 + torch.zeros(GR_Joint_layer.n, PK.n),
        )

        # 输入 joint 相关
        Parallelfiber_Anti = Connection(
            source=GR_Joint_layer,
            target=PK_Anti,
            wmin=0,
            wmax=10,
            nu=0.1,
            update_rule=STDP,
            w=0.1 + torch.zeros(GR_Joint_layer.n, PK_Anti.n)
        )

        Climbingfiber = Connection(
            source=IO,
            target=PK,
            update_rule=IO_Record,
        )

        Climbingfiber_Anti = Connection(
            source=IO_Anti,
            target=PK_Anti,
            update_rule=IO_Record,
        )

        PK_DCN = Connection(
            source=PK,
            target=self.DCN,
            w=-0.1 * torch.ones(PK.n, self.DCN.n)
        )

        PK_DCN_Anti = Connection(
            source=PK_Anti,
            target=self.DCN_Anti,
            w=-0.1 * torch.ones(PK_Anti.n, self.DCN_Anti.n)
        )

        GR_DCN = Connection(
            source=GR_Joint_layer,
            target=self.DCN,
            w=0.1 * torch.ones(GR_Joint_layer.n, self.DCN.n)
        )

        GR_DCN_Anti = Connection(
            source=GR_Joint_layer,
            target=self.DCN_Anti,
            w=0.1 * torch.ones(GR_Joint_layer.n, self.DCN_Anti.n)
        )

        self.network.add_layer(layer=GR_Joint_layer, name="GR_Joint_layer")
        self.network.add_layer(layer=PK, name="PK")
        self.network.add_layer(layer=PK_Anti, name="PK_Anti")
        self.network.add_layer(layer=IO, name="IO")
        self.network.add_layer(layer=IO_Anti, name="IO_Anti")
        self.network.add_layer(layer=self.DCN, name="DCN")
        self.network.add_layer(layer=self.DCN_Anti, name="DCN_Anti")
        self.network.add_connection(connection=Climbingfiber, source="IO", target="PK")
        self.network.add_connection(connection=Climbingfiber_Anti, source="IO_Anti", target="PK_Anti")
        self.network.add_connection(connection=Parallelfiber, source="GR_Joint_layer", target="PK")
        self.network.add_connection(connection=Parallelfiber_Anti, source="GR_Joint_layer", target="PK_Anti")

        self.network.add_connection(connection=PK_DCN, source="PK", target="DCN")
        self.network.add_connection(connection=PK_DCN_Anti, source="PK_Anti", target="DCN_Anti")

        self.network.add_connection(connection=GR_DCN, source="GR_Joint_layer", target="DCN")
        self.network.add_connection(connection=GR_DCN_Anti, source="GR_Joint_layer", target="DCN_Anti")

    def encoding(self) -> None:
        # language=rst
        """
        """
        data_Joint = bernoulli_RBF(self.Traj_result, self.num_GR, self.encoding_time, self.dt)  # Input_DATA, neural_num, time, dt

        self.supervise = torch.Tensor( [self.Info_network["Input"] ] )
        Curr, Curr_Anti = Error2IO_Current(self.supervise)
        IO_Input = IO_Current2spikes(Curr, self.num_IO, self.encoding_time, self.dt)  # Supervise_DATA, neural_num, time, dt
        IO_Anti_Input = IO_Current2spikes(Curr_Anti, self.num_IO, self.encoding_time, self.dt)

        self.inputs = {
            "IO": IO_Input,
            "GR_Joint_layer": data_Joint,
            "IO_Anti": IO_Anti_Input
        }

    def step(self):

        DCN_monitor = Monitor(
            obj=self.DCN,
            state_vars=("s"),
            time=self.encoding_time,
        )
        DCN_Anti_monitor = Monitor(
            obj=self.DCN_Anti,
            state_vars=("s"),
            time=self.encoding_time
        )
        self.network.add_monitor(monitor=DCN_monitor, name="DCN")
        self.network.add_monitor(monitor=DCN_Anti_monitor, name="DCN_Anti")

        print(self.network)
        print(self.inputs)
        self.network.run(inputs=self.inputs, time=self.encoding_time)

        DCN = DCN_monitor.get("s")
        Output = Decode_Output(DCN, self.num_DCN, self.encoding_time, self.dt, 10.0)
        print(DCN)

        DCN_Anti = DCN_Anti_monitor.get("s")
        Output_Anti = Decode_Output(DCN_Anti, self.num_DCN, self.encoding_time, self.dt, 10.0)

        self.Info_network["Output"] = Output
        self.Info_network["Output_Anti"] = Output_Anti

        print("---- Output of DCN neural ----")
        print("The out put of the network is ")
        print(Output)
        print(Output_Anti)


    def reset(self) -> None:
        # language=rst
        """
        Abstract method header for ``reset()``.
        """
        pass

    def render(self) -> None:
        # language=rst
        """
        Abstract method header for ``render()``.
        """
        pass

    def close(self) -> None:
        # language=rst
        """
        Abstract method header for ``close()``.
        """
        pass

    def preprocess(self) -> None:
        # language=rst
        """
        Abstract method header for ``preprocess()``.
        """
        pass


class WholeEnvironment_sim(Environment):
    # language=rst
    """
    A wrapper around the OpenAI ``gym`` environments.
    """

    def __init__(self,
                 encoding_time:float,
                 goal:float,
                 steps:int,
                 **kwargs) -> None:
        # language=rst
        """
           :param encoding_time: ensure the same time of the environments
        """
        self.goal = goal
        self.Info_Muscle = {"Muscle":0.1, "Command":0, "Command_Anti":0}
        self.steps = steps

        self.Traj_planner()
        self.Traj_Info = torch.Tensor([0.1, 0.1, 0.1, 0.1]) #TODO
        self.Supervisor_Info = torch.Tensor([0.1, 0.1, 0.1, 0.1])##TODO

        self.Network_env = NetworkEnvironment_sim(encoding_time, 0, 0)
        self.Network_env.Network_str()



    def step(self):
        """
        """
        #for steps in range(10):
        self.Network_env.Traj_result = self.Traj_Info[self.steps]
        self.Network_env.Info_network['Input'] = self.Supervisor_Info[self.steps]   #Simulation on Sender
        self.Network_env.encoding()
        self.Network_env.step()
        self.Receiver()
        #self.Muscle_env.step()
        #Monitor

    def Traj_planner(self):
        # language=rst
        """
        """
        # TODO how to calculate the planner
           #[0,1]

    def Sender(self):
        # language=rst
        """
        """
        self.Network_env.Info_network['Input'] = self.Info_muscle['Muscle']


    def Receiver(self):
        """
        """
        self.Info_Muscle["Command"] = self.Network_env.Info_network["Output"]
        self.Info_Muscle["Command_Anti"] = self.Network_env.Info_network["Output_Anti"]
        print("Info_Muscle")
        print(self.Info_Muscle)

    def reset(self) -> None:
        # language=rst
        """
        Abstract method header for ``reset()``.
        """
        pass

    def render(self) -> None:
        # language=rst
        """
        Abstract method header for ``render()``.
        """
        pass

    def close(self) -> None:
        # language=rst
        """
        Abstract method header for ``close()``.
        """
        pass

    def preprocess(self) -> None:
        # language=rst
        """
        Abstract method header for ``preprocess()``.
        """
        pass


class MuscleEnvironment:
    # language=rst
    """
    A wrapper around the OpenAI ``gym`` environments.
    """

    def __init__(self,
                 step_min:float=0.1,
                 **kwargs) -> None:
        # language=rst
        """
           :param n_mat_step: one step network run ,n_mat_step eng run
           :param MATLABSTEPTIME: eng time per step

        """
        matlab.engine.start_matlab()  # start the topic from matlab
        self.eng = matlab.engine.connect_matlab()  # connect the topic
        assert (self.eng is not None), "Failed to connect with  matlab"  # if not, exit
        print("Successfully connected!")
        self.Info_muscle = {"pos":0,"vel":0}
        self.sim_name = None
        self.env_step_count = 0
        self.step_min = step_min

    def start(self,sim_name:str='actuator'):
        # language=rst
        """
            start the Simulink
           :param sim_name: the name of simulink file prepared to run

        """
        self.sim_name = sim_name
        self.eng.load_system(sim_name)  # load the model
        print("-"*10+"Simulink start"+"-"*10)

    def step(self,
             record_list:list,
             command_list:list) -> None:
        # language=rst
        """
           simulate a single step and record output from simulink
           :param record_list: names of the variable in eng you want to record into Info_muscle,every name must be a string type

        """
        # Send command to eng
        self.Send_control(command_list)
        # Call eng environment to run for n_mat_step
        if self.env_step_count is 0:
            self.eng.set_param(self.sim_name, "SimulationCommand", "start", nargout=0)
            self.eng.set_param(self.sim_name, "SimulationCommand", "step", nargout=0)
            self.eng.set_param(self.sim_name, "SimulationCommand", "pause", nargout=0)
            self.env_step_count += 1
        t_now = self.eng.workspace['tout'][self.env_step_count]
        t_now = self.eng.single(t_now)
        t_next = self.eng.workspace['tout'][self.env_step_count+1]
        t_next = self.eng.single(t_next)
        while t_next-t_now<self.step_min:
            self.eng.set_param(self.sim_name, "SimulationCommand", "start", nargout=0)
            self.eng.set_param(self.sim_name, "SimulationCommand", "step", nargout=0)
            self.eng.set_param(self.sim_name, "SimulationCommand", "pause", nargout=0)
            self.env_step_count += 1
            t_next = self.eng.workspace['tout'][self.env_step_count]
            t_next = self.eng.single(t_next)
        # load data from eng to Info

        self.Rec_eng_Info(record_list)

    def Rec_eng_Info(self,para_list:list)->None:
        # TODO due to the data structure of matlab workspace
        # TODO deal with non-exist para_name
        # language=rst
        """
            load desired eng variable from workspace to "Info_muscle"
           :param para_list: name list of the eng variable you want to record
        """

        if len(para_list) is 0:
            print("You want to record empty!")
        else:
            for l in para_list:
                assert isinstance(l,str),"Invaild record key! Key must be string type"
                self.Info_muscle[l] = self.eng.single(self.eng.workspace[l][self.env_step_count][1])

    def Send_control(self,command_list:list):
        # TODO due to the data structure of matlab workspace
        # language=rst
        """
            load desired eng variable from workspace to "Info_muscle"
           :param command_list: name list of the variable you want to send from Info_muscle to eng
        """

        if len(command_list) is 0:
            print("You want to record empty!")
        else:
            for c in command_list:
                assert isinstance(c,str),"Invaild command key! Key must be string type"
                assert self.Info_muscle.get(c) is not None,"No such key in Info_muscle"
                self.eng.workspace[c] = self.Info_muscle[c]

    def reset(self) -> None:
        # language=rst
        """
        reset eng and clear the Info dictionary
        """
        self.eng.run("Para_in.m",nargout=0)  # load the data again
        self.Info_muscle = {"pos":0,"vel":0}
        self.env_step_count = 0
    def close(self) -> None:
        # language=rst
        """
        Wrapper around the OpenAI ``gym`` environment ``close()`` function.
        """
        assert self.sim_name is not None,"No simulink is running!"
        self.eng.set_param(self.sim_name, "SimulationCommand", "stop", nargout=0)
        self.eng.clear(nargout=0)
    def render(self) -> None:
        # language=rst
        """
        Abstract method header for ``render()``.
        """
        pass

    @abstractmethod
    def preprocess(self) -> None:
        # language=rst
        """
        Abstract method header for ``preprocess()``.
        """
        pass
