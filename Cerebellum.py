import torch
import matplotlib.pyplot as plt

from bindsnet.encoding.encodings import bernoulli_RBF, poisson_IO, IO_Current2spikes, Decode_Output
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, LIF_Train
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor, Our_Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_weights
from bindsnet.learning import STDP, IO_Record, PostPre, NoOp
from bindsnet.utils import Error2IO_Current
from bindsnet.encoding import poisson, bernoulli, bernoulli_pre
import numpy as np

time = 50
network = Network(dt=1)
# GR_Movement_layer = Input(n=100)
GR_Joint_layer = Input(n=100, traces=True)
PK = LIF_Train(n=32, traces=True, refrac=0, thresh=-40)
PK_Anti = LIF_Train(n=32, traces=True, refrac=0, thresh=-40)
IO = Input(n=32)
IO_Anti = Input(n=32, traces=True)
DCN = LIFNodes(n=100, thresh=-57, traces=True)
DCN_Anti = LIFNodes(n=100, thresh=-57, trace=True)

# 输入motor相关
Parallelfiber = Connection(
    source=GR_Joint_layer,
    target=PK,
    wmin=0,
    wmax=1,
    update_rule=STDP,
    nu=[0.1, 0.1],
    w=0.1 + torch.zeros(GR_Joint_layer.n, PK.n),
)

# 输入 joint 相关
Parallelfiber_Anti = Connection(
    source=GR_Joint_layer,
    target=PK_Anti,
    wmin=0,
    nu=[0.1, 0.1],
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
    target=DCN,
    w=-0.1 * torch.ones(PK.n, DCN.n)
)

PK_DCN_Anti = Connection(
    source=PK_Anti,
    target=DCN_Anti,
    w=-0.1 * torch.ones(PK_Anti.n, DCN_Anti.n)
)

GR_DCN = Connection(
    source=GR_Joint_layer,
    target=DCN,
    w=0.1 * torch.ones(GR_Joint_layer.n, DCN.n)
)

GR_DCN_Anti = Connection(
    source=GR_Joint_layer,
    target=DCN_Anti,
    w=0.1 * torch.ones(GR_Joint_layer.n, DCN_Anti.n)
)

network.add_layer(layer=GR_Joint_layer, name="GR_Joint_layer")
network.add_layer(layer=PK, name="PK")
network.add_layer(layer=PK_Anti, name="PK_Anti")
network.add_layer(layer=IO, name="IO")
network.add_layer(layer=IO_Anti, name="IO_Anti")
network.add_layer(layer=DCN, name="DCN")
network.add_layer(layer=DCN_Anti, name="DCN_Anti")
network.add_connection(connection=Climbingfiber, source="IO", target="PK")
network.add_connection(connection=Climbingfiber_Anti, source="IO_Anti", target="PK_Anti")
network.add_connection(connection=Parallelfiber, source="GR_Joint_layer", target="PK")
network.add_connection(connection=Parallelfiber_Anti, source="GR_Joint_layer", target="PK_Anti")

network.add_connection(connection=PK_DCN, source="PK", target="DCN")
network.add_connection(connection=PK_DCN_Anti, source="PK_Anti", target="DCN_Anti")

network.add_connection(connection=GR_DCN, source="GR_Joint_layer", target="DCN")
network.add_connection(connection=GR_DCN_Anti, source="GR_Joint_layer", target="DCN_Anti")

GR_monitor = Monitor(
    obj=GR_Joint_layer,
    state_vars=("s"),
    time=time
)
PK_monitor = Monitor(
    obj=PK,
    state_vars=("s", "v"),
    time=time
)
PK_Anti_monitor = Monitor(
    obj=PK_Anti,
    state_vars=("s", "v"),
    time=time
)

IO_monitor = Monitor(
    obj=IO,
    state_vars=("s"),
    time=time
)
IO_Anti_monitor = Monitor(
    obj=IO_Anti,
    state_vars=("s"),
    time=time
)

DCN_monitor = Monitor(
    obj=DCN,
    state_vars=("s", "v"),
    time=time,
)

DCN_Anti_monitor = Monitor(
    obj=DCN_Anti,
    state_vars=("s", "v"),
    time=time
)

IO_Our_monitor = Our_Monitor(
    obj=IO,
    state_vars=("s")
)
network.add_monitor(monitor=GR_monitor, name="GR")
network.add_monitor(monitor=PK_monitor, name="PK")
network.add_monitor(monitor=PK_Anti_monitor, name="PK_Anti")
network.add_monitor(monitor=IO_monitor, name="IO")
network.add_monitor(monitor=IO_Anti_monitor, name="IO_Anti")
network.add_monitor(monitor=DCN_monitor, name="DCN")
network.add_monitor(monitor=DCN_Anti_monitor, name="DCN_Anti")
network.add_monitor(monitor=IO_Our_monitor, name="IO_Our_Monitor")

# 单次网络输入测试
encoding_time = 50
dt = 1

# 输入信号编码测试
neu_GR = 100
data = bernoulli_pre(0.5)
data_Joint = bernoulli_RBF(datum=data, neural_num=neu_GR, time=time, dt=1)  # Input_DATA, neural_num, time, dt

# 监督信号编码测试
neu_IO = 32
supervise = torch.Tensor([0])

## 根据监督信号生成电流值 相同监督相同电流
Curr, Curr_Anti = Error2IO_Current(supervise)
print("Curr: {}".format(Curr))
print("Curr_Anti: {}".format(Curr_Anti))
IO_Input = IO_Current2spikes(Curr, neu_IO, encoding_time, dt)  # Supervise_DATA, neural_num, time, dt
IO_Anti_Input = IO_Current2spikes(Curr_Anti, neu_IO, encoding_time, dt)

for i in range(10):
    print('-' * 10 + str(i) + '-' * 10)
    if i % 2:
        Curr = torch.Tensor([0.1*i])
        Curr_Anti = torch.Tensor([0.1*i])
        data = 0.1*i
        data = bernoulli_pre(data)
        data_Joint = bernoulli_RBF(datum=data, neural_num=neu_GR, time=time, dt=1)  # Input_DATA, neural_num, time, dt
    else:
        data = 0.1*i
        data = bernoulli_pre(data)
        data_Joint = bernoulli_RBF(datum=data, neural_num=neu_GR, time=time, dt=1)  # Input_DATA, neural_num, time, dt
        Curr = torch.Tensor([0.1*i])
        Curr_Anti = torch.Tensor([0.1*i])
        # 根据监督信号生成电流值 相同监督相同电流
    # Curr, Curr_Anti = Error2IO_Current(supervise)
    print("Curr: {}".format(Curr))
    print("Curr_Anti: {}".format(Curr_Anti))
    IO_Input = IO_Current2spikes(Curr, neu_IO, encoding_time, dt)  # Supervise_DATA, neural_num, time, dt
    print(IO_Input)
    IO_Anti_Input = IO_Current2spikes(Curr_Anti, neu_IO, encoding_time, dt)
    inputs = {
        "IO": IO_Input,
        "GR_Joint_layer": data_Joint,
        "IO_Anti": IO_Anti_Input
    }
    network.learning = False
    network.run(inputs=inputs, time=time)

spikes = {
    "IO": IO_Our_monitor.get("s"),
    "GR_Joint_layer": GR_monitor.get("s"),
    "DCN": DCN_monitor.get("s")
    # "DCN_Anti":DCN_Anti_monitor.get("s")
}

voltages = {
    "DCN": DCN_monitor.get("v"),
    "PK": PK_monitor.get("v"),
    "PK_Anti": PK_Anti_monitor.get("v")
}
plt.ioff()
plot_spikes(spikes)
plot_weights(network.connections["GR_Joint_layer", "PK"].w)
print("---- Output of DCN neural ----")

DCN = DCN_monitor.get("s")
Output = Decode_Output(DCN, 100, encoding_time, dt, 10.0)

DCN_Anti = DCN_Anti_monitor.get("s")
Output_Anti = Decode_Output(DCN, 100, encoding_time, dt, 10.0)

print("The out put of the network is ")
print(Output)
print(Output_Anti)

plot_voltages(voltages, plot_type="line")
plt.show()
