import torch
import matplotlib.pyplot as plt

from bindsnet.encoding.encodings import bernoulli_RBF, poisson_IO, IO_Current2spikes
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IO_Input
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_weights
from bindsnet.learning import MSTDP, PostPre, Hebbian
from bindsnet.utils import Error2IO_Current
from bindsnet.encoding import poisson, bernoulli

time = 1000
network = Network(dt=1)
# GR_Movement_layer = Input(n=100)
GR_Joint_layer = Input(n=100, traces=True)
PK = LIFNodes(n=8, traces=True)
PK_Anti = LIFNodes(n=8, traces=True)
IO = IO_Input(n=8)
IO_Anti = IO_Input(n=8)
DCN = LIFNodes(n=100, thresh=-57, traces=True)
DCN_Anti = LIFNodes(n=100, thresh=-57, trace=True)

# 输入motor相关
Parallelfiber = Connection(
    source=GR_Joint_layer,
    target=PK,
    wmin=0,
    wmax=10,
    update_rule = Hebbian ,   # 此处可替换为自己写的LTP
    nu=0.1,
    w=0.1+torch.zeros(GR_Joint_layer.n, PK.n)
)

# 输入 joint 相关
Parallelfiber_Anti = Connection(
    source=GR_Joint_layer,
    target=PK_Anti,
    wmin=0,
    wmax=10,
    nu = 0.1,
    update_rule=Hebbian,  # 此处同样替换为自己写的LTP
    w=0.1+torch.zeros(GR_Joint_layer.n, PK_Anti.n)
)

Climbingfiber = Connection(
    source=IO,
    target=PK,
    wmax=0,
    #update_rule=PostPre,  # 此处替换为自己写的LTD
    w=-0.1*torch.ones(IO.n, PK_Anti.n)
)

Climbingfiber_Anti = Connection(
    source=IO_Anti,
    target=PK_Anti,
    wmax=0,
    #update_rule=PostPre,  # 此处同样替换为自己写的LTD
    w=-0.1*torch.ones(IO.n, PK_Anti.n)
)

PK_DCN = Connection(
    source=PK,
    target=DCN,
    w_max=0,
    w=-0.1*torch.ones(PK.n, DCN.n)
)

PK_DCN_Anti = Connection(
    source = PK_Anti,
    target = DCN_Anti,
    w_max = 0,
    w = -0.1*torch.ones(PK_Anti.n, DCN_Anti.n)
)

network.add_layer(layer=GR_Joint_layer, name="GR_Joint_layer")
network.add_layer(layer=PK, name="PK")
network.add_layer(layer=PK_Anti, name="PK_Anti")
network.add_layer(layer=IO, name="IO")
network.add_layer(layer=IO_Anti, name="IO_Anti")
network.add_layer(layer=DCN, name="DCN")
network.add_layer(layer=DCN_Anti, name="DCN_Anti")
network.add_connection(connection=Parallelfiber, source="GR_Joint_layer", target="PK")
network.add_connection(connection=Parallelfiber_Anti, source="GR_Joint_layer", target="PK_Anti")
network.add_connection(connection=Climbingfiber, source="IO", target="PK")
network.add_connection(connection=Climbingfiber_Anti, source="IO_Anti", target="PK_Anti")
network.add_connection(connection=PK_DCN, source="PK", target="DCN")
network.add_connection(connection=PK_DCN_Anti, source="PK_Anti", target="DCN_Anti")

GR_monitor = Monitor(
    obj=GR_Joint_layer,
    state_vars=("s"),
    time = time
)
PK_monitor = Monitor(
    obj=PK,
    state_vars=("s","v"),
    time = time
)
PK_Anti_monitor = Monitor(
    obj=PK_Anti,
    state_vars=("s","v"),
    time = time
)

IO_monitor = Monitor(
    obj=IO,
    state_vars=("s"),
    time = time
)
DCN_monitor = Monitor(
    obj=DCN,
    state_vars=("s","v"),
    time = time,
)

DCN_Anti_monitor = Monitor(
    obj= DCN_Anti,
    state_vars=("s", "v"),
    time = time
)
network.add_monitor(monitor=GR_monitor, name="GR")
network.add_monitor(monitor=PK_monitor, name="PK")
network.add_monitor(monitor=PK_Anti_monitor, name="PK_Anti")
network.add_monitor(monitor=IO_monitor, name="IO")
network.add_monitor(monitor=DCN_monitor, name="DCN")
network.add_monitor(monitor=DCN_Anti_monitor, name="DCN_Anti")

#单次网络输入测试
time = 5
dt = 0.2

# 输入信号编码测试
neu_GR = 100
a = torch.Tensor([0.5])
data_Joint = bernoulli_RBF(a, neu_GR, time, dt)                    # Input_DATA, neural_num, time, dt

# 监督信号编码测试
neu_IO = 32
supervise = torch.Tensor([0.1])
## 根据监督信号生成电流值 相同监督相同电流
Curr, Curr_Anti = Error2IO_Current(supervise)
IO_Input = IO_Current2spikes(Curr, neu_IO, time, dt)               # Supervise_DATA, neural_num, time, dt
IO_Anti_Input = IO_Current2spikes(Curr_Anti, neu_IO, time, dt)


inputs = {"GR_Joint_layer": data_Joint,
          "IO": IO_Input,
          "IO_Anti": IO_Anti_Input
          }


for i in range(10):
    network.run(inputs=inputs, time=time)

    spikes = {
        "GR": GR_monitor.get("s"),
  #  "PK":PK_monitor.get("s"),
  #  "PK_Anti":PK_Anti_monitor.get("s"),
        "IO":IO_monitor.get("s"),
  #  "DCN":DCN_monitor.get("s"),
   # "DCN_Anti":DCN_Anti_monitor.get("s")
    }
    spikes2 = {
  #  "GR": GR_monitor.get("s")
        "PK":PK_monitor.get("v"),
  #  "PK_Anti":PK_Anti_monitor.get("s"),
   # "IO":IO_monitor.get("s"),
  #  "DCN":DCN_monitor.get("s"),
   # "DCN_Anti":DCN_Anti_monitor.get("s")
    }

weight = Parallelfiber.w
plot_weights(weights=weight)
voltages = {"DCN": DCN_monitor.get("v")}
plt.ioff()
plot_spikes(spikes)
plot_voltages(spikes2, plot_type="line")
plot_voltages(voltages, plot_type="line")
plt.show()






