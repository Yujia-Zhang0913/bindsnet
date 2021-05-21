from bindsnet.encoding import bernoulli_pre, bernoulli_RBF
import torch
import numpy as np
from bindsnet.analysis.pipeline_analysis import MatplotlibAnalyzer

# for i in range(10):
#     rate = bernoulli_pre(i*0.1,num_group=10)
#     print(rate)
#     print(bernoulli_RBF(datum=rate,neural_num=20,num_group=10,time=10))
torch.set_printoptions(threshold=np.inf)
net = torch.load("network.pt")
w = net.connections[('GR_Joint_layer', 'PK')].w
print(w)

p = MatplotlibAnalyzer()
p.plot_reward(show_list={"tout":[0,1,2,3,4],"error":[1,2,3,4,5],"ll":[9,10,11,12]})
p.finalize_step()
