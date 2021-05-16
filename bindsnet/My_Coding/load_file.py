# import  torch
# from  bindsnet.analysis.plotting import plot_weights
# net = torch.load("net")
# print(net.dt)
# plot_weights(net.connections[''])
import torch
import matplotlib.pyplot as plt

from bindsnet.network import Network
from bindsnet.network.nodes import Input
from bindsnet.network.monitors import Monitor

# Build simple network.
network = Network()
network.add_layer(Input(500), name='I')
network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

# Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

# Run network simulation.
network.run(inputs={'I': spikes}, time=500)

# Look at input spiking activity.
spikes = network.monitors['I'].get('s')
plt.matshow(spikes, cmap='binary')
plt.xticks(());
plt.yticks(());
plt.xlabel('Time');
plt.ylabel('Neuron index')
plt.title('Input spiking')
plt.show()