from __future__ import print_function
from brian import *
from brian.library.synapses import *

duration = 100*ms
#tau_list = array([i*0.1*ms for i in range(1, N+1)])
tau_list = [0.2*ms, 0.2*ms, 0.6*ms]
targets = [0.1*mV, 0.5*mV, 0.6*mV]
N = 3
network = Network()
eqs = Equations("""
dV/dt = (Ia-V)/(20*ms) : volt
#Ia = alpha*(t/(0.1*ms))*exp(1-t/(1*ms)) : volt
dIa/dt = (a-Ia)/(tau_input) : volt
da/dt = -a/(tau_input) : volt
tau_input : second
""")
eqs.prepare()
nrns = NeuronGroup(N, eqs, threshold="V>20*mV", reset=0*mV, refractory=2*ms)
nrns.tau_input = tau_list

inpspikes = SpikeGeneratorGroup(1, [(0, 20*ms)])
connection = Connection(inpspikes, nrns, 'a')
synapse = SynapticEquations


vmon = StateMonitor(nrns, 'V', record=True)
inpmon = StateMonitor(nrns, 'Ia', record=True)
weight_list = [10.5*mV, 52.5*mV, 22.5*mV]
for idx in range(N):
    connection[0,idx] = weight_list[idx]
network.add(nrns, inpspikes, connection, vmon, inpmon)

network.run(duration)

colours = ["b", "r", "g", "m", "k"]
for i in range(N):
    print("w: %f mV, p: %f mV" % (weight_list[i]*1000, max(vmon[i])*1000))
    plot(vmon.times, vmon[i], label=tau_list[i], color=colours[i])
    plot([0*second, duration], [targets[i]]*2, colours[i]+"--")
axis(ymax=float(max(targets))*1.1)
legend(loc="best")
show()
