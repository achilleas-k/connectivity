from __future__ import print_function
from brian import *
from brian.library.synapses import *

N = 10
duration = 100*ms
#tau_list = array([i*0.1*ms for i in range(1, N+1)])
tau_list = repeat([0.2*ms, 0.6*ms], N/2)
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
weight_list = [((idx+1) % (N/2))*mV*10 for idx in range(N)]
for idx in range(N):
    weight = weight_list[idx]
    connection[0,idx] = weight
network.add(nrns, inpspikes, connection, vmon, inpmon)

network.run(duration)

for i in range(N):
    print("w: %f mV, p: %f mV" % (weight_list[i]*1000, max(vmon[i])*1000))
    colour = "r" if (i >= N/2) else "b"
    plot(vmon.times, vmon[i], label=tau_list[i], color=colour)
plot([0*second, duration], [0.1*mV]*2, "b--")
plot([0*second, duration], [0.6*mV]*2, "r--")
legend(loc="best")
show()
