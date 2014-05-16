from brian.equations import Equations
from brian.network import NeuronGroup, Network, Connection
from brian.directcontrol import PoissonGroup
from brian.monitor import StateMonitor, SpikeMonitor
from brian.units import mvolt, msecond, uamp, uvolt, second, hertz
from brian import raster_plot
from matplotlib.pyplot import plot, show, subplot

Vrest = -60*mvolt
tau = 20*msecond
lif_eq = Equations("""
dV/dt = (Vrest-V)/tau : volt
I : volt
""")
lif_eq.prepare()
lif_group = NeuronGroup(1000, lif_eq, threshold="V>-50*mvolt", reset=Vrest,
                       refractory=2*msecond)

inp_group = PoissonGroup(100, rates=30*hertz)
inp_conn = Connection(inp_group, lif_group, weight=1*mvolt,
                      sparseness=0.05)
lif_group.V = Vrest
lif_conn = Connection(lif_group, lif_group, weight=2*mvolt,
                      sparseness=0.01)

trace_mon = StateMonitor(lif_group, "V", record=True)
input_mon = SpikeMonitor(inp_group)
spike_mon = SpikeMonitor(lif_group)
network = Network(lif_group, inp_group, inp_conn, lif_conn,
                  trace_mon, spike_mon, input_mon)
network.run(5*second)

subplot(211)
raster_plot(input_mon)
subplot(212)
raster_plot(spike_mon)
show()

# gaussian convolution on sum of spikes across time
# there's definitely some synchronisation going on
# figure out why
