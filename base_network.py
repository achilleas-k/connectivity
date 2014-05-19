from brian.equations import Equations
from brian.network import NeuronGroup, Network, Connection
from brian.directcontrol import PoissonGroup
from brian.monitor import StateMonitor, SpikeMonitor
from brian.units import mvolt, msecond, uamp, uvolt, second, hertz
from brian import raster_plot, defaultclock
from matplotlib.pyplot import plot, show, subplot, figure, scatter, axis
from numpy import argmax
import spikerlib

print("Preparing simulation ...")
defaultclock.dt = dt = 0.1*msecond
duration = 2*second
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

print("Setting up monitors ...")
trace_mon = StateMonitor(lif_group, "V", record=True)
input_mon = SpikeMonitor(inp_group)
spike_mon = SpikeMonitor(lif_group)
network = Network(lif_group, inp_group, inp_conn, lif_conn,
                  trace_mon, spike_mon, input_mon)
print("Running for %f seconds ..." % (duration))
network.run(duration)
print("Simulation run finished.")
print("Plotting spikes ...")
figure("Spike trains")
subplot(211)
raster_plot(input_mon)
subplot(212)
raster_plot(spike_mon)

print("Performing Gaussian convolution ...")
t, conv_spikes = spikerlib.tools.spikeconvolve(spike_mon, 5*msecond)
print("Calculating pairwise Kreuz metric ...")
# pairwise Kreuz on 1k trains -- might have to wait a while
kr_t, kr_d = spikerlib.metrics.kreuz.pairwise_mp(spike_mon.spiketimes.values(),
        0*second, duration, duration/(2*msecond))
print("Plotting calculated stuff ...")
figure("Synchrony")
subplot(211)
plot(t, conv_spikes)
axis(xmin=0, xmax=duration)
scatter(t[argmax(conv_spikes)], max(conv_spikes), s=10)
print("Synchrony peaked at t = %f s" % (t[argmax(conv_spikes)]))
subplot(212)
plot(kr_t, kr_d)
axis(xmin=0, xmax=duration)
show()

# after that, start doing some connectivity inference using the measure
# correlations
print("All done!")
