import sys
from brian.equations import Equations
from brian.network import NeuronGroup, Network, Connection
from brian.directcontrol import PoissonGroup
from brian.monitor import StateMonitor, SpikeMonitor
from brian.units import mvolt, msecond, second, hertz
from brian import raster_plot, defaultclock
from matplotlib.pyplot import plot, show, subplot, figure, scatter, axis, title
import matplotlib.mlab as mlab
import numpy as np
import spikerlib as sl

print("Preparing simulation ...")
defaultclock.dt = dt = 0.1*msecond
duration = 2*second
w = 2*msecond  # coincidence window for npss
Vrest = -60*mvolt
Vth = -50*mvolt
tau = 20*msecond

n_ext = 100
p_ext = 0.5
p_int = 0.05
w_ext = 0.9*mvolt
r_ext = 10*hertz
w_int = 0.1*mvolt
lif_eq = Equations("""
dV/dt = (Vrest-V)/tau : volt
I : volt
""")
lif_eq.prepare()
lif_group = NeuronGroup(1000, lif_eq, threshold="V>Vth", reset=Vrest,
                       refractory=2*msecond)

inp_group = PoissonGroup(n_ext, rates=r_ext)
# external input only applied to a part of the network
n_ext_rec = 100
inp_conn = Connection(inp_group, lif_group[:n_ext_rec], weight=w_ext,
                      sparseness=p_ext, fixed=True, delay=5*msecond)
asympt_inp = n_ext*p_ext*w_ext*r_ext*tau
voltage_range = Vth-Vrest
if asympt_inp <= voltage_range:
    print("NOTICE: Network spikes unlikely to occur.")
    print("Mean input potential %f <= %f Relative threshold" % (asympt_inp, voltage_range))
#    print("Aborting!")
#    sys.exit()
lif_group.V = Vrest
lif_conn = Connection(lif_group, lif_group, weight=w_int, sparseness=p_int,
                      delay=5*msecond)

print("Setting up monitors ...")
trace_mon = StateMonitor(lif_group, "V", record=True)
input_mon = SpikeMonitor(inp_group)
spike_mon = SpikeMonitor(lif_group)
network = Network(lif_group, inp_group, inp_conn, lif_conn,
                  trace_mon, spike_mon, input_mon)
print("Running for %f seconds ..." % (duration))
network.run(duration)
print("Simulation run finished.")
if spike_mon.nspikes == 0:
    print("No spikes were fired by the network. Aborting!")
    sys.exit()
print("Performing Gaussian convolution ...")
t, conv_spikes = sl.tools.spikeconvolve(spike_mon, 5*msecond)
figure("Spike trains")
splt = subplot(311)
title("External inputs")
raster_plot(input_mon)
subplot(312)
title("Network spikes")
raster_plot(spike_mon)
subplot(313)
plot(t, conv_spikes)
axis(xmin=0, xmax=float(duration))
scatter(t[np.argmax(conv_spikes)], max(conv_spikes), s=10)
show(block=False)
print("Network synchrony peaked at t = %f s" % (t[np.argmax(conv_spikes)]))

# let's run NPSS on all neurons and see what we get
print("Calculating NPSS ...")
npss = []
max_idx = 0
trace_mon.insert_spikes(spike_mon, Vth+30*mvolt)  # not really necessary, but I like it
for v, sp in zip(trace_mon.values, spike_mon.spiketimes.itervalues()):
    if len(sp) < 2:
        npss.append([0])
        continue
    npss.append(sl.tools.npss(v, sp, Vrest, Vth, tau, w))
    if max(npss[-1]) > max(npss[max_idx]):
        max_idx = len(npss)-1

# get winner's inputs for comparison
if max_idx < n_ext_rec:
    input_matrix = inp_conn.W.todense()
    source_inps = mlab.find(input_matrix[:,max_idx])
else:
    source_inps = []
lif_matrix = lif_conn.W.todense()
source_nrns = mlab.find(lif_matrix[:,max_idx])
figure("NPSS")
subplot(211)
# scatter plot of inputs
inp_idx = 0
for src in source_inps:
    y = np.zeros(len(input_mon[src]))+inp_idx
    scatter(input_mon[src], y, c="b")
    inp_idx += 1
for src in source_nrns:
    y = np.zeros(len(spike_mon[src]))+inp_idx
    scatter(spike_mon[src], y, c="r")
    inp_idx += 1

axis(xmin=0, xmax=float(duration))
subplot(212)
plot(spike_mon[max_idx][1:], npss[max_idx], "-o")
axis(xmin=0, xmax=float(duration))
show()


#print("Calculating pairwise Kreuz metric ...")
# pairwise Kreuz on 1k trains -- might have to wait a while
#kr_t, kr_d = spikerlib.metrics.kreuz.pairwise_mp(spike_mon.spiketimes.values(),
#        0*second, duration, duration/(2*msecond))
# give it some blanks for now, until we write and debug NPSS
#subplot(212)
#plot(kr_t, kr_d)
#axis(xmin=0, xmax=float(duration))

print("All done!")
