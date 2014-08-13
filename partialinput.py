from __future__ import print_function
from __future__ import division
from brian import *
import spikerlib as sl


def geninputsignal(*spikemons):
    """
    Return a convolved version of the combination of all provided spike trains.

    Collects all spike trains from the given monitors and convolves them
    with an exponential kernel to generate a signal waveform that describes
    the entirety of the input population.

    Arguments: Any number of spike monitors
    """
    kwidth = 10*tau
    nbins = int(duration/dt)
    binnedcounts = zeros(nbins)
    for monitor in spikemons:
        for st in monitor.spiketimes.itervalues():
            binnedcounts += sl.tools.times_to_bin(st, dt, duration)
    kernel = exp(-arange(0*second, kwidth, dt)/tau)
    signal = convolve(binnedcounts, kernel)
    return signal

def calcslopes(vmon, spikemon):
    """
    Calculate slopes (non normalised, for now)

    Returns average slopes (per cell) and individual slope values (flattened)
    TODO: Calculate normalised slopes
    """
    allslopes = []
    avgslopes = []
    for idx in vmon.recordindex.iterkeys():
        if len(spikemon[idx]) == 0:
            continue
        spikeidx = array(spikemon[idx]/dt).astype('int')
        slopestart = spikeidx-int(w/dt)
        slopes = (Vth-vmon[idx][slopestart])/w
        allslopes.extend(slopes)
        avgslopes.append(mean(slopes))
    return avgslopes, allslopes

def collectinputs(idx, group, *connections):
    """
    Return the indices of all cells that connect to the given neuron `idx`.

    Collects the indices of all neurons that provide input for neuron `idx`
    in the neuron group `group`. Any number of Connection objects may
    be provided. The function returns a list of lists, where each list
    represents the indices of neurons that drive the target neuron from a
    specific source. The order of the sources is the same as they appear in the
    argument list of the function for `*connections`.
    """
    inputs = []
    for conn in connections:
        if conn.target is group:
            inputs.append(conn.W.coli[idx].tolist())
        else:
            inputs.append([])
    return inputs


def printstats(vmon, spikemon):
    """
    Print spiking stats
    """
    spiketrains = spikemon.spiketimes.values()
    spikecounts = [len(sp) for sp in spiketrains]
    spikerates = [sp/duration for sp in spikecounts]
    avgrate = mean(spikerates)
    xcorrs = sl.metrics.corrcoef.corrcoef_spiketrains(spiketrains)
    print("Spike rates: ")
    for idx, sr in enumerate(spikerates):
        print("%3i:\t%0.2f Hz" % (idx, sr))
    print("Avg:\t%0.2f Hz" % (avgrate))
    print("\nSpike train correlations")
    print("\t"+"\t".join("%4i" % i for i in range(len(xcorrs))))
    for idx, corr in enumerate(xcorrs):
        print(str(idx)+"\t"+"\t".join("%.2f" % c for c in corr))

print("Preparing simulation ...")
network = Network()
defaultclock.dt = dt = 0.1*ms
duration = 1*second
w = 2*ms
Vrest = 0*mV
Vth = 20*mV
tau = 20*ms
Nnrns = 10
Ningroups = 5
Nin = 100000
fin = 1*Hz
Sin = 0.2
sigma = 0*ms
weight = 0.1*mV
tau_syn = 0.2*ms
Nconn = int(0.01*Nin)  # number of connections each cell receives from each group
#lifeq_exc = Equations("""
#                      dV/dt = (a-Vrest-V)/tau : volt
#                      da/dt = (gIn-a)/tau_syn : volt
#                      dgIn/dt = -gIn/tau_syn : volt
#                      """)
lifeq_exc = Equations("dV/dt = (Vrest-V)/tau : volt")
lifeq_exc.prepare()
nrngroup = NeuronGroup(Nnrns, lifeq_exc, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)
nrngroup.V = Vrest
network.add(nrngroup)
print("Setting up inputs and connections ...")
ingroups = []
inpconns = []
for ing in range(Ningroups):
    ingroup = sl.tools.fast_synchronous_input_gen(Nin, fin,
                                                  Sin, sigma, duration,
                                                  shuffle=False)
    inpconn = Connection(ingroup, nrngroup, 'V')
    # connect random subset of inputs to each cell
    for nrn in range(Nnrns):
        inputids = np.random.choice(range(Nin), Nconn, replace=False)
        for inp in inputids:
            inpconn[inp, nrn] = weight
    ingroups.append(ingroup)
    inpconns.append(inpconn)
network.add(*ingroups)
network.add(*inpconns)

print("Setting up monitors ...")
inpmons = [SpikeMonitor(ing) for ing in ingroups]
network.add(*inpmons)
vmon = StateMonitor(nrngroup, 'V', record=True)
network.add(vmon)
spikemon = SpikeMonitor(nrngroup)
network.add(spikemon)

print("Running simulation for %s ..." % (duration))
network.run(duration, report="stdout")
if spikemon.nspikes:
    vmon.insert_spikes(spikemon, Vth*2)
    printstats(vmon, spikemon)
ion()
figure("Spikes")
suptitle("Spike trains")
subplot(2,1,1)
title("Input")
raster_plot(*inpmons)
axis(xmin=0, xmax=duration/ms)
subplot(2,1,2)
title("Neurons")
raster_plot(spikemon)
axis(xmin=0, xmax=duration/ms)
figure("Voltages")
title("Membrane potential traces")
vmon.plot()
legend()
figure("Input signal")
title("Input signal")
inpsignal = geninputsignal(*inpmons)
t = arange(0*second, duration, dt)
plot(t, inpsignal[:len(t)])
show()
