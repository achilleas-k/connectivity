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

def geninputsignal_partial(idxlist, *spikemons):
    # TODO: Generate the input signal for each receiving neuron
    inpsignals = []
    for idxs, monitor in zip(idxlist, spikemons):
        pass


def calcslopes(vmon, spikemon):
    """
    Calculate slopes (non normalised, for now)

    Returns average slopes (per cell) and individual slope values (flattened)
    TODO: Calculate normalised slopes
    """
    allslopes = []
    avgslopes = []
    for trace, spikes in zip(vmon.values, spikemon.spiketimes.itervalues()):
        if len(spikes) == 0:
            allslopes.append([])
            avgslopes.append([])
            continue
        spikeidx = array(spikes/dt).astype('int')
        slopestart = spikeidx-int(w/dt)
        slopes = (Vth-trace[slopestart])/w
        allslopes.append(slopes)
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
Nnrns = 5
Ningroups = 10
Nin = 1000
fin = 1*Hz
Sin = 0.1
sigma = 0*ms
weight = 1.0*mV
Nconn = 100  # number of connections each cell receives from each group
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
    ingroups.append(ingroup)
    inpconns.append(inpconn)
inputneurons = []
# TODO: Shove connections into a function
for nrn in range(Nnrns):
    inputids = np.random.choice(range(Nin*Ningroups), Nconn*Ningroups,
                                replace=False)
    inputneurons.append(inputids)
    for inp in inputids:
        inpgroup = int(inp/Nin)
        inpidx = inp % Nin
        inpconns[inpgroup][inpidx, nrn] = weight
network.add(*ingroups)
network.add(*inpconns)
asympt_v = fin*weight*tau*Nconn*Ningroups
print("Asymptotic threshold-free membrane potential: %s" % (asympt_v))

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
plot([0*second, duration], [Vth, Vth], 'k--')
legend()
figure("Input signal")
title("Input signal")
inpsignal = geninputsignal(*inpmons)
t = arange(0*second, duration, dt)
plot(t, inpsignal[:len(t)])
figure("Slopes")
mslopes, allslopes = calcslopes(vmon, spikemon)
for sp, slopes in zip(spikemon.spiketimes.itervalues(), allslopes):
    plot(sp, slopes)
    axis(xmin=0*second, xmax=duration)
figure("Per cell input signal")

show()
