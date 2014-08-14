from __future__ import print_function
from __future__ import division
from brian import (Network, Equations, NeuronGroup, SpikeMonitor, StateMonitor,
                   Connection, raster_plot,
                   defaultclock, second, ms, mV, Hz)
from numpy import arange, zeros, exp, convolve, array, mean, corrcoef, random
from matplotlib import pyplot
import spikerlib as sl
from QuickGA import GA


def gen_population_signal(*spikemons):
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

def gen_input_signals(idxlist, *spikemons):
    inpsignals = []
    kwidth = 10*tau
    kernel = exp(-arange(0*second, kwidth, dt)/tau)
    nbins = int(duration/dt)
    for targetinputs in idxlist:
        binnedcounts = zeros(nbins)
        for ingrpid, inpidx in targetinputs:
            inputgroup = spikemons[ingrpid]
            inspikes = inputgroup[inpidx]
            binnedcounts += sl.tools.times_to_bin(inspikes, dt, duration)
        signal = convolve(binnedcounts, kernel)
        inpsignals.append(signal)
    return inpsignals

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

def find_input_set(slopes, outspikes, *inpmons):
    """
    Find set of inputs that maximises the correlation between input and slopes

    Uses a GA to find the set of input spike trains that maximises the
    correlation between the input signal (see `gen_input_signals`) -
    discretised by output spike times - with the slopes of the membrane
    potential at each spike time.
    """
    # Since the GA uses fixes length chromosomes, I'm going to assume I know
    # that the number of inputs is Nconn*Ningroups
    # TODO: Implement variable length chromosomes
    maxpop = 100
    chromlength = Nconn*Ningroups
    ga = GA(maxpop, chromlength)
    def fitnessfunc():
        pass
    ga.fitnessfunc = fitnessfunc

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
# CONNECTIONS
for nrn in range(Nnrns):
    inputids = random.choice(range(Nin*Ningroups), Nconn*Ningroups,
                                replace=False)
    inpnrns_row = []
    for inp in inputids:
        inpgroup = int(inp/Nin)
        inpidx = inp % Nin
        inpnrns_row.append((inpgroup, inpidx))
        inpconns[inpgroup][inpidx, nrn] = weight
    inputneurons.append(inpnrns_row)
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
pyplot.ion()
# spike trains figure
pyplot.figure("Spikes")
pyplot.suptitle("Spike trains")
pyplot.subplot(2,1,1)
pyplot.title("Input")
raster_plot(*inpmons)
pyplot.axis(xmin=0, xmax=duration/ms)
pyplot.subplot(2,1,2)
pyplot.title("Neurons")
raster_plot(spikemon)
pyplot.axis(xmin=0, xmax=duration/ms)
# voltages of target neurons
pyplot.figure("Voltages")
pyplot.title("Membrane potential traces")
vmon.plot()
pyplot.plot([0*second, duration], [Vth, Vth], 'k--')
pyplot.legend()
# global input population signal (exponential convolution)
pyplot.figure("Input signal")
pyplot.title("Input signal")
inpsignal = gen_population_signal(*inpmons)
t = arange(0*second, duration, dt)
pyplot.plot(t, inpsignal[:len(t)])
# membrane potential slopes with individual input signals
pyplot.figure("Slopes and signals")
mslopes, allslopes = calcslopes(vmon, spikemon)
inpsignals = gen_input_signals(inputneurons, *inpmons)
nplot = 0
disc_signals = []
print("\nCorrelation between input signal and slopes")
for sp, slopes, insgnl in zip(spikemon.spiketimes.itervalues(),
                              allslopes,
                              inpsignals):
    nplot += 1
    pyplot.subplot(Nnrns, 1, nplot)
    pyplot.plot(sp, slopes/max(slopes))
    inds = array(sp/dt).astype("int")
    pyplot.plot(sp, insgnl[inds]/max(insgnl))
    pyplot.axis(xmin=0*second, xmax=duration)
    correlation = corrcoef(slopes, insgnl[inds])[0,1]
    print("%i:\t%.4f" % (nplot-1, correlation))
    disc_signals.append(insgnl[inds])
pyplot.show()

# TODO: Run GA to find combination of inputs that maximises
# correlation between input signal and slopes

