from __future__ import print_function
from __future__ import division
from brian import (Network, Equations, NeuronGroup, SpikeMonitor, StateMonitor,
                   Connection, raster_plot,
                   defaultclock, second, ms, mV, Hz)
from numpy import (arange, zeros, exp, convolve, array, mean, corrcoef, random,
                   shape)
from matplotlib import pyplot
import itertools as it
import multiprocessing as mp
import spikerlib as sl
from quickga import GA

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

def dist_global(distmatrix):
    """Global distance is just the average across pairs"""
    dshape = shape(distmatrix)
    newshape = (dshape[0]*dshape[1], dshape[2])
    distmatrix.reshape(newshape)
    gdist = mean(distmatrix, 0)
    return gdist

def dist_inputs(idxes, distmatrix):
    """
    Average distance between all pairs of input spike trains specified by idxes
    """
    pairs = list(it.combinations(idxes, 2))
    dist_sum = zeros(nkreuzsamples)
    for i, j in pairs:
        dist_sum += distmatrix[i,j]
    dist_mean = dist_sum/len(pairs)
    return dist_mean

def _kreuz_pair(args):
    idces, spiketrains = args
    a, b = idces
    t, d = sl.metrics.kreuz.distance(spiketrains[a], spiketrains[b],
                                     0*second, duration, nkreuzsamples)
    return d

def dist_all_pairs(*spikemons):
    """
    Calculate the Kreuz distance between all pairs of spike trains and return
    the results in an NxN array (where N is the total number of spike trains).
    NB: This can take ages [Npairs = N*(N+1)/2]
    """
    # collect all spike trains into an array of arrays
    allspiketrains = []
    for smon in spikemons:
        allspiketrains.extend(smon.spiketimes.values())
    allspiketrains = array(allspiketrains)
    ntrains = len(allspiketrains)
    # use a multiprocessing pool to calculate all pairwise distances
    pair_idces = list(it.combinations(range(ntrains), 2))
    pool = mp.Pool()
    pool_results = pool.map(_kreuz_pair, zip(pair_idces,
                                             it.repeat(allspiketrains)))
    pool.close()
    pool.join()
    distances = zeros((ntrains, ntrains, nkreuzsamples))
    for idx, pr in zip(pair_idces, pool_results):
        i, j = idx
        distances[i,j] = pr
        distances[j,i] = pr
    return distances

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
        slopes = sl.tools.npss(trace, spikes, Vrest, Vth, tau, w, dt)
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

def find_input_set(slopes, outspikes_idx, inpmons):
    """
    Find set of inputs that maximises the correlation between input and slopes

    Uses a GA to find the set of input spike trains that maximises the
    correlation between the input signal (see `gen_input_signals`) -
    discretised by output spike times - with the slopes of the membrane
    potential at each spike time.
    """
    # Since the GA uses fixes length chromosomes, I'm going to assume I know
    # that the number of inputs is Nconn*Ningroups
    # TODO: Implement variable length chromosomes (in quickga)
    # On the other hand, I can have fixed length chromosome with length equal
    # to the number of inputs in total, that is just a bit string (on/off per
    # input index)
    maxpop = 100
    chromlength = Nconn*Ningroups
    mutation_prob = 0.01
    mutation_strength = 10
    genemin = 0
    genemax = Ningroups*Nin-1  # genemax is inclusive
    outfile = "ga_input_set.log"
    ga = GA(maxpop, chromlength, mutation_probability=mutation_prob,
            mutation_strength=mutation_strength, genemin=genemin,
            genemax=genemax, logfile=outfile, genetype=int)
    ga.fitnessfunc = fitnessfunc
    ga.optimise(1000, slopes, outspikes_idx, inpmons)
    # could just return population, but returning entire class is better for
    # checking on all individuals and maybe running a few more optimisation
    # rounds
    return ga

def fitnessfunc(individual, slopes, outspikes_idx, inpmons):
    inputidces = individual.chromosome
    kwidth = 10*tau
    kernel = exp(-arange(0*second, kwidth, dt)/tau)
    nbins = int(duration/dt)
    binnedcounts = zeros(nbins)
    try:
        for idx in inputidces:
            ingrpid = int(idx/Nin)
            inpidx = idx%Nin
            inputgroup = inpmons[ingrpid]
            inspikes = inputgroup[inpidx]
            binnedcounts += sl.tools.times_to_bin(inspikes, dt, duration)
    except IndexError:
        print(idx)
        print(ingrpid)
        raise
    signal = convolve(binnedcounts, kernel)
    signal_disc = signal[outspikes_idx]
    correlation = corrcoef(slopes, signal_disc)
    individual.fitness = 1-correlation[0,1]

print("Preparing simulation ...")
doplot = False
network = Network()
defaultclock.dt = dt = 0.1*ms
duration = 1*second
w = 2*ms
nkreuzsamples = 100
Vrest = 0*mV
Vth = 20*mV
tau = 20*ms
Nnrns = 5
Ningroups = 2
Nin = 100
fin = 10*Hz
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
        inpidx = inp%Nin
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
print("Done. Computing results ...")
inpsignal = gen_population_signal(*inpmons)
t = arange(0*second, duration, dt)
if spikemon.nspikes:
    vmon.insert_spikes(spikemon, Vth*2)
    printstats(vmon, spikemon)
    mslopes, allslopes = calcslopes(vmon, spikemon)
    inpsignals = gen_input_signals(inputneurons, *inpmons)
    nplot = 0
    disc_signals = []
    print("\nCorrelation between input signal and slopes")
    for sp, slopes, insgnl in zip(spikemon.spiketimes.itervalues(),
                                  allslopes,
                                  inpsignals):
        nplot += 1
        inds = array(sp/dt).astype("int")
        correlation = corrcoef(slopes, insgnl[inds])[0,1]
        disc_signals.append(insgnl[inds])
        print("%i:\t%.4f" % (nplot-1, correlation))
        if doplot:
            pyplot.subplot(Nnrns, 1, nplot)
            pyplot.plot(sp, slopes/max(slopes))
            pyplot.plot(sp, insgnl[inds]/max(insgnl))
            pyplot.axis(xmin=0*second, xmax=duration)


if doplot:
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
    pyplot.plot(t, inpsignal[:len(t)])
# membrane potential slopes with individual input signals
    pyplot.figure("Slopes and signals")


#optimisers = []
#for idx in range(Nnrns):
#    outspikes = spikemon[idx]
#    outspikes_idx = array(outspikes/dt).astype("int")
#    slopes = allslopes[idx]
#    ga = find_input_set(slopes, outspikes_idx, inpmons)
#    optimisers.append(ga)
#    # count hits for best individual
#    best_ind = ga.alltime_bestind
#    best_inputs = [(int(gene/Nin), int(gene%Nin))
#                   for gene in best_ind.chromosome]
#    hits = [1 if pair in inputneurons[idx] else 0
#            for pair in best_inputs]
#    accuracy = sum(hits)/len(hits)
#    print("Input search for neuron %i --- accuracy %2.2f %%" % (idx, accuracy*100))
