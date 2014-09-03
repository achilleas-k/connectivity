from __future__ import print_function
from __future__ import division
from brian import (Network, Equations, NeuronGroup, SpikeMonitor, StateMonitor,
                   Connection, raster_plot,
                   defaultclock, second, ms, mV, Hz)
import numpy as np
from matplotlib import pyplot, mlab
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
    binnedcounts = np.zeros(nbins)
    for monitor in spikemons:
        for st in monitor.spiketimes.itervalues():
            binnedcounts += sl.tools.times_to_bin(st, dt, duration)
    kernel = np.exp(-np.arange(0*second, kwidth, dt)/tau)
    signal = np.convolve(binnedcounts, kernel)
    return signal

def gen_input_signals(idxlist, *spikemons):
    inpsignals = []
    kwidth = 10*tau
    kernel = np.exp(-np.arange(0*second, kwidth, dt)/tau)
    nbins = int(duration/dt)
    for targetinputs in idxlist:
        binnedcounts = np.zeros(nbins)
        for ingrpid, inpidx in targetinputs:
            inspikes = spikemons[ingrpid][inpidx]
            binnedcounts += sl.tools.times_to_bin(inspikes, dt, duration)
        signal = np.convolve(binnedcounts, kernel)
        inpsignals.append(signal)
    return inpsignals

def dist_global(dist_array):
    """
    Global distance is just the average across pairs
    """
    dshape = np.shape(dist_array)
    newshape = (dshape[0]*dshape[1], dshape[2])
    dist_array = dist_array.reshape(newshape)
    gdist = np.mean(dist_array, 0)
    return gdist

def dist_inputs(idxes, dist_array):
    """
    Average distance between all pairs of input spike trains specified by idxes
    """
    pairs = list(it.combinations(idxes, 2))
    dist_sum = np.zeros(nkreuzsamples)
    for i, j in pairs:
        a = i[0]*Nin_per_group+i[1]
        b = j[0]*Nin_per_group+j[1]
        dist_sum += dist_array[a,b]
    dist_mean = dist_sum/len(pairs)
    return dist_mean

def _interval(argtuple):
    """
    Helper function called by multiprocessing pool in `calc_new_pairs`.
    """
    spiketrains = (argtuple[0], argtuple[1])
    outspikes = argtuple[2]
    return sl.metrics.kreuz.interval(spiketrains, outspikes, mp=False)

def calc_new_pairs(idxlist, outspikes, spikemons):
    """
    Returns a new distance matrix with all pairwise distances found in `idxlist`
    calculated.
    """
    global distmatrix
    global n_pair_calcs
    global pair_calcs_saved
    pairs_to_calc = []
    mp_args = []
    for pair in it.combinations(idxlist, 2):
        one, two = pair
        if one == two: continue  # dist = 0, just carry on
        if np.any(distmatrix[one, two] > -1):
            # already calculated
            pair_calcs_saved += 1
            continue
        n_pair_calcs += 1
        oneg, onei = int(one//Nin_per_group), one%Nin_per_group
        twog, twoi = int(two//Nin_per_group), two%Nin_per_group
        input_one = spikemons[oneg][onei]
        input_two = spikemons[twog][twoi]
        mp_args.append((input_one, input_two, outspikes))
        pairs_to_calc.append(pair)
    pool = mp.Pool()  # use threads instead
    pool_results = pool.map(_interval, mp_args)
    pool.close()
    pool.join()
    for pair, result in zip(pairs_to_calc, pool_results):
        newdist = result[1]  # first element is `times`
        if len(newdist) != len(outspikes)-1:
            print("WARNING: Lengths don't match")
            print("Outspikes-1: %i" % (len(outspikes)-1))
            print("Distances:   %i" % (len(newdist)))
            print("This may cause, or be indicative of, bugs in the optimiser.")
        one, two = pair
        distmatrix[one, two] = newdist
        distmatrix[two, one] = newdist
    # TODO: REMOVE THE FOLLOWING LINES
    #print("Pair calculations so far: %i" % (n_pair_calcs))
    #print("Pair calculations saved:  %i" % (pair_calcs_saved))

def dist_inputs_interval(idxlist, outspikes, spikemons):
    """
    Returns the average pairwise distance between the inputs of a neuron,
    using the interval method, which segments the input spike trains based on
    the output.  Distances between pairs are saved in a matrix, symmetrically,
    in order to avoid recalculating known pairs. This is done to speed up the
    search when searching for an input group with the GA.

    The `idxlist` uses flat indexing, i.e., it is a list of numbers in the
    range [0, Nallin) and not tuples of (Neurongroup number, Neuron index)
    pairs.
    """
    global distmatrix
    if distmatrix is None:
        # initialise distance matrix (actually, numpy array)
        # -1 values indicate the pair has not been evaluated yet
        nspikes = len(outspikes)
        distmatrix = np.zeros((Nallin, Nallin, nspikes-1))-1
        for i in range(Nallin):
            distmatrix[i,i] = 0
    calc_new_pairs(idxlist, outspikes, spikemons)
    total_dist = 0
    count = 0
    for pair in it.combinations(idxlist, 2):
        i, j = pair
        total_dist += distmatrix[i,j]
        count += 1
    return total_dist/count

def dist_inputs_interval_all(idxlist, outspikemon, *spikemons):
    """
    Calculates distances of inputs for each neuron, using the interval method,
    which segments the input spike trains based on the output.
    This requires calculating a separate distance matrix for each output cell.
    This function returns the averaged distances directly, not a matrix of
    calculated paired distances.

    Unlike `dist_inputs_interval`, this function calculates the distances for
    multiple target (receiving) neurons and doesn't save the distances between
    each pair of input. The `idxlist` argument is a list of lists of tuples.
    Each tuple is a (Neurongroup number, Neuron index) pair and each list
    groups all the inputs of a neuron.
    """
    inputdists = []
    outspikes = outspikemon.spiketimes.values()
    for outsp, cellinputs in zip(outspikes, idxlist):
        if len(outsp) == 0:
            inputdists.append(-1)
            continue
        inputtrains = []
        for ingrpid, inpidx in cellinputs:
            inputtrains.append(spikemons[ingrpid][inpidx])
        t, indists = sl.metrics.kreuz.interval(inputtrains, outsp, nkreuzsamples)
        inputdists.append(np.mean(indists, axis=1))
        #inputdists.append(np.array(indists)[:,-1])
        print("%i/%i ..." % (len(inputdists), Nnrns))
    return inputdists

def _kreuz_pair(args):
    """
    Helper function called by multiprocessing pool in `dist_all_pairs`.
    """
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
    allspiketrains = np.array(allspiketrains)
    ntrains = len(allspiketrains)
    # use a multiprocessing pool to calculate all pairwise distances
    pair_idces = list(it.combinations(range(ntrains), 2))
    pool = mp.Pool()
    pool_results = pool.map(_kreuz_pair, zip(pair_idces,
                                             it.repeat(allspiketrains)))
    pool.close()
    pool.join()
    distances = np.zeros((ntrains, ntrains, nkreuzsamples))
    for idx, pr in zip(pair_idces, pool_results):
        i, j = idx
        distances[i,j] = pr
        distances[j,i] = pr
    return distances

def calcslopes(vmon, spikemon):
    """
    Calculate NPSS.

    Returns average slopes (per cell) and individual slope values (flattened)
    """
    allslopes = []
    avgslopes = []
    for trace, spikes in zip(vmon.values, spikemon.spiketimes.itervalues()):
        if len(spikes) == 0:
            allslopes.append([])
            avgslopes.append([])
            continue
        slopes = sl.tools.npss(trace, spikes, Vrest, Vth, tau, w, dt)
        slopes = slopes[1:]
        allslopes.append(slopes)
        avgslopes.append(np.mean(slopes))
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
    avgrate = np.mean(spikerates)
    xcorrs = sl.metrics.corrcoef.corrcoef_spiketrains(spiketrains)
    print("Spike rates: ")
    for idx, sr in enumerate(spikerates):
        print("%3i:\t%0.2f Hz" % (idx, sr))
    print("Avg:\t%0.2f Hz" % (avgrate))
    print("\nSpike train correlations")
    print("\t"+"\t".join("%4i" % i for i in range(len(xcorrs))))
    for idx, corr in enumerate(xcorrs):
        print(str(idx)+"\t"+"\t".join("%.2f" % c for c in corr))

def find_input_set(slopes, outspikes, inpmons):
    """
    Find set of inputs that maximises the correlation between input and slopes

    Uses a GA to find the set of input spike trains that maximises the
    correlation between the input signal (see `gen_input_signals`) -
    discretised by output spike times - with the slopes of the membrane
    potential at each spike time.
    """
    # Since the GA uses fixes length chromosomes, I'm going to assume I know
    # that the number of inputs is Nin
    # TODO: Implement variable length chromosomes (in quickga)
    # On the other hand, I can have fixed length chromosome with length equal
    # to the number of inputs in total, that is just a bit string (on/off per
    # input index). This would also take care of not allowing duplicates.
    maxpop = 10
    chromlength = Nin
    mutation_prob = 0.01
    mutation_strength = 10
    genemin = 0
    genemax = Ningroups*Nin_per_group-1  # genemax is inclusive
    outfile = "ga_input_set.log"
    ga = GA(maxpop, chromlength, mutation_probability=mutation_prob,
            mutation_strength=mutation_strength, genemin=genemin,
            genemax=genemax, logfile=outfile, genetype=int)
    ga.fitnessfunc = fitnessfunc
    ga.optimise(500, slopes, outspikes, inpmons)
    # could just return population, but returning entire class is better for
    # checking on all individuals and maybe running a few more optimisation
    # rounds
    return ga

def fitnessfunc(individual, slopes, outspikes, inpmons):
    win = len(outspikes)//2
    inputidces = individual.chromosome
    input_dist = dist_inputs_interval(inputidces, outspikes, inpmons)
    correlation = cor_movavg(slopes, input_dist, win)
    # TODO: negative correlation --- this should be fixed (maybe in the GA?)
    individual.fitness = 1-abs(correlation)

def cor_movavg(slopes, kreuz, win):
    masl = mlab.movavg(slopes, win)
    makr = mlab.movavg(kreuz, win)
    return np.corrcoef(masl, makr)[1,0]

def cor_movavg_all(allslopes, allkreuz, win):
    return [cor_movavg(sl, kr, win) for sl, kr in zip(allslopes, allkreuz)]

print("Preparing simulation ...")
doplot = False
network = Network()
defaultclock.dt = dt = 0.1*ms
duration = 10*second
w = 2*ms
nkreuzsamples = 3
Vrest = 0*mV
Vth = 20*mV
tau = 20*ms
Nnrns = 4
Ningroups = 1
Nin_per_group = 50
fin = 20*Hz
ingroup_sync = [0.5]
sigma = 0*ms
weight = 2.0*mV
Nallin = Nin_per_group*Ningroups
Nin = 25  # total number of connections each cell receives

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
    ingroup = sl.tools.fast_synchronous_input_gen(Nin_per_group, fin,
                                                  ingroup_sync[ing], sigma, duration,
                                                  shuffle=False)
    inpconn = Connection(ingroup, nrngroup, 'V')
    ingroups.append(ingroup)
    inpconns.append(inpconn)
inputneurons = []

# CONNECTIONS
Sin = []
for nrn in range(Nnrns):
    #inputids = np.random.choice(range(Nin_per_group*Ningroups), Nin,
    #                            replace=False)
    cur_sin = np.random.rand()
    Sin.append(cur_sin)
    Nsync = int(Nin*cur_sin)
    Nrand = Nin-Nsync
    randids = np.random.choice(range(0, Nallin//2), Nrand, replace=False)
    syncids = np.random.choice(range(Nallin//2, Nallin), Nsync, replace=False)
    inputids = np.append(syncids, randids)
    inpnrns_row = []
    for inp in inputids:
        inpgroup = int(inp/Nin_per_group)
        inpidx = inp%Nin_per_group
        inpnrns_row.append((inpgroup, inpidx))
        inpconns[inpgroup][inpidx, nrn] = weight
    inputneurons.append(inpnrns_row)

fake_inputneurons = []
for cur_sin in Sin:
    Nsync = int(Nin*cur_sin)
    Nrand = Nin-Nsync
    randids = np.random.choice(range(0, Nallin//2), Nrand, replace=False)
    syncids = np.random.choice(range(Nallin//2, Nallin), Nsync, replace=False)
    inputids = np.append(syncids, randids)
    inpnrns_row = []
    for inp in inputids:
        inpgroup = int(inp/Nin_per_group)
        inpidx = inp%Nin_per_group
        inpnrns_row.append((inpgroup, inpidx))
    fake_inputneurons.append(inpnrns_row)
network.add(*ingroups)
network.add(*inpconns)
asympt_v = fin*weight*tau*Nin
print("Asymptotic threshold-free membrane potential: %s" % (asympt_v))
max_volley = max(ingroup_sync)*Nin*weight
print("Max spike volley potential: %s" % (max_volley))

print("Setting up monitors ...")
inpmons = [SpikeMonitor(ing) for ing in ingroups]
network.add(*inpmons)
vmon = StateMonitor(nrngroup, 'V', record=True)
network.add(vmon)
spikemon = SpikeMonitor(nrngroup)
network.add(spikemon)

print("Running simulation for %s ..." % (duration))
network.run(duration, report="stdout")
print("Done.")

###### THE OTHER STUFF ######

print("Computing results ...")
t = np.arange(0*second, duration, dt)
best_corr = -1
if spikemon.nspikes:
    vmon.insert_spikes(spikemon, Vth*2)
    print("Calculating pairwise, interval Kreuz distance for the inputs of each cell ...")
    krinpdist = dist_inputs_interval_all(inputneurons, spikemon, *inpmons)
    meankr = [np.mean(kr) for kr in krinpdist]
    printstats(vmon, spikemon)
    mslopes, allslopes = calcslopes(vmon, spikemon)
    n = 0
    disc_signals = []
    print("\nCorrelation between Kreuz distances and slopes")
    n = 0
    for slopes, kr in zip(allslopes, krinpdist):
        correlation = np.corrcoef(slopes, kr)[0,1]
        print("%i:\t%.4f" % (n, correlation))
        n += 1
    print("\nCorrelations between moving averages")
    minspikes = min([len(sp) for sp in spikemon.spiketimes.itervalues()])
    win = minspikes//2
    #for win in np.linspace(1, minspikes/2, 3):
    print("\nWindow length: %i" % win)
    corrs = cor_movavg_all(allslopes, krinpdist, win)
    best_corr = np.argmin(corrs)
    for n, c in enumerate(corrs):
        print("%i\t%.4f" % (n, c))

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
    if Nnrns < 10:
        # skip plotting if too many
        pyplot.figure("Voltages")
        pyplot.title("Membrane potential traces")
        for idx in range(Nnrns):
            pyplot.subplot(Nnrns, 1, idx+1)
            pyplot.plot(vmon.times, vmon[idx])
            pyplot.plot([0*second, duration], [Vth, Vth], 'k--')
            pyplot.axis(ymax=float(Vth*2))
# membrane potential slopes with individual distances
    pyplot.figure("Slopes and distances")
    nplot = 0
    if spikemon.nspikes:
        for sp, slopes, kr, in zip(spikemon.spiketimes.itervalues(),
                                       allslopes,
                                       krinpdist):
            nplot += 1
            pyplot.subplot(Nnrns, 1, nplot)
            pyplot.plot(sp[1:], slopes)
            pyplot.plot(sp[1:], kr)

optimisers = []
global distmatrix
distmatrix = None
# TODO: Dictionary of pair distances would save on memory
# Matrix preallocates worst-case memory
global n_pair_calcs
n_pair_calcs = 0
global pair_calcs_saved
pair_calcs_saved = 0
for idx in range(Nnrns):
    outspikes = spikemon[idx]
    slopes = allslopes[idx]
    ga = find_input_set(slopes, outspikes, inpmons)
    optimisers.append(ga)
    # count hits for best individual
    best_ind = ga.alltime_bestind
    best_inputs = [(int(gene/Nin_per_group), int(gene%Nin_per_group))
                   for gene in best_ind.chromosome]
    hits = [1 if pair in inputneurons[idx] else 0
            for pair in best_inputs]
    accuracy = sum(hits)/len(hits)
    print("Input search for neuron %i --- accuracy %2.2f %%" % (idx, accuracy*100))
