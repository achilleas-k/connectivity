from __future__ import print_function
from __future__ import division
from brian import *
import spikerlib as sl


def calcslopes(vmon, spikemon):
    """
    Calculate slopes (non normalised, for now)
    TODO: Calculate normalised slopes

    Returns average slopes (per cell) and individual slope values (flattened)
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
Nexc = 4
Ningroups = 5
Nin = 50000
fin = 5*Hz
Sin = 0.2
sigma = 0*ms
weight = 0.1*mV
tau_exc = 0.2*ms
Nconn = int(0.01*Nin)  # number of connections each cell receives from each group
lifeq_exc = Equations("""
                      dV/dt = (a-Vrest-V)/tau : volt
                      da/dt = (gIn-a)/tau_exc : volt
                      dgIn/dt = -gIn/tau_exc : volt
                      """)
lifeq_exc.prepare()
nrngroup = NeuronGroup(Nexc, lifeq_exc, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)
nrngroup.V = Vrest
network.add(nrngroup)
print("Setting up inputs and connections ...")
ingroups = []
inpconns = []
for ing in range(Ningroups):
    ingroup = sl.tools.fast_synchronous_input_gen(Nin, fin,
                                                  Sin, sigma, duration)
    inpconn = Connection(ingroup, nrngroup, 'V')
    # connect random subset of inputs to each cell
    for nrn in range(Nexc):
        inputids = np.random.choice(range(Nin), Nconn, replace=False)
        for inp in inputids:
            inpconn[inp, nrn] = weight
    ingroups.append(ingroup)
    inpconns.append(inpconn)
network.add(*ingroups)
network.add(*inpconns)

print("Setting up monitors ...")
# record a few random cells as well (make sure they're not in sf chains)
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
figure("Spikes")
suptitle("Spike trains")
subplot(2,1,1)
title("Input")
raster_plot(*inpmons)
subplot(2,1,2)
title("Neurons")
raster_plot(spikemon)
figure("Voltages")
title("Membrane potential traces")
vmon.plot()
legend()
ion()
show()
#printstats(excrates, chainspikes, inhrates, synfirenrns)
