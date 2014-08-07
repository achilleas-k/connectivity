from __future__ import print_function
from __future__ import division
from brian import *
import spikerlib as sl
import itertools as it
import random as rnd


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
    in the neuron group `source_group`. Any number of Connection objects may
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


def printstats(excrates, chainspikes, inhrates, synfirenrns):
    """
    Print spiking stats
    """
    avg_exc_rate = mean(excrates)
    avg_inh_rate = mean(inhrates)
    print("Average excitatory firing rate: %s" % (avg_exc_rate))
    if len(excrates) > count_nonzero(excrates):
        avg_exc_rate_spikeonly = sum(excrates)/count_nonzero(excrates)
        print("Average excitatory firing rate (spiking cells only): %s" % (
            avg_exc_rate_spikeonly))
    else:
        print("All excitatory cells fired.")
    print("Average inhibitory firing rate: %s" % (avg_inh_rate))
    if len(inhrates) > count_nonzero(inhrates):
        avg_inh_rate_spikeonly = sum(inhrates)/count_nonzero(inhrates)
        print("Average inhibitory firing rate (spiking cells only): %s" % (
            avg_inh_rate_spikeonly))
    else:
        print("All inhibitory cells fired.")
    spiking_sf_nrns = 0
    spiking_nonsf_nrns = 0
    synfireidx_flat = unique(flatten(synfirenrns))
    Nsf = len(synfireidx_flat)
    for idx in range(Nexc):
        if (idx in synfireidx_flat) and (excrates[idx] > 0):
            spiking_sf_nrns += 1
        elif (idx not in synfireidx_flat) and (excrates[idx] > 0):
            spiking_nonsf_nrns += 1
    print("Number of neurons which spiked")
    print("Synfire chain:      %4i/%4i" % (spiking_sf_nrns, Nsf))
    print("Non synfire chain:  %4i/%4i" % (spiking_nonsf_nrns, Nexc-Nsf))
    print("Excitatory (total): %4i/%4i" % (count_nonzero(excrates),
                                           len(excrates)))
    print("Inhibitory:         %4i/%4i" % (count_nonzero(inhrates),
                                           len(inhrates)))
    mean_chainspikes = mean(chainspikes, axis=0)
    maxdepth = flatnonzero(mean_chainspikes)[-1]+1
    meandepth = mean([flatnonzero(array(cs))[-1]+1 for cs in chainspikes])
    print("The longest chain propagation depth was %i (max %i)" % (
        maxdepth, len(chainspikes[0])))
    print("The average chain propagation depth was %f (max %i)" % (
        meandepth, len(chainspikes[0])))


print("Preparing simulation ...")
network = Network()
defaultclock.dt = dt = 0.1*ms
duration = 1*second
w = 2*ms
Vrest = 0*mV
Vth = 20*mV
tau = 20*ms
C = 250*pF
Nexc = 4
tau_exc = 0.2*ms
tau_inh = 0.6*ms
lifeq_exc = Equations("""
                      dV/dt = (a-Vrest-V)/tau : volt
                      da/dt = (gIn-a)/tau_exc : volt
                      dgIn/dt = -gIn/tau_exc : volt
                      """)
lifeq_exc.prepare()
excgroup = NeuronGroup(Nexc, lifeq_exc, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)
excgroup.V = Vrest

print("Setting up monitors ...")
# record a few random cells as well (make sure they're not in sf chains)
vmon = StateMonitor(excgroup, 'V', record=True)
spikemon = SpikeMonitor(excgroup)
inpmon = SpikeMonitor(inpgroup)
network.add(inpmon, vmon, spikemon)

print("Running simulation for %s ..." % (duration))
network.run(duration, report="stdout")
if spikemon.nspikes:
    synfirevmon.insert_spikes(excspikemon, Vth*2)
    excvmon.insert_spikes(excspikemon, Vth*2)
    excrates, inhrates = calcrates(excspikemon, inhspikemon)
    chainspikes = calcdepthstats(excspikemon, synfirenrns)
    printstats(excrates, chainspikes, inhrates, synfirenrns)
    print("Calculating slope distributions ...")
    synfire_slopes = calcslopes(synfirevmon, excspikemon)
    nonsf_slopes = calcslopes(excvmon, excspikemon)
else:
    print("No spikes were fired.")
