from __future__ import print_function
from __future__ import division
from brian import *
import spikerlib as sl
import itertools as it
import random as rnd


def connect_recurrent(excgroup, inhgroup):
    exc_weight = 10.5*mV  # 0.1 mV EPSP peak
    inh_weight = -22.5*mV  # -0.6 mV IPSP peak
    print("Constructing excitatory to excitatory connections ...")
    exexconn = Connection(excgroup, excgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=exc_weight,
                             spareseness=0.001)
    print("Constructing excitatory to inhibitory connections ...")
    exinconn = Connection(excgroup, inhgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=exc_weight,
                             sparseness=0.4)
    print("Constructing inhibitory recurrent connections ...")
    ininconn = Connection(inhgroup, inhgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=inh_weight,
                             sparseness=0.5)
    print("Constructing inhibitory to excitatory connections ...")
    inexconn = Connection(inhgroup, excgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=inh_weight,
                             sparseness=0.4)
    return exexconn, exinconn, inexconn, ininconn

def create_chains(excgroup):
    print("Creating synfire chains ...")
    nchains = 50
    nchains = 10
    nlinks = 20
    width = 100
    width = 10
    weight = 52.5*mV  # 0.5 mV EPSP peak
    synfirechainids = []
    synfireconns = []
    for nc in range(nchains):
        print("\r%i/%i ..." % (nc+1, nchains), end="")
        sys.stdout.flush()
        chainidxes = array(rnd.sample(range(len(excgroup)), nlinks*width))
        chainidxes = chainidxes.reshape(nlinks, width)
        synfirechainids.append(chainidxes)
        for prevlnk, nxtlnk in zip(chainidxes[:-1], chainidxes[1:]):
            delay = rand()*2.5*ms+0.5*ms  # uniform [0.5, 3]
            layerconn = Connection(excgroup, excgroup, state='gIn', delay=delay)
            for pl in prevlnk:
                for nl in nxtlnk:
                    layerconn[pl, nl] = weight
            synfireconns.append(layerconn)
    print()
    return synfirechainids, synfireconns

def create_synfire_inputs(excgroup, synfirenrns):
    print("Creating inputs for first link of each synfire chain ...")
    inputs = []
    connections = []
    weight = 52.5*mV  # 0.5 mV EPSP peak
    for chain in synfirenrns:
        chaininput = sl.tools.fast_synchronous_input_gen(100, 10*Hz, 1, 1*ms, duration)
        conn = Connection(chaininput, excgroup, state='gIn')
        firstlink = chain[0]
        for sfnrn in firstlink:
            conn[:,sfnrn] = weight
        inputs.append(chaininput)
        connections.append(conn)
    return inputs, connections

def plotchains(synfirenrns, spikemon):
    yheight = 0
    colours = it.cycle(['b', 'g', 'r', 'c', 'm'])
    for chain in synfirenrns:
        for layer in chain:
            c = colours.next()
            for nrn in layer:
                spikes = spikemon[nrn]
                ypts = ones(len(spikes))*yheight
                plot(spikes, ypts, c+'.')
                yheight += 1
        # chain separator
        plot([0, float(duration)], [yheight, yheight], 'k--')

def plotexcsorted(synfireidxes, spikemon):
    sfidx = unique(synfireidxes)
    notsfidx = delete(range(Nexc), sfidx)
    plotseq = append(sfidx, notsfidx)
    yheight = 0
    for idx in plotseq:
        spikes = spikemon[idx]
        ypts = ones(len(spikes))*yheight
        plot(spikes, ypts, 'b.')
        yheight += 1
    # sf - non-sf separator
    plot([0, float(duration)], [len(sfidx)]*2, 'k--')

def calcrates(excspikemon, inhspikemon):
    """
    Return list of firing rate of each cell
    """
    excrates = [len(spikes)/duration
                for spikes in excspikemon.spiketimes.itervalues()]
    inhrates = [len(spikes)/duration
                for spikes in inhspikemon.spiketimes.itervalues()]
    return excrates, inhrates

def calcdepthstats(excspikemon, synfirenrns):
    chainspikes = []
    for chain in synfirenrns:
        layerspikes = []
        for layer in chain:
            nspikeslayer = sum(len(excspikemon[idx])
                               for idx
                               in layer)
            layerspikes.append(nspikeslayer)
        chainspikes.append(layerspikes)
    return chainspikes

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
Iexc = 0*pA
Iinh = 250*pA
Nexc = 4000
Ninh = 1000
tau_exc = 0.2*ms
tau_inh = 0.6*ms
lifeq_exc = Equations("""
                      dV/dt = (a-Vrest-V)/tau+Iexc/C : volt
                      da/dt = (gIn-a)/tau_exc : volt
                      dgIn/dt = -gIn/tau_exc : volt
                      """)
lifeq_inh = Equations("""
                      dV/dt = (a-Vrest-V)/tau+Iinh/C : volt
                      da/dt = (gIn-a)/tau_inh : volt
                      dgIn/dt = -gIn/tau_inh : volt
                      """)
# I/C = 1.4 volt/second
lifeq_exc.prepare()
lifeq_inh.prepare()
excgroup = NeuronGroup(Nexc, lifeq_exc, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)
excgroup.V = Vrest
network.add(excgroup)
inhgroup = NeuronGroup(Ninh, lifeq_inh, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)
inhgroup.V = Vrest
network.add(inhgroup)
recurrent_conns = connect_recurrent(excgroup, inhgroup)
synfirenrns, exc2excconn = create_chains(excgroup)
synfireinput, synfireinputconn = create_synfire_inputs(excgroup, synfirenrns)
network.add(*recurrent_conns)
network.add(*exc2excconn)
network.add(*synfireinput)
network.add(*synfireinputconn)

print("Setting up monitors ...")
# record V of first link in first chain
recsynfire = synfirenrns[0].flatten()
synfirevmon = StateMonitor(excgroup, 'V', record=recsynfire)
# record a few random cells as well (make sure they're not in sf chains)
recsample = rnd.sample(range(Nexc), 100)
recsample = delete(recsample, recsynfire)
excvmon = StateMonitor(excgroup, 'V', record=recsample)
excspikemon = SpikeMonitor(excgroup)
#inhvmon = StateMonitor(inhgroup, 'V', record=True)
inhspikemon = SpikeMonitor(inhgroup)
network.add(excvmon, synfirevmon, excspikemon, inhspikemon)

print("Running simulation for %s ..." % (duration))
network.run(duration, report="stdout")
if excspikemon.nspikes:
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
