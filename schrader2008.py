from __future__ import print_function
from __future__ import division
from brian import *
import spikerlib as sl
import itertools as it
import random as rnd


def connect_recurrent(excgroup, inhgroup):
    exc_weight = 10.5*mV  # 0.1 mV EPSP peak
    inh_weight = -22.5*mV  # -0.6 mV IPSP peak
    print("Constructing excitatory to inhibitory connections ...")
    exc2inhconn = Connection(excgroup, inhgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=exc_weight,
                             sparseness=0.9)
    print("Constructing inhibitory recurrent connections ...")
    inh2inhconn = Connection(inhgroup, inhgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=inh_weight,
                             sparseness=0.5)
    print("Constructing inhibitory to excitatory connections ...")
    inh2excconn = Connection(inhgroup, excgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=inh_weight,
                             sparseness=1.0)
    return exc2inhconn, inh2excconn, inh2inhconn

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
    nonspiking_sf_nrns = 0
    spiking_nonsf_nrns = 0
    synfireidx_flat = [idx for idx in flatten(synfirenrns)]
    for idx in range(Nexc):
        if (idx in synfireidx_flat) and (not excrates[idx]):
            nonspiking_sf_nrns += 1
        elif (idx not in synfireidx_flat) and (excrates[idx]):
            spiking_nonsf_nrns += 1
    print("%i neurons were in a synfire chain and did not spike" % (
        nonspiking_sf_nrns))
    print("%i neurons were not in a synfire chain and spiked" % (
        spiking_nonsf_nrns))
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
I = 350*pA
Nexc = 4000
Ninh = 1000
tau_exc = 0.2*ms
tau_inh = 0.6*ms
lifeq_exc = Equations("""
dV/dt = (a-Vrest-V)/tau+I/C : volt
da/dt = (gIn-a)/tau_exc : volt
dgIn/dt = -gIn/tau_exc : volt
""")
lifeq_inh = Equations("""
dV/dt = (a-Vrest-V)/tau+I/C : volt
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
network.add(*recurrent_conns)
synfirenrns, exc2excconn = create_chains(excgroup)
network.add(*exc2excconn)
synfireinput, synfireinputconn = create_synfire_inputs(excgroup, synfirenrns)
network.add(*synfireinput)
network.add(*synfireinputconn)

print("Setting up monitors ...")
# record V of first link in first chain
synfirevmon = StateMonitor(excgroup, 'V', record=synfirenrns[0].flatten())
# record a few random cells as well
recsample = rnd.sample(range(Nexc), 20)
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
    # TODO: Print chain stats: Average number of spikes per link, detailed
    # number of spikes per link per chain, max propagation depth, average
    # propagation depth
    excrates, inhrates = calcrates(excspikemon, inhspikemon)
    chainspikes = calcdepthstats(excspikemon, synfirenrns)
    printstats(excrates, chainspikes, inhrates, synfirenrns)
    print("done.\nPlotting ...")
    t = arange(0*ms, duration, dt)
    figure()
    subplot(2,1,1)
    raster_plot(excspikemon)
    title("Excitatory population")
    subplot(2,1,2)
    raster_plot(inhspikemon)
    title("Inhibitory population")
    show()
else:
    print("No spikes were fired.")
