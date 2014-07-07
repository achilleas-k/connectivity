from __future__ import print_function
from brian import *
import spikerlib as sl


def connect_recurrent(excgroup, inhgroup):
    print("Constructing excitatory to inhibitory connections ...")
    exc2inhconn = Connection(excgroup, inhgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=0.1*mV,
                             sparseness=0.1)
    print("Constructing inhibitory recurrent connections ...")
    inh2inhconn = Connection(inhgroup, inhgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=-0.6*mV,
                             sparseness=0.1)
    print("Constructing inhibitory to excitatory connections ...")
    inh2excconn = Connection(inhgroup, excgroup, state='gIn',
                             delay=(0.5*ms, 3*ms), weight=-0.6*mV,
                             sparseness=0.1)
    return exc2inhconn, inh2excconn, inh2inhconn

def create_chains(excgroup):
    import random as rnd
    print("Creating synfire chains ...")
    nchains = 50
    nchains = 5
    nlinks = 20
    width = 100
    width = 10
    weight = 0.5*mV
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
    weight = 0.5*mV
    weight = 2*mV
    for chain in synfirenrns:
        chaininput = sl.tools.fast_synchronous_input_gen(100, 10*Hz, 1, 1*ms, duration)
        conn = Connection(chaininput, excgroup, state='gIn')
        firstlink = chain[0]
        for sfnrn in firstlink:
            conn[:,sfnrn] = weight
        inputs.append(chaininput)
        connections.append(conn)
    return inputs, connections


print("Preparing simulation ...")
network = Network()
defaultclock.dt = dt = 0.1*ms
duration = 2*second
w = 2*ms
Vrest = 0*mV
Vth = 20*mV
tau = 20*ms
C = 250*pF
I = 350*pA
I = 240*pA
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
excvmon = StateMonitor(excgroup, 'V', record=synfirenrns[0].flatten())  # entire first chain
excspikemon = SpikeMonitor(excgroup)
#inhvmon = StateMonitor(inhgroup, 'V', record=True)
inhspikemon = SpikeMonitor(inhgroup)
network.add(excvmon, excspikemon, inhspikemon)

print("Running simulation for %s ..." % (duration))
network.run(duration, report="stdout")
if excspikemon.nspikes:
    excvmon.insert_spikes(excspikemon, Vth*2)
    avg_exc_rate = excspikemon.nspikes/duration/Nexc
    avg_inh_rate = inhspikemon.nspikes/duration/Ninh
    print("Average excitatory firing rate: %s" % (avg_exc_rate))
    print("Average inhibitory firing rate: %s" % (avg_inh_rate))
    t = arange(0*ms, duration, dt)
    print("done.\nPlotting ...")
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
