from __future__ import print_function
from brian import *
import spikerlib as sl

def connect_recurrent(excgroup, inhgroup):
    print("Constructing excitatory to inhibitory connections ...")
    exc2inhconn = Connection(excgroup, inhgroup, state='V',
                             delay=(0.5*ms, 3*ms), weight=0.1*mV,
                             sparseness=0.1)
    print("Constructing inhibitory recurrent connections ...")
    inh2inhconn = Connection(inhgroup, inhgroup, state='V',
                             delay=(0.5*ms, 3*ms), weight=-0.6*mV,
                             sparseness=0.1)
    print("Constructing inhibitory to excitatory connections ...")
    inh2excconn = Connection(inhgroup, excgroup, state='V',
                             delay=(0.5*ms, 3*ms), weight=-0.6*mV,
                             sparseness=0.1)
    return exc2inhconn, inh2excconn, inh2inhconn

def create_chains(excgroup):
    print("Creating synfire chains ...")
    nchains = 50
    nchains = 1
    nlinks = 20
    width = 100
    weight = 0.5*mV
    synfireconns = []
    for nc in range(nchains):
        print("\r%i/%i ..." % (nc+1, nchains), end="")
        sys.stdout.flush()
        chainidxes = range(len(excgroup))
        #shuffle(chainidxes)
        chainidxes = array(chainidxes[:nlinks*width])
        chainidxes = chainidxes.reshape(nlinks, width)
        for prevlnk, nxtlnk in zip(chainidxes[:-1], chainidxes[1:]):
            delay = rand()*2.5*ms+0.5*ms  # uniform [0.5, 3]
            layerconn = Connection(excgroup, excgroup, state='V', delay=delay)
            for pl in prevlnk:
                for nl in nxtlnk:
                    layerconn[pl, nl] = weight
            synfireconns.append(layerconn)
    return synfireconns

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
Nexc = 4000
Ninh = 1000
tau_exc = 0.2*ms
tau_inh = 0.6*ms
a = 0*mV
lifeq_exc = Equations("""
dV/dt = (a-Vrest-V)/tau+I/C : volt
#da/dt = (gIn-a)/tau_exc : volt
#dgIn/dt = -gIn/tau_exc : volt
""")
lifeq_inh = Equations("""
dV/dt = (a-Vrest-V)/tau+I/C : volt
#da/dt = (gIn-a)/tau_inh : volt
#dgIn/dt = -gIn/tau_inh : volt
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
exc2excconn = create_chains(excgroup)
network.add(*exc2excconn)

print("Setting up monitors ...")
excvmon = StateMonitor(excgroup, 'V', record=range(100))
excspikemon = SpikeMonitor(excgroup)
#inhvmon = StateMonitor(inhgroup, 'V', record=True)
inhspikemon = SpikeMonitor(inhgroup)
network.add(excvmon, excspikemon, inhspikemon)

print("Running simulation for %s ..." % (duration))
network.run(duration, report="stdout")
if excspikemon.nspikes:
    t = arange(0*ms, duration, dt)
    print("done.\nPlotting ...")
    figure()
    subplot(2,1,1)
    title("Excitatory population")
    raster_plot(excspikemon)
    subplot(2,1,2)
    title("Inhibitory population")
    raster_plot(inhspikemon)
    show()
else:
    print("No spikes were fired.")
