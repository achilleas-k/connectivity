from brian import *
import spikerlib as sl

def connect_recurrent(excgroup, inhgroup):
    # TODO: binomial probability of connecting
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
    # TODO: create synfire chains
    print("NIY: Creatying synfire chains")
    exc2excconn = Connection(excgroup, excgroup)
    return exc2excconn

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
Nexc = 1000
Ninh = 1000
lifeq = Equations("""
dV/dt = (Vrest-V)/tau+I/C : volt
""")
# I/C = 1.4 volt/second
lifeq.prepare()
excgroup = NeuronGroup(Nexc, lifeq, threshold="V>Vth", reset=Vrest,
                        refractory=2*ms)
excgroup.V = Vrest
network.add(excgroup)
inhgroup = NeuronGroup(Ninh, lifeq, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)
inhgroup.V = Vrest
network.add(inhgroup)
exc2inhconn, inh2excconn, inh2inhconn = connect_recurrent(excgroup, inhgroup)
network.add(exc2inhconn, inh2excconn, inh2inhconn)
exc2excconn = create_chains(excgroup)
network.add(exc2excconn)

print("Setting up monitors ...")
excvmon = StateMonitor(excgroup, 'V', record=1)
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
    raster_plot(excspikemon)
    subplot(2,1,2)
    raster_plot(inhspikemon)
    show()
else:
    print("No spikes were fired.")
