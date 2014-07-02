from brian import *
import spikerlib as sl

def connect_recurrent(excgroup, inhgroup):
    print("Constructing inhibitory recurrent connections ...")
    # TODO: make inh2inhconn
    print("Constructing inhibitory to excitatory connections ...")
    # TODO: make inh2excconn
    print("Constructing excitatory to inhibitory connections ...")
    # TODO: make exc2inhconn
    return None, None, None

print("Preparing simulation ...")
defaultclock.dt = dt = 0.1*ms
duration = 2*second
w = 2*ms
Vrest = 0*mV
Vth = 20*mV
tau = 20*ms
C = 250*pF
I = 350*pA
Nexc = 10000
Ninh = 10000
lifeq = Equations("""
dV/dt = (Vrest-V)/tau+I/C : volt
""")
# I/C = 1.4 volt/second
lifeq.prepare()
excgroup = NeuronGroup(Nexc, lifeq, threshold="V>Vth", reset=Vrest,
                        refractory=2*ms)
excgroup.V = Vrest
inhgroup = NeuronGroup(Ninh, lifeq, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)
inhgroup.V = Vrest
exc2inhconn, ing2excconn, inh2inhconn = connect_recurrent(excgroup, inhgroup)

print("Setting up monitors ...")
excvmon = StateMonitor(excgroup, 'V', record=1)
excspikemon = SpikeMonitor(excgroup)
#inhvmon = StateMonitor(inhgroup, 'V', record=True)
inhspikemon = SpikeMonitor(inhgroup)

print("Running simulation for %s ..." % (duration))
run(duration, report="stdout")
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
