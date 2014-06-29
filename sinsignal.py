from brian import *
import spikerlib as sl


print("Preparing simulation ...")
defaultclock.dt = dt = 0.1*ms
duration = 2*second
w = 2*ms
Vrest = -60*mV
Vth = -50*mV
tau = 20*ms
n_ext = 100
p_ext = 0.5
w_ext = 0.1*mV
r_ext = 10*Hz
lifeq = Equations("""
                    dV/dt = (Vrest-V+I)/tau+xi*0.2*mvolt/sqrt(dt) : volt
                    I = 10*mV*sin(t*5*Hz*pi) : volt
                   """)
lifeq.prepare()
lifgroup = NeuronGroup(1000, lifeq, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)
lifgroup.V = Vrest

inpgroup = PoissonGroup(n_ext, rates=r_ext)
inpconn = Connection(inpgroup, lifgroup, weight=w_ext,
                      sparseness=p_ext, fixed=True, delay=1*ms)

print("Setting up monitors ...")
inpmon = SpikeMonitor(inpgroup)
vmon = StateMonitor(lifgroup, 'V', record=True)
spikemon = SpikeMonitor(lifgroup)
print("Running simulation for %s ..." % (duration))
run(duration, report="stdout")
if spikemon.nspikes:
    vmon.insert_spikes(spikemon, Vth+50*mV)
    t = arange(0*ms, duration, dt)
    print("done.\nPlotting ...")
    figure()
    subplot2grid((4, 1), (0, 0), rowspan=1)
    plot(t, sin(t*5*Hz*pi))
    axis(xmin=0, xmax=float(duration))
    subplot2grid((4, 1), (1, 0), rowspan=2)
    raster_plot(spikemon)
    axis(xmin=0, xmax=duration/ms)
    t, convspikes = sl.tools.spikeconvolve(spikemon, 5*ms)
    subplot2grid((4, 1), (3, 0), rowspan=3)
    plot(t, convspikes)
    axis(xmin=0, xmax=float(duration))
    show()
else:
    print("No spikes were fired.")
