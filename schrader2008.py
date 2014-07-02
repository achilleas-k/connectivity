from brian import *
import spikerlib as sl


print("Preparing simulation ...")
defaultclock.dt = dt = 0.1*ms
duration = 2*second
w = 2*ms
Vrest = 0*mV
Vth = 20*mV
tau = 20*ms
C = 250*pF
Nexc = 10000
Ninh = 10000
lifeq = Equations("""
dV/dt = (Vrest-V)/tau+I/C : volt
""")
lifeq.prepare()
excgrouup = NeuronGroup(Nexc, lifeq, threshold="V>Vth", reset=Vrest,
                        refractory=2*ms)
excgrouup.V = Vrest


print("Setting up monitors ...")
inpmon = SpikeMonitor(inpgroup)
vmon = StateMonitor(excgrouup, 'V', record=True)
spikemon = SpikeMonitor(excgrouup)
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
