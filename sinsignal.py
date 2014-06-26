from brian import *


print("Preparing simulation ...")
defaultclock.dt = dt = 0.1*ms
duration = 20*second
w = 2*ms
Vrest = -60*mV
Vth = -50*mV
tau = 20*ms

DO THE CONVOLUTION THING

n_ext = 100
p_ext = 0.5
p_int = 0.05
w_ext = 0.5*mV
r_ext = 10*Hz
w_int = 0.1*mV
lif_eq = Equations("""
                    dV/dt = (Vrest-V)/tau+xi*I/sqrt(dt) : volt
                    I = 10*mV*sin(t*5*Hz*pi) : volt
                   """)
lif_eq.prepare()
lifgroup = NeuronGroup(1000, lif_eq, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)
lifgroup.V = Vrest

inp_group = PoissonGroup(n_ext, rates=r_ext)
inp_conn = Connection(inp_group, lifgroup, weight=w_ext,
                      sparseness=p_ext, fixed=True, delay=5*ms)

print("Setting up monitors ...")
# monitors
inpmon = SpikeMonitor(inp_group)
vmon = StateMonitor(lifgroup, 'V', record=True)
spikemon = SpikeMonitor(lifgroup)
print("Running simulation for %s ..." % (duration))
run(duration, report="stdout")
print("done.\nPlotting ...")
raster_plot(spikemon)
show()
