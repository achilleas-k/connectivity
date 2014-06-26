from brian import *


print("Preparing simulation ...")
defaultclock.dt = dt = 0.1*ms
duration = 2*second
w = 2*ms
Vrest = -60*mV
Vth = -50*mV
tau = 20*ms

n_ext = 100
p_ext = 0.5
p_int = 0.05
w_ext = 0.9*mV
r_ext = 10*Hz
w_int = 0.1*mV
lif_eq = Equations("""
                    dV/dt = (Vrest-V)/tau+xi*I/sqrt(dt) : volt
                    I = 10*mV*sin(t*10*Hz*pi) : volt
                   """)
lif_eq.prepare()
lif_group = NeuronGroup(1000, lif_eq, threshold="V>Vth", reset=Vrest,
                       refractory=2*ms)

inp_group = PoissonGroup(n_ext, rates=r_ext)
# external input only applied to a part of the network
n_ext_rec = 100
inp_conn = Connection(inp_group, lif_group[:n_ext_rec], weight=w_ext,
                      sparseness=p_ext, fixed=True, delay=5*ms)

# monitors
inp_mon = SpikeMonitor(inp_group)
v_mon = StateMonitor(lif_group, 'V', record=True)
