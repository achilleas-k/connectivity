from brian import *
import spikerlib


defaultclock.dt = dt = 0.1*ms
duration = 1*second
w = 2*ms
Vrest = -70*mV
Vth = -50*mV
tau = 10*ms

N_layers = 11
N_per_layer = 100
N_total = N_layers*N_per_layer
N_in = 10
r_inp = 50*Hz
w_in = 0.3*mV

network = Network()

lif_eq = Equations("dV/dt = (Vrest-V)/tau : volt")
lif_eq.prepare()
lif_group = NeuronGroup(N_total, lif_eq, threshold="V>Vth", reset=Vrest,
                        refractory=1*msecond)
lif_group.V = Vrest
network.add(lif_group)
inp_groups = []
inp_mons = []
for layer in range(N_layers):
    print("Constructing inputs for layer %i ..." % (layer))
    sync, rand = spikerlib.tools.gen_input_groups(N_in, r_inp, 0.1*layer,
                                                  0*ms, duration, dt)
    start = layer*N_per_layer
    end = (layer+1)*N_per_layer
    syncconn = Connection(source=sync, target=lif_group[start:end],
                            weight=w_in, sparseness=1.0)
    randconn = Connection(source=rand, target=lif_group[start:end],
                            weight=w_in, sparseness=1.0)
    syncmon = SpikeMonitor(sync)
    randmon = SpikeMonitor(rand)
    network.add(sync, rand, syncconn, randconn, syncmon, randmon)
    inp_groups.append((sync, rand))
    inp_mons.append((syncmon, randmon))

print("Inputs defined and connected.")
print("Setting up global monitors ...")
spikemon = SpikeMonitor(lif_group)
vmon = StateMonitor(lif_group, "V", record=True)
network.add(spikemon, vmon)
print("Running simulation for %s" % duration)
network.run(duration)
npss = []
print("Calculating NPSS ...")
for v, sp in zip(vmon.values, spikemon.spiketimes.itervalues()):
    if len(sp)>2:
        slopes = spikerlib.tools.npss(v, sp, Vrest, Vth, tau, w, dt)
        mslope = mean(slopes)
        npss.append(mslope)
    else:
        npss.append(0)
print("Calculating pairwise Kreuz metric ...")
kreuz = []
idx = 0
for idx, monset in enumerate(inp_mons):
    smon, rmon = monset
    start = layer*N_per_layer
    end = (layer+1)*N_per_layer
    allinputs = []
    for synctrain in smon.spiketimes.itervalues():
        allinputs.append(synctrain)
    for randtrain in rmon.spiketimes.itervalues():
        allinputs.append(randtrain)
    outputspiketrains = spikemon.spiketimes.values()[start:end]
    for sp in outputspiketrains:
        idx += 1
        print("%i/%i" % (idx, N_total))
        t, krd = spikerlib.metrics.kreuz.interval(allinputs, sp)
        kreuz.append((t, krd))
print("DONE!")
