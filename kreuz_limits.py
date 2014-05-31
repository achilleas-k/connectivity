"""
Random sampling
"""
from brian import *
import spikerlib


def calc_npss(vmon, spikemon):
    npss = []
    print("Calculating NPSS ...")
    for v, sp in zip(vmon.values, spikemon.spiketimes.itervalues()):
        if len(sp)>2:
            slopes = spikerlib.tools.npss(v, sp, Vrest, Vth, tau, w, dt)
            npss.append((sp[1:], slopes))
        else:
            npss.append((0, 0))
    return npss


def collect_input_spikes(inp_mons):
    allinputs = []
    for monset in inp_mons:
        inputset = []
        smon, rmon = monset
        for synctrain in smon.spiketimes.itervalues():
            inputset.append(synctrain)
        for randtrain in rmon.spiketimes.itervalues():
            inputset.append(randtrain)
        allinputs.append(inputset)
    return allinputs


def calc_kreuz(allinputs):
    kreuz = []
    samples = []
    trains = []
    max_samples = 100
    max_trains = len(allinputs[0])
    for nsamples in range(10, max_samples+1, 10):
        for ntrains in range(10, max_trains+1, 5):
            for idx, inputset in enumerate(allinputs):
                t, krd = spikerlib.metrics.kreuz.pairwise_mp(
                        inputset[:ntrains], 0*second,
                        duration, nsamples)
                kreuz.append(mean(krd))
                trains.append(ntrains)
                samples.append(nsamples)
                print("Samples: %i - Trains: %s - Spike distance: %f" % ( 
                    nsamples, ntrains, mean(krd)))
    return samples, trains, kreuz


defaultclock.dt = dt = 0.1*ms
duration = 2*second
w = 2*ms
Vrest = -70*mV
Vth = -50*mV
tau = 10*ms

N_total = 3
N_in = 50
r_inp = 50*Hz
w_in = 1.01*mV

network = Network()

lif_eq = Equations("dV/dt = (Vrest-V)/tau : volt")
lif_eq.prepare()
lif_group = NeuronGroup(N_total, lif_eq, threshold="V>Vth", reset=Vrest,
                        refractory=1*msecond)
lif_group.V = Vrest
#network.add(lif_group)
inp_groups = []
inp_mons = []
for nrn_i in range(N_total):
    Sin = 1.0*nrn_i/(N_total-1)
    print("Constructing inputs for neuron %i with S_in %f ..." % (nrn_i, Sin))
    sync, rand = spikerlib.tools.gen_input_groups(N_in, r_inp, Sin,
                                                  0*ms, duration, dt)
    syncconn = Connection(source=sync, target=lif_group[nrn_i],
                            weight=w_in, sparseness=1.0)
    randconn = Connection(source=rand, target=lif_group[nrn_i],
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
#npss = calc_npss(vmon, spikemon)
print("Calculating pairwise Kreuz metric ...")
allinputs = collect_input_spikes(inp_mons)
samples, trains, kreuz = calc_kreuz(allinputs)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
rgb = ["r", "g", "b"]
colours = [rgb[i % 3] for i in range(len(kreuz))]
ax.scatter(samples, trains, kreuz, c=colours)
ax.set_xlabel("N samples")
ax.set_ylabel("N trains")
ax.set_zlabel("Spike distance")
show()
print("DONE!")



