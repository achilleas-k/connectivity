from brian import *
import spikerlib
import itertools as it

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
    nsamples = 100
    for idx, inputset in enumerate(allinputs):
        t, krd = spikerlib.metrics.kreuz.pairwise_mp(
                inputset, 0*second,
                duration, nsamples)
        kreuz.append(mean(krd))
        print("%i/%i ..." % (idx+1, len(allinputs)))
    return kreuz


defaultclock.dt = dt = 0.1*ms
duration = 2*second
w = 2*ms
Vrest = -70*mV
Vth = -50*mV
tau = 10*ms

N_in_lst = [30, 50, 70]
r_in_lst = [50*Hz]
w_in_lst = [0.4*mV, 0.5*mV, 0.7*mV]
S_in_lst = linspace(0, 1, 11)
N_total = len(N_in_lst)*len(r_in_lst)*len(w_in_lst)*len(S_in_lst)

network = Network()

lif_eq = Equations("dV/dt = (Vrest-V)/tau : volt")
lif_eq.prepare()
lif_group = NeuronGroup(N_total, lif_eq, threshold="V>Vth", reset=Vrest,
                        refractory=1*msecond)
lif_group.V = Vrest
network.add(lif_group)
inp_groups = []
inp_mons = []
randidx = range(N_total)
#shuffle(randidx)
nrnidx = 0
configs = []
for N_in, r_in, w_in, S_in in it.product(N_in_lst, r_in_lst, w_in_lst,
        S_in_lst):
    print("Constructing inputs for neuron %i/%i ..." % (nrnidx+1, N_total))
    sync, rand = spikerlib.tools.gen_input_groups(N_in, r_in, S_in,
                                                  0*ms, duration, dt)
    target = randidx[nrnidx]
    syncconn = Connection(source=sync, target=lif_group[target],
                            weight=w_in, sparseness=1.0)
    randconn = Connection(source=rand, target=lif_group[target],
                            weight=w_in, sparseness=1.0)
    syncmon = SpikeMonitor(sync)
    randmon = SpikeMonitor(rand)
    network.add(sync, rand, syncconn, randconn, syncmon, randmon)
    inp_groups.append((sync, rand))
    inp_mons.append((syncmon, randmon))
    configs.append((N_in, r_in, w_in, S_in))
    nrnidx += 1

print("Inputs defined and connected.")
print("Setting up global monitors ...")
spikemon = SpikeMonitor(lif_group)
vmon = StateMonitor(lif_group, "V", record=True)
network.add(spikemon, vmon)
print("Running simulation for %s" % duration)
network.run(duration, report="stdout")
npss = []
print("Calculating NPSS ...")
for v, sp in zip(vmon.values, spikemon.spiketimes.itervalues()):
    if len(sp)>2:
        slopes = spikerlib.tools.npss(v, sp, Vrest, Vth, tau, w, dt)
        npss.append(mean(slopes))
    else:
        npss.append(0)
print("Calculating pairwise Kreuz metric ...")
kreuz = []
allinputs = collect_input_spikes(inp_mons)
for nrnidx in range(N_total):
    print("%i/%i" % (nrnidx+1, N_total))
    #t, krd = spikerlib.metrics.kreuz.interval(allinputs[nrnidx],
    #        spikemon[nrnidx], samples=100)
    t, krd = spikerlib.metrics.kreuz.pairwise_mp(allinputs[nrnidx], 0*second,
            duration, 100)
    kreuz.append(mean(krd))
kreuz = array(kreuz)
npss = array(npss)
npss_kr = sqrt(1-kreuz/0.3)
errors = npss-npss_kr
print("Plotting ...")
#scatter(npss, kreuz, c=S_in)
#show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
#ax.scatter(S_in, kreuz, npss)
#ax.set_xlabel("S_in")
#ax.set_ylabel("D_spike")
#ax.set_zlabel("NPSS")
Nw_in = [c[0]*c[2] for c in configs]
color = [1 if nw > 24*mV else 0 for nw in Nw_in]
ax.scatter(Nw_in, npss, sqrt(1-array(kreuz)/0.3), c=color)
ax.set_xlabel("$N_{in}w_{in}$")
ax.set_ylabel("$NPSS$")
ax.set_zlabel("$NPSS'$")
figure()
scatter(npss, npss_kr)
plot([0, 1], "k--")
for n, e in zip(npss, errors):
    plot([n, n], [n, n-e], "b-")
axis([-0.05, 1.05, -0.05, 1.05])
xlabel("NPSS")
ylabel("Spike distance (rescaled)")
print("DONE!")
