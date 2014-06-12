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
    nsamples = 100
    for idx, inputset in enumerate(allinputs):
        t, krd = spikerlib.metrics.kreuz.pairwise_mp(
                inputset, 0*second,
                duration, nsamples)
        kreuz.append(mean(krd))
        print("%i/%i ..." % (idx+1, len(allinputs)))
    return kreuz


defaultclock.dt = dt = 0.1*ms
duration = 5*second
w = 2*ms
Vrest = -70*mV
Vth = -50*mV
tau = 10*ms

# number of simulations constant
N_total = 5000
# keep parameters as lists, at least for now
# random N_in within [30, 70]
N_in_lst = [n for n in randint(30, 71, N_total)]
# random r_in within [40, 100] Hz
r_in_lst = [int(r)*Hz for r in randint(40, 101, N_total)]
# random w_in within bounds set according to the following contraint
# 1.1*(Vth-Vrest) <= N_in*w_in <= 2*(Vth-Vrest)
Nw_low = 1.1*(Vth-Vrest)
Nw_high = 2*(Vth-Vrest)
w_low = [Nw_low/N_in for N_in in N_in_lst]
w_high = [Nw_high/N_in for N_in in N_in_lst]
w_in_lst = [(rand()*(h-l)+l) for h, l in zip(w_high, w_low)]
# random S_in within [0, 1] (rounded to 2 decimals)
S_in_lst = np.round(rand(N_total), 2)

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
for N_in, r_in, w_in, S_in in zip(N_in_lst, r_in_lst, w_in_lst, S_in_lst):
    print("Constructing inputs for neuron %i/%i ..." % (nrnidx+1, N_total))
    sync, rand = spikerlib.tools.gen_input_groups(N_in, r_in, S_in,
                                                  0*ms, duration, dt)
    target = randidx[nrnidx]
    syncconn = Connection(sync, lif_group, 'V')
    syncconn.connect_full(sync, lif_group[target], weight=w_in)
    randconn = Connection(rand, lif_group, 'V')
    randconn.connect_full(rand, lif_group[target], weight=w_in)
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
print("Running simulations for %s" % duration)
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
#print("Plotting ...")
##from mpl_toolkits.mplot3d import Axes3D
##fig = plt.figure()
##ax = fig.add_subplot(111, projection="3d")
##ax.scatter(Nw_in, npss, sqrt(1-array(kreuz)/0.3), c=color)
##ax.set_xlabel("$N_{in}w_{in}$")
##ax.set_ylabel("$NPSS$")
##ax.set_zlabel("$NPSS'$")
#figure()
#scatter(npss, npss_kr)
#plot([0, 1], "k--")
#for n, e in zip(npss, errors):
#    plot([n, n], [n, n-e], "b-")
#axis([-0.05, 1.05, -0.05, 1.05])
#xlabel("NPSS")
#ylabel("Spike distance (rescaled)")
filename = "npss_kreuz_mc.npz"
print("Saving results to %s ... " % filename)
np.savez(filename,
        N_in=N_in_lst,
        r_in=r_in_lst,
        w_in=w_in_lst,
        S_in=S_in_lst,
        npss=npss,
        kreuz=kreuz)


print("DONE!")
