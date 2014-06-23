from brian import *
import spikerlib
import time

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
    for mon in inp_mons:
        allinputs.append(mon.spiketimes.values())
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
w = 0.1*ms
Vrest = -70*mV
Vth = -50*mV
tau = 10*ms

# number of simulations constant
N_total = 100
# keep parameters as lists, at least for now
# random N_in within [30, 70]
N_in_list = [n for n in randint(30, 71, N_total)]
# random r_in within [40, 100] Hz
r_in_list = [int(r)*Hz for r in randint(40, 101, N_total)]
# random w_in within bounds set according to the following contraint
# 1.1*(Vth-Vrest) <= N_in*w_in <= 2*(Vth-Vrest)
Nw_low = 1.1*(Vth-Vrest)
Nw_high = 2*(Vth-Vrest)
w_low = [Nw_low/N_in for N_in in N_in_list]
w_high = [Nw_high/N_in for N_in in N_in_list]
w_in_list = [(rand()*(h-l)+l) for h, l in zip(w_high, w_low)]
# random S_in within [0, 1] (rounded to 2 decimals)
S_in_list = np.round(rand(N_total), 2)
sigma_in_list = np.round(rand(N_total)*4*ms, 4)
# make half the simulations have sigma = 0 ms
for idx in range(N_total/2):
    sigma_in_list[idx] = 0*ms

network = Network()

lif_eq = Equations("dV/dt = (Vrest-V)/tau : volt")
lif_eq.prepare()
lif_group = NeuronGroup(N_total, lif_eq, threshold="V>Vth", reset=Vrest,
                        refractory=1*msecond)
lif_group.V = Vrest
network.add(lif_group)
inp_groups = []
inp_mons = []
nrnidx = 0
configs = []
for N_in, r_in, w_in, S_in, sigma_in in zip(N_in_list, r_in_list, w_in_list,
        S_in_list, sigma_in_list):
    print("Constructing inputs for neuron %i/%i ..." % (nrnidx+1, N_total))
    input_group = spikerlib.tools.fast_synchronous_input_gen(
            N_in, r_in, S_in,
            sigma_in*second,
            duration)
    input_conn = Connection(input_group, lif_group, 'V')
    input_conn.connect_full(input_group, lif_group[nrnidx], weight=w_in)
    input_mon = SpikeMonitor(input_group)
    network.add(input_group, input_conn, input_mon)
    inp_groups.append(input_group)
    inp_mons.append(input_mon)
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
filename = "npss_kreuz_mc_%i.npz" % (int(time.time()))
print("Saving results to %s ... " % filename)
np.savez(filename,
        N_in=N_in_list,
        r_in=r_in_list,
        w_in=w_in_list,
        S_in=S_in_list,
        sigma_in=sigma_in_list,
        npss=npss,
        kreuz=kreuz)
print("DONE!")
