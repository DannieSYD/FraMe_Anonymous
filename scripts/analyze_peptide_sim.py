import warnings
from pyemma.util.exceptions import PyEMMA_DeprecationWarning

warnings.filterwarnings("ignore", category=PyEMMA_DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mddir', type=str, default='share/4AA_sims')
parser.add_argument('--pdbdir', type=str, required=True)
parser.add_argument('--save', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--save_name', type=str, default='out.pkl')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--no_msm', action='store_true')
parser.add_argument('--no_decorr', action='store_true')
parser.add_argument('--no_traj_msm', action='store_true')
parser.add_argument('--truncate', type=int, default=None)
parser.add_argument('--msm_lag', type=int, default=10)
parser.add_argument('--ito', action='store_true')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--peptide_timeout', type=int, default=1800,
                    help='Per-peptide wall-clock timeout in seconds. '
                         '0 disables. Requires POSIX (SIGALRM); ignored on Windows. '
                         'Note: mdtraj C-level reads can swallow SIGALRM and retry, '
                         'so wall-clock can exceed the timeout by 1-2x.')
parser.add_argument('--cache_dir', type=str, default=None,
                    help='Per-peptide cache directory (defaults to {pdbdir}/.analysis_cache). '
                         'Successful peptide results are pickled here so re-runs can skip them.')
parser.add_argument('--no_cache', action='store_true',
                    help='Disable per-peptide caching entirely.')
parser.add_argument('--force', action='store_true',
                    help='Re-run analysis even if cache hit. Cache is still updated on success.')

args = parser.parse_args()

import mdgen.analysis
import pyemma, tqdm, os, pickle, signal, time
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool
import numpy as np
np.bool = np.bool_
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; must be before pyplot import
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf, acf
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Per-peptide cache: stores `out` dicts keyed by peptide name. Errors are
# never cached (so a timeout / failed peptide is retried on re-run).
if args.no_cache:
    CACHE_DIR = None
else:
    CACHE_DIR = args.cache_dir or os.path.join(args.pdbdir, '.analysis_cache')

def _peptide_timeout_handler(signum, frame):
    raise TimeoutError(f'peptide processing exceeded {args.peptide_timeout}s')


def main(name):
    t0 = time.time()

    # Cache lookup: skip if we already have a successful result for this name.
    cache_path = os.path.join(CACHE_DIR, f'{name}.pkl') if CACHE_DIR else None
    if cache_path and not args.force and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and 'error' not in cached:
                print(f'=== {name}: cache hit, skipping', flush=True)
                return name, cached
            else:
                print(f'... {name}: cached result has error, re-running', flush=True)
        except Exception as e:
            print(f'... {name}: cache load failed ({e}), re-running', flush=True)

    print(f'>>> {name}: starting', flush=True)
    out = {}
    np.random.seed(137)

    # Per-peptide wall-clock timeout (POSIX SIGALRM only). Each multiprocessing
    # worker has its own signal state, so this is safe under Pool.imap_unordered.
    use_timeout = args.peptide_timeout > 0 and hasattr(signal, 'SIGALRM')
    if use_timeout:
        signal.signal(signal.SIGALRM, _peptide_timeout_handler)
        signal.alarm(args.peptide_timeout)

    try:
        result_name, result_out = _main_body(name, out)
        # Persist successful result so future runs skip this peptide.
        if cache_path and 'error' not in result_out:
            os.makedirs(CACHE_DIR, exist_ok=True)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result_out, f)
            except Exception as e:
                print(f'... {name}: cache save failed ({e})', flush=True)
        return result_name, result_out
    except TimeoutError as e:
        dt = time.time() - t0
        print(f'!!! {name}: TIMEOUT after {dt:.0f}s ({e})', flush=True)
        return name, {'error': f'timeout after {args.peptide_timeout}s'}
    except Exception as e:
        print(f'!!! {name}: ERROR {type(e).__name__}: {e}', flush=True)
        return name, {'error': f'{type(e).__name__}: {e}'}
    finally:
        if use_timeout:
            signal.alarm(0)
        dt = time.time() - t0
        print(f'<<< {name}: done in {dt:.1f}s', flush=True)


def _main_body(name, out):
    # Only allocate the figure when we actually need it
    fig = None
    if args.plot:
        fig, axs = plt.subplots(4, 4, figsize=(20, 20))

    ### BACKBONE torsion marginals PLOT ONLY
    if args.plot:
        feats, traj = mdgen.analysis.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=False, cossin=False)
        if args.truncate: traj = traj[:args.truncate]
        feats, ref = mdgen.analysis.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=False, cossin=False)
        # if args.truncate: ref = ref[::100]
        pyemma.plots.plot_feature_histograms(ref, feature_labels=feats, ax=axs[0,0], color=colors[0])
        pyemma.plots.plot_feature_histograms(traj, ax=axs[0,0], color=colors[1])
        axs[0,0].set_title('BB torsions')


    ### JENSEN SHANNON DISTANCES ON ALL TORSIONS
    feats, traj = mdgen.analysis.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=True, cossin=False)
    if args.truncate: traj = traj[:args.truncate]
    feats, ref = mdgen.analysis.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True, cossin=False)

    out['features'] = feats.describe()

    out['JSD'] = {}
    for i, feat in enumerate(feats.describe()):
        ref_p = np.histogram(ref[:,i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj[:,i], range=(-np.pi, np.pi), bins=100)[0]
        out['JSD'][feat] = jensenshannon(ref_p, traj_p)

    for i in [1,3]:
        ref_p = np.histogram2d(*ref[:,i:i+2].T, range=((-np.pi, np.pi),(-np.pi,np.pi)), bins=50)[0]
        traj_p = np.histogram2d(*traj[:,i:i+2].T, range=((-np.pi, np.pi),(-np.pi,np.pi)), bins=50)[0]
        out['JSD']['|'.join(feats.describe()[i:i+2])] = jensenshannon(ref_p.flatten(), traj_p.flatten())

    ############ Torsion decorrelations
    if args.no_decorr:
        pass
    else:
        out['md_decorrelation'] = {}
        for i, feat in enumerate(feats.describe()):

            autocorr = acovf(np.sin(ref[:,i]), demean=False, adjusted=True, nlag=100000) + acovf(np.cos(ref[:,i]), demean=False, adjusted=True, nlag=100000)
            baseline = np.sin(ref[:,i]).mean()**2 + np.cos(ref[:,i]).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if args.plot:
                if 'PHI' in feat or 'PSI' in feat:
                    axs[0,1].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
                else:
                    axs[0,2].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])

            out['md_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)

        if args.plot:
            axs[0,1].set_title('Backbone decorrelation')
            axs[0,2].set_title('Sidechain decorrelation')
            axs[0,1].set_xscale('log')
            axs[0,2].set_xscale('log')

        out['our_decorrelation'] = {}
        for i, feat in enumerate(feats.describe()):

            autocorr = acovf(np.sin(traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000) + acovf(np.cos(traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000)
            baseline = np.sin(traj[:,i]).mean()**2 + np.cos(traj[:,i]).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if args.plot:
                if 'PHI' in feat or 'PSI' in feat:
                    axs[1,1].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
                else:
                    axs[1,2].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])

            out['our_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)

        if args.plot:
            axs[1,1].set_title('Backbone decorrelation')
            axs[1,2].set_title('Sidechain decorrelation')
            axs[1,1].set_xscale('log')
            axs[1,2].set_xscale('log')

    ####### TICA #############
    feats, traj = mdgen.analysis.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=True, cossin=True)
    if args.truncate: traj = traj[:args.truncate]
    feats, ref = mdgen.analysis.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True, cossin=True)

    tica, _ = mdgen.analysis.get_tica(ref)
    ref_tica = tica.transform(ref)
    traj_tica = tica.transform(traj)

    tica_0_min = min(ref_tica[:,0].min(), traj_tica[:,0].min())
    tica_0_max = max(ref_tica[:,0].max(), traj_tica[:,0].max())

    tica_1_min = min(ref_tica[:,1].min(), traj_tica[:,1].min())
    tica_1_max = max(ref_tica[:,1].max(), traj_tica[:,1].max())

    ref_p = np.histogram(ref_tica[:,0], range=(tica_0_min, tica_0_max), bins=100)[0]
    traj_p = np.histogram(traj_tica[:,0], range=(tica_0_min, tica_0_max), bins=100)[0]
    out['JSD']['TICA-0'] = jensenshannon(ref_p, traj_p)

    ref_p = np.histogram2d(*ref_tica[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]
    traj_p = np.histogram2d(*traj_tica[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]
    out['JSD']['TICA-0,1'] = jensenshannon(ref_p.flatten(), traj_p.flatten())

    #### 1,0, 1,1 TICA FES
    if args.plot:
        pyemma.plots.plot_free_energy(*ref_tica[::100, :2].T, ax=axs[2,0], cbar=False)
        pyemma.plots.plot_free_energy(*traj_tica[:, :2].T, ax=axs[2,1], cbar=False)
        axs[2,0].set_title('TICA FES (MD)')
        axs[2,1].set_title('TICA FES (ours)')


    ####### TICA decorrelation ########
    if args.no_decorr:
        pass
    else:
        # x, adjusted=False, demean=True, fft=True, missing='none', nlag=None
        autocorr = acovf(ref_tica[:,0], nlag=100000, adjusted=True, demean=False)
        out['md_decorrelation']['tica'] = autocorr.astype(np.float16)
        if args.plot:
            axs[0,3].plot(autocorr)
            axs[0,3].set_title('MD TICA')


        autocorr = acovf(traj_tica[:,0], nlag=1 if args.ito else 1000, adjusted=True, demean=False)
        out['our_decorrelation']['tica'] = autocorr.astype(np.float16)
        if args.plot:
            axs[1,3].plot(autocorr)
            axs[1,3].set_title('Traj TICA')

    ###### Markov state model stuff #################
    if not args.no_msm:
        kmeans, ref_kmeans = mdgen.analysis.get_kmeans(tica.transform(ref))
        try:
            msm, pcca, cmsm = mdgen.analysis.get_msm(ref_kmeans, nstates=10)

            out['kmeans'] = kmeans
            out['msm'] = msm
            out['pcca'] = pcca
            out['cmsm'] = cmsm

            traj_discrete = mdgen.analysis.discretize(tica.transform(traj), kmeans, msm)
            ref_discrete = mdgen.analysis.discretize(tica.transform(ref), kmeans, msm)
            out['traj_metastable_probs'] = (traj_discrete == np.arange(10)[:,None]).mean(1)
            out['ref_metastable_probs'] = (ref_discrete == np.arange(10)[:,None]).mean(1)
            #########

            msm_transition_matrix = np.eye(10)
            for a, i in enumerate(cmsm.active_set):
                for b, j in enumerate(cmsm.active_set):
                    msm_transition_matrix[i,j] = cmsm.transition_matrix[a,b]

            out['msm_transition_matrix'] = msm_transition_matrix
            out['pcca_pi'] = pcca._pi_coarse

            msm_pi = np.zeros(10)
            msm_pi[cmsm.active_set] = cmsm.pi
            out['msm_pi'] = msm_pi

            if args.no_traj_msm:
                pass
            else:

                traj_msm = pyemma.msm.estimate_markov_model(traj_discrete, lag=args.msm_lag)
                out['traj_msm'] = traj_msm

                traj_transition_matrix = np.eye(10)
                for a, i in enumerate(traj_msm.active_set):
                    for b, j in enumerate(traj_msm.active_set):
                        traj_transition_matrix[i,j] = traj_msm.transition_matrix[a,b]
                out['traj_transition_matrix'] = traj_transition_matrix


                traj_pi = np.zeros(10)
                traj_pi[traj_msm.active_set] = traj_msm.pi
                out['traj_pi'] = traj_pi

        except Exception as e:
            print('ERROR', e, name, flush=True)

    if args.plot:
        fig.savefig(f'{args.pdbdir}/{name}.pdf')
        plt.close(fig)  # release figure memory; otherwise it leaks across iterations

    return name, out

if args.pdb_id:
    pdb_id = args.pdb_id
else:
    pdb_id = [nam.split('.')[0] for nam in os.listdir(args.pdbdir) if '.pdb' in nam and not '_traj' in nam]
pdb_id = [nam for nam in pdb_id if os.path.exists(f'{args.pdbdir}/{nam}.xtc')]
print('number of trajectories', len(pdb_id))


if args.num_workers > 1:
    # maxtasksperchild=1: each worker process processes exactly one peptide
    # then is recycled. This isolates per-peptide state corruption (notably
    # mdtraj's C-level file-descriptor state after a SIGALRM-induced
    # TimeoutError) so that a hung or failed peptide cannot poison the
    # worker for subsequent peptides. Slight startup overhead, but the
    # robustness is worth it for long-running analysis jobs.
    p = Pool(args.num_workers, maxtasksperchild=1)
    p.__enter__()
    # imap_unordered yields results as they complete (not in pdb_id order),
    # so a single slow/hung peptide doesn't head-of-line-block the tqdm bar.
    __map__ = p.imap_unordered
else:
    __map__ = map
out = dict(tqdm.tqdm(__map__(main, pdb_id), total=len(pdb_id)))
if args.num_workers > 1:
    p.__exit__(None, None, None)

if args.save:
    with open(f"{args.pdbdir}/{args.save_name}", 'wb') as f:
        f.write(pickle.dumps(out))
