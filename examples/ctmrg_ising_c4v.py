import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import *
from groups.pg import make_c4v_symm
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v, transferops_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from models import ising
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--hx", type=float, default=0., help="transverse field")
parser.add_argument("--q", type=float, default=0, help="next nearest-neighbour coupling")
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"
    + "of transfer operator to compute")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"
    + " during CTM convergence")
args, unknown_args= parser.parse_known_args()

def main():
    # 0) parse command line arguments and configure simulation parameters
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model = ising.ISING_C4V(hx=args.hx, q=args.q)
    energy_f= model.energy_1x1_nn if args.q==0 else model.energy_1x1_plaqette

    # 1) initialize an ipeps - read from file or create a random one
    if args.instate!=None:
        state = read_ipeps_c4v(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
        state.sites[(0,0)]= state.sites[(0,0)]/torch.max(torch.abs(state.sites[(0,0)]))
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype,device=cfg.global_args.device)
        A= make_c4v_symm(A)
        A= A/torch.max(torch.abs(A))

        state = IPEPS_C4V(A)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    # 2) define convergence criterion for CTM algorithm. This function is to be 
    #    invoked at every CTM step. We also use it to evaluate observables of interest 
    #    during the course of CTM
    # 2a) convergence criterion based on on-site energy
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history=[]
            
            e_curr = energy_f(state, env)
            history.append(e_curr.item())

            if args.obs_freq>0 and \
                (len(history)%args.obs_freq==0 or (len(history)-1)%args.obs_freq==0):
                obs_values, obs_labels = model.eval_obs(state, env)
                print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
            else:
                print(", ".join([f"{len(history)}",f"{e_curr}"]))

            if len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol:
                log.info({"history_length": len(history), "history": history,
                    "final_multiplets": compute_multiplets(env)})
                return True, history
            elif len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history,
                    "final_multiplets": compute_multiplets(env)})
                return False, history
        return False, history

    # 2b) convergence criterion based on 2-site reduced density matrix 
    #     of nearest-neighbours
    def ctmrg_conv_rdm2x1(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history=dict({"log": []})
            rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
            dist= float('inf')
            if len(history["log"]) > 1:
                dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
            # log dist and observables
            if args.obs_freq>0 and \
                (len(history["log"])%args.obs_freq==0 or 
                (len(history["log"])-1)%args.obs_freq==0):
                e_curr = energy_f(state, env)
                obs_values, obs_labels = model.eval_obs(state, env)
                print(", ".join([f"{len(history['log'])}",f"{dist}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
            else:
                print(f"{len(history['log'])}, {dist}")
            # update history
            history["rdm"]=rdm2x1
            history["log"].append(dist)
            if dist<ctm_args.ctm_conv_tol:
                log.info({"history_length": len(history['log']), "history": history['log'],
                    "final_multiplets": compute_multiplets(env)})
                return True, history
            elif len(history['log']) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history['log']), "history": history['log'],
                    "final_multiplets": compute_multiplets(env)})
                return False, history
        return False, history

    # 3) initialize environment 
    ctm_env_init = ENV_C4V(args.chi, state)
    init_env(state, ctm_env_init)

    # 4) (optional) compute observables as given by initial environment 
    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    # 5) (main) execute CTM algorithm
    ctm_env_init, *ctm_log = ctmrg_c4v.run(state, ctm_env_init, conv_check=ctmrg_conv_rdm2x1)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # 7) ----- additional observables ---------------------------------------------
    # environment diagnostics
    print("\n\nspectrum(C)")
    u,s,v= torch.svd(ctm_env_init.C[ctm_env_init.keyC], compute_uv=False)
    for i in range(args.chi):
        print(f"{i} {s[i]}")

    # transfer operator spectrum
    print("\n\nspectrum(T)")
    l= transferops_c4v.get_Top_spec_c4v(args.top_n, state, ctm_env_init)
    for i in range(l.size()[0]):
        print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestCtmrg(unittest.TestCase):
    def setUp(self):
        args.instate=None 
        args.hx=3.0
        args.q=1.0
        args.bond_dim=2
        args.chi=16
        args.GLOBALARGS_device="cpu"

    # basic tests
    def test_ctmrg_SYMEIG(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_SYMEIG_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()