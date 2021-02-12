import context
import copy
import torch
import argparse
import config as cfg
from u1sym.ipeps_u1 import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from ctm.one_site_c4v import transferops_c4v
from models import j1j2
from optim.ad_optim_lbfgs_mod import optimize_state
import u1sym.sym_ten_parser as tenU1
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--u1_class", type=str, default="B", choices=["A", "B", "C", "D", "E"])
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--hz_stag", type=float, default=0., help="staggered mag. field")
parser.add_argument("--delta_zz", type=float, default=1., help="easy-axis (nearest-neighbour) anisotropy")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args= parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2, hz_stag=args.hz_stag, \
        delta_zz=args.delta_zz)
    energy_f= model.energy_1x1_lowmem

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_u1(args.instate, vertexToSite=None)
        assert len(state.coeffs)==1, "Not a 1-site ipeps"

        # TODO extending from smaller bond-dim to higher bond-dim is 
        # currently not possible
        
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.bond_dim in [2,3,4,5,6,7,8]:
            u1sym_t= tenU1.import_sym_tensors(2,args.bond_dim,"A_1",\
                infile=f"u1sym/D{args.bond_dim}_U1_{args.u1_class}.txt",\
                dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported --bond_dim= "+str(args.bond_dim))
        A= torch.zeros(len(u1sym_t), dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        coeffs = {(0,0): A}
        state= IPEPS_U1SYM(u1sym_t, coeffs)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        if args.bond_dim in [2,3,4,5,6,7,8]:
            u1sym_t= tenU1.import_sym_tensors(2, args.bond_dim, "A_1", \
                infile=f"u1sym/D{args.bond_dim}_U1_{args.u1_class}.txt", \
                dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported --bond_dim= "+str(args.bond_dim))

        A= torch.rand(len(u1sym_t), dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        A= A/torch.max(torch.abs(A))
        coeffs = {(0,0): A}
        state = IPEPS_U1SYM(u1sym_t, coeffs)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    @torch.no_grad()
    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
        history["rdm"]=rdm2x1
        history["log"].append(dist)
        if dist<ctm_args.ctm_conv_tol:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return True, history
        elif len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return False, history
        return False, history

    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)

    loss0 = energy_f(state, ctm_env, force_cpu=True)
    obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=True)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # build on-site tensors from su2sym components
        state.sites= state.build_onsite_tensors()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_obs= ctmrg_c4v.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        loss0 = energy_f(state, ctm_env_out, force_cpu=True)
        
        loc_ctm_args= copy.deepcopy(ctm_args)
        loc_ctm_args.ctm_max_iter= 1
        ctm_env_out, history1, t_ctm1, t_obs1= ctmrg_c4v.run(state, ctm_env_out, \
            ctm_args=loc_ctm_args)
        loss1 = energy_f(state, ctm_env_out, force_cpu=True)

        #loss=(loss0+loss1)/2
        loss= torch.max(loss0,loss1)

        return loss, ctm_env_out, history, t_ctm, t_obs

    def _to_json(l):
        re=[l[i,0].item() for i in range(l.size()[0])]
        im=[l[i,1].item() for i in range(l.size()[0])]
        return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
        obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=True)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

        if (not opt_context["line_search"]) and args.top_freq>0 and epoch%args.top_freq==0:
            coord_dir_pairs=[((0,0), (1,0))]
            for c,d in coord_dir_pairs:
                # transfer operator spectrum
                print(f"TOP spectrum(T)[{c},{d}] ",end="")
                l= transferops_c4v.get_Top_spec_c4v(args.top_n, state, ctm_env)
                print("TOP "+json.dumps(_to_json(l)))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_u1(outputstatefile)
    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state,ctm_env,force_cpu=True)
    obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=True)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=3
        args.chi=18
        args.opt_max_iter=3
        try:
            import scipy.sparse.linalg
            self.SCIPY= True
        except:
            print("Warning: Missing scipy. ARNOLDISVD is not available.")
            self.SCIPY= False

    # basic tests
    def test_opt_SYMEIG_LS(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        main()

    def test_opt_SYMEIG_LS_SYMARP(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        args.OPTARGS_line_search_svd_method="SYMARP"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_LS_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        main()
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_LS_SYMARP_gpu(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        args.OPTARGS_line_search_svd_method="SYMARP"
        main()