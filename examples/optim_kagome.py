import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import kagomej1
from torchviz import make_dot
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--tiling", default="1x1", help="tiling of the lattice")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model = kagomej1.KagomeJ1(j1=args.j1, j2=args.j2)

    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling == "2x2":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 2) % 2
            return (vx, vy)

    elif args.tiling == "1x1":
        def lattice_to_site(coord):
            return (0, 0)

    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"3x3")

    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "2x2":
            state= IPEPS(dict(), lX=2, lY=2)
        elif args.tiling == "1x1":
            state= IPEPS(dict(), lX=1, lY=1)

        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A = torch.rand((model.phys_dim**3, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype,device=cfg.global_args.device)

        A = A/torch.max(torch.abs(A))
        sites = {(0,0): A}

        if args.tiling == "2x2": 
            B = torch.rand((model.phys_dim**3, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            C = torch.rand((model.phys_dim**3, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            D = torch.rand((model.phys_dim**3, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)

            B = B/torch.max(torch.abs(B))
            C = C/torch.max(torch.abs(C))
            D = D/torch.max(torch.abs(D))

            sites[(1,0)] = B/torch.max(torch.abs(B))
            sites[(0,1)] = C/torch.max(torch.abs(C))
            sites[(1,1)] = D/torch.max(torch.abs(D))

        state = IPEPS(sites, vertexToSite=lattice_to_site)
        
        
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")


    print(state)
    # 2) select the "energy" function 
    if args.tiling == "1x1":
        energy_f=model.energy_2x2_1site

    elif args.tiling == "2x2":
        energy_f=model.energy_2x2_4site

    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = energy_f(state, env)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)

    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    loss0 = energy_f(state, ctm_env)
    #obs_values, obs_labels = model.eval_obs(state,ctm_env)
    #print(", ".join(["epoch","energy"]+obs_labels))
    #print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))
    print(", ".join(["epoch","energy"]))
    print(", ".join([f"{-1}",f"{loss0}"]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_energy, ctm_args=ctm_args)
        
        # 2) evaluate loss with the converged environment
        loss = energy_f(state, ctm_env_out)
        #print(loss)
        
        return (loss, ctm_env_out, *ctm_log)

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1]
            #obs_values, obs_labels = model.eval_obs(state,ctm_env)
            #print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))
            print(", ".join([f"{epoch}",f"{loss}"]))
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))

    # optimize
    #optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)
    print("Start optimization")
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, vertexToSite=state.vertexToSite)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(state,ctm_env)
    #obs_values, obs_labels = model.eval_obs(state,ctm_env)
    #print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))
    print("Enegy",opt_energy)  

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=2
        args.chi=16
        args.opt_max_iter=3
        try:
            import scipy.sparse.linalg
            self.SCIPY= True
        except:
            print("Warning: Missing scipy. Arnoldi methods not available.")
            self.SCIPY= False

    # basic tests
    def test_opt_GESDD_BIPARTITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    def test_opt_GESDD_BIPARTITE_LS_strong_wolfe(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        args.line_search="strong_wolfe"
        main()

    def test_opt_GESDD_BIPARTITE_LS_backtracking(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        args.line_search="backtracking"
        args.line_search_svd_method="ARP"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_BIPARTITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    def test_opt_GESDD_BIPARTITE_LS_backtracking_gpu(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        args.line_search="backtracking"
        args.line_search_svd_method="ARP"
        main()

    def test_opt_GESDD_4SITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_4SITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()