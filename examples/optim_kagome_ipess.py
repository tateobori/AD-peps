import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import kagomej1
from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod_ipess import optimize_state
from collections import OrderedDict
import unittest
import logging
log = logging.getLogger(__name__)

torch.set_printoptions(edgeitems=1000)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=1., help="next nearest-neighbour coupling")
parser.add_argument("--tiling", default="1x1", help="tiling of the lattice")
parser.add_argument("--file", default="pess", help="pess file name")
args, unknown_args = parser.parse_known_args()

def initial_ipess(model, bond_dim):

    A = torch.rand((bond_dim, model.phys_dim, bond_dim),\
        dtype=cfg.global_args.dtype,device=cfg.global_args.device)

    B = torch.rand((bond_dim, model.phys_dim, bond_dim),\
        dtype=cfg.global_args.dtype,device=cfg.global_args.device)

    C = torch.rand((bond_dim, model.phys_dim, bond_dim),\
        dtype=cfg.global_args.dtype,device=cfg.global_args.device)

    R_up = torch.rand((bond_dim, bond_dim, bond_dim),\
        dtype=cfg.global_args.dtype,device=cfg.global_args.device)

    R_down = torch.rand((bond_dim, bond_dim, bond_dim),\
        dtype=cfg.global_args.dtype,device=cfg.global_args.device)

    A = A/torch.max(torch.abs(A))
    B = B/torch.max(torch.abs(B))
    C = C/torch.max(torch.abs(C))
    R_up = R_up/torch.max(torch.abs(R_up))
    R_down = R_down/torch.max(torch.abs(R_down))

    dict1= {0: A}
    dict1[1] = B
    dict1[2] = C
    dict1[3] = R_up
    dict1[4] = R_down

    pess = OrderedDict(dict1).values()

    for par in pess: par.requires_grad_(True)

    
    CR_u = torch.tensordot(C.clone(), R_up.clone(), ([0],[1]))
    BR_d = torch.tensordot(B.clone(), R_down.clone(), ([2],[1]))
    ABR_d = torch.tensordot(A.clone(), BR_d.clone(), ([2],[2]))
    T0 = torch.tensordot(CR_u, ABR_d, ([1],[4]))
    #T0 = T0.permute(4,6,0,5,3,1,2)
    T0 = T0.permute(4,6,0,3,1,2,5)
    """
    AR_d = torch.tensordot(A, R_down, ([2],[0]))
    CR_up = torch.tensordot(C, R_up, ([2],[0]))
    BCR_up = torch.tensordot(B, CR_up, ([0],[3]))
    T0 = torch.tensordot(AR_d, BCR_up, ([2],[1]))
    T0 = T0.permute(1,3,5,4,0,2,6)
    """
    T0 = T0.contiguous().view(T0.size()[0]*T0.size()[1]*T0.size()[2], T0.size()[3], T0.size()[4], T0.size()[5], T0.size()[6])
    T0 = T0/torch.max(torch.abs(T0))



    return T0, pess

def combine_ipess_into_ipeps(A, B, C, R_up, R_down):

    """
    A = A/torch.max(torch.abs(A))
    B = B/torch.max(torch.abs(B))
    C = C/torch.max(torch.abs(C))
    R_up = R_up/torch.max(torch.abs(R_up))
    R_down = R_down/torch.max(torch.abs(R_down))
    """

    CR_u = torch.tensordot(C.clone(), R_up.clone(), ([0],[1]))
    BR_d = torch.tensordot(B.clone(), R_down.clone(), ([2],[1]))
    ABR_d = torch.tensordot(A.clone(), BR_d.clone(), ([2],[2]))

    T1 = torch.tensordot(CR_u, ABR_d, ([1],[4]))
    #T1 = T1.permute(4,6,0,5,3,1,2)
    T1 = T1.permute(4,6,0,3,1,2,5)
    T1 = T1.contiguous().view(T1.size()[0]*T1.size()[1]*T1.size()[2], T1.size()[3], T1.size()[4], T1.size()[5], T1.size()[6])
    T1 = T1/torch.max(torch.abs(T1))

    return T1

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

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

    elif args.tiling == "3x3":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 3) % 3
            vy = (coord[1] + abs(coord[1]) * 3) % 3
            return (vx, vy)

    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"2x2 or 1x1")

    if args.instate!=None:

        test = read_ipess(args.instate, vertexToSite=lattice_to_site)
        test.add_noise(args.instate_noise)
        unit= []
        for t in test.sites.values():
            for par in t:
                #par.requires_grad_(True)
                unit.append(par)


        if args.tiling =="2x2":

            A1, B1, C1, R1_up, R1_down,\
            A2, B2, C2, R2_up, R2_down,\
            A3, B3, C3, R3_up, R3_down,\
            A4, B4, C4, R4_up, R4_down= unit

            i=0
            pess = OrderedDict()
            pess1 = OrderedDict()
            pess2 = OrderedDict()
            pess3 = OrderedDict()
            pess4 = OrderedDict()

            for par in unit:
                par.requires_grad_(True)

                if i<5: pess1[i]=par
                elif i<10: pess2[i-5]=par
                elif i<15: pess3[i-10]=par
                else:  pess4[i-15]=par
                i +=1

            pess[(0,0)]=pess1.values()
            pess[(1,0)]=pess2.values()
            pess[(0,1)]=pess3.values()
            pess[(1,1)]=pess4.values()

            pess=OrderedDict(pess).values()

            T1 = combine_ipess_into_ipeps(A1, B1, C1, R1_up, R1_down)
            T2 = combine_ipess_into_ipeps(A2, B2, C2, R2_up, R2_down)
            T3 = combine_ipess_into_ipeps(A3, B3, C3, R3_up, R3_down)
            T4 = combine_ipess_into_ipeps(A4, B4, C4, R4_up, R4_down)

            sites = {(0,0): T1}
            sites[(1,0)] = T2
            sites[(0,1)] = T3
            sites[(1,1)] = T4

            state = IPEPS(sites, vertexToSite=lattice_to_site)
        
     
        if args.tiling =="1x1":
            A, B, C, R_up, R_down = unit
            pess = OrderedDict()
            dict1= {0: A}
            dict1[1] = B
            dict1[2] = C
            dict1[3] = R_up
            dict1[4] = R_down
            pess[(0,0)] = OrderedDict(dict1).values()
            pess=OrderedDict(pess).values()

            CR_u = torch.tensordot(C.clone(), R_up.clone(), ([0],[1]))
            BR_d = torch.tensordot(B.clone(), R_down.clone(), ([2],[1]))
            ABR_d = torch.tensordot(A.clone(), BR_d.clone(), ([2],[2]))

            T1 = torch.tensordot(CR_u, ABR_d, ([1],[4]))
            #T1 = T1.permute(4,6,0,5,3,1,2)
            T1 = T1.permute(4,6,0,3,1,2,5)
            T1 = T1.contiguous().view(T1.size()[0]*T1.size()[1]*T1.size()[2], T1.size()[3], T1.size()[4], T1.size()[5], T1.size()[6])
            T1 = T1/torch.max(torch.abs(T1))

            sites = {(0,0): T1}
            state = IPEPS(sites, vertexToSite=lattice_to_site)
        
        #if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
           #state = extend_bond_dim(state, args.bond_dim)
    elif args.opt_resume is not None:
        if args.tiling == "2x2":
            state= IPEPS(dict(), lX=2, lY=2)
        elif args.tiling == "1x1":
            state= IPEPS(dict(), lX=1, lY=1)

        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim

        pess=OrderedDict()

        T1, pess1 = initial_ipess(model, bond_dim)
        pess[(0,0)]=pess1
        sites = {(0,0): T1}
        
        if args.tiling == "2x2":

            T2, pess2 = initial_ipess(model, bond_dim)
            T3, pess3 = initial_ipess(model, bond_dim)
            T4, pess4 = initial_ipess(model, bond_dim) 

            sites[(1,0)] = T2
            sites[(0,1)] = T3
            sites[(1,1)] = T4

            pess[(1,0)]=pess2
            pess[(0,1)]=pess3
            pess[(1,1)]=pess4



        if args.tiling == "3x3":

            T2, pess2 = initial_ipess(model, bond_dim)
            T3, pess3 = initial_ipess(model, bond_dim)
            T4, pess4 = initial_ipess(model, bond_dim) 
            T5, pess5 = initial_ipess(model, bond_dim) 
            T6, pess6 = initial_ipess(model, bond_dim) 
            T7, pess7 = initial_ipess(model, bond_dim) 
            T8, pess8 = initial_ipess(model, bond_dim)
            T9, pess9 = initial_ipess(model, bond_dim) 

            sites[(0,1)] = T2
            sites[(0,2)] = T3
            sites[(1,0)] = T4
            sites[(1,1)] = T5
            sites[(1,2)] = T6
            sites[(2,0)] = T7
            sites[(2,1)] = T8
            sites[(2,2)] = T9

            pess[(0,1)]=pess2
            pess[(0,2)]=pess3
            pess[(1,0)]=pess4
            pess[(1,1)]=pess5
            pess[(1,2)]=pess6
            pess[(2,0)]=pess7
            pess[(2,1)]=pess8
            pess[(2,2)]=pess9


        pess=pess.values()
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

    elif args.tiling == "3x3":
        energy_f=model.energy_2x2_9site

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
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(obs_labels)
    print([f"{v}" for v in obs_values])
    print(", ".join(["epoch","energy"]))
    print(", ".join([f"{-1}",f"{loss0}"]))
    #exit()

    def loss_fn(state, pess, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)


        # 0) combining ipess into the ipeps
        #for 1x1 tiling
        if args.tiling =="1x1":
            A, B, C, R_up, R_down = pess
            A = A/torch.max(torch.abs(A))
            B = B/torch.max(torch.abs(B))
            C = C/torch.max(torch.abs(C))
            R_up = R_up/torch.max(torch.abs(R_up))
            R_down = R_down/torch.max(torch.abs(R_down))

            unit = [A, B, C, R_up, R_down]
            unit = {(0,0): unit}
            unit = OrderedDict(unit)
            test = IPESS(unit, vertexToSite=lattice_to_site)
            test.write_to_file(args.file, normalize=True)

            T1 = combine_ipess_into_ipeps(A, B, C, R_up, R_down)

            sites = {(0,0): T1}
            state = IPEPS(sites, vertexToSite=lattice_to_site)
    
        
        if args.tiling =="2x2":
            A1, B1, C1, R1_up, R1_down,\
            A2, B2, C2, R2_up, R2_down,\
            A3, B3, C3, R3_up, R3_down,\
            A4, B4, C4, R4_up, R4_down = pess

            
            unit1 = [A1, B1, C1, R1_up, R1_down]
            unit2 = [A2, B2, C2, R2_up, R2_down]
            unit3 = [A3, B3, C3, R3_up, R3_down]
            unit4 = [A4, B4, C4, R4_up, R4_down]

            unit = {(0,0): unit1}
            unit[(0,1)] = unit2
            unit[(1,0)] = unit3
            unit[(1,1)] = unit4

            unit = OrderedDict(unit)
            test = IPESS(unit, vertexToSite=lattice_to_site)
            test.write_to_file(args.file, normalize=True)

            T1 = combine_ipess_into_ipeps(A1, B1, C1, R1_up, R1_down)
            T2 = combine_ipess_into_ipeps(A2, B2, C2, R2_up, R2_down)
            T3 = combine_ipess_into_ipeps(A3, B3, C3, R3_up, R3_down)
            T4 = combine_ipess_into_ipeps(A4, B4, C4, R4_up, R4_down)

            sites = {(0,0): T1}
            sites[(1,0)] = T2
            sites[(0,1)] = T3
            sites[(1,1)] = T4
            state = IPEPS(sites, vertexToSite=lattice_to_site)


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
            #print(state.get_parameters())
            print(", ".join([f"{epoch}",f"{loss}"]))
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))

    # optimize
    print("Start optimization")
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn, parameters=pess)

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