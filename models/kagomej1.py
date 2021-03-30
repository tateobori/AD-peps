import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import corrf
from ctm.one_site_c4v.env_c4v import ENV_C4V
from ctm.one_site_c4v import rdm_c4v
from ctm.one_site_c4v.rdm_c4v_specialized import rdm2x2_NNN_tiled,\
    rdm2x2_NN_tiled, rdm2x1_tiled
from ctm.one_site_c4v import corrf_c4v
from math import sqrt
import itertools

class KagomeJ1():
    def __init__(self, j1=1.0, j2=0.0, global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param global_args: global configuration
        :type j1: float
        :type j2: float
        :type global_args: GLOBALARGS

        Build Spin-1/2 :math:`J_1-J_2` Hamiltonian

        .. math:: H = J_1\sum_{<i,j>} h2_{ij} + J_2\sum_{<<i,j>>} h2_{ij}

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), and the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`)::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h2_{ij} = \mathbf{S_i}.\mathbf{S_j}` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        
        self.Ham= self.get_h()
        self.h2x2_down, self.h2x2_nn, self.h2x2_nnn= self.get_h_2x2()
        self.obs_ops= self.get_obs_ops()

    def get_h_another(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id3= torch.eye(8**3,dtype=self.dtype,device=self.device)
        id3= id3.view(8,8,8,8,8,8).contiguous()

        # h_on : on site hamiltonian in d=8
        expr_kron = 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        SS= SS.contiguous()
        expr_kron = 'ijab,kc->ijkabc'
        SSS= torch.einsum(expr_kron,SS,s2.I())
        SSS = SSS + SSS.permute(2,0,1,5,3,4) # -A-B- + -B-C- 
        h_on= SSS.view(8,8).contiguous() 
        h2x2_on= torch.einsum('ia,jklbcd->ijklabcd',h_on,id3)

        h2x2_on= h2x2_on + h2x2_on.permute(3,0,1,2,7,4,5,6) + h2x2_on.permute(2,3,0,1,6,7,4,5)\
            + h2x2_on.permute(1,2,3,0,5,6,7,4)## -A1- + -A2- + -A3- + -A4-

        h2x2_on= h2x2_on.contiguous()

        # h_x:   corresponds to h_ca and h_ba on up triangle
        # h_y:   corresponds to h_ac and h_bc on down triangle
        id4= torch.eye(16, dtype=self.dtype, device=self.device)
        id4= id4.view(2,2,2,2,2,2,2,2).contiguous()
        h_ca_up= torch.einsum('ijab,klmncdef->klijmncdabef', SS, id4)
        h_ba_up= torch.einsum('ijab,klmncdef->kiljmncadbef', SS, id4)
        h_ac_down= torch.einsum('ijab,klmncdef->iklmnjacdefb', SS, id4)
        h_bc_down= torch.einsum('ijab,klmncdef->kilmnjcadefb', SS, id4)

        h_x= h_ca_up.contiguous().view(8,8,8,8) + h_ba_up.contiguous().view(8,8,8,8)
        h_y= h_ac_down.contiguous().view(8,8,8,8) + h_bc_down.contiguous().view(8,8,8,8)

        id2= torch.eye(8**2, dtype=self.dtype, device=self.device)
        id2= id2.view(8,8,8,8).contiguous()

        h2x2_x= torch.einsum('ijab,klcd->ijklabcd',h_x,id2)
        h2x2_x= h2x2_x + h2x2_x.permute(2,3,0,1,6,7,4,5)

        h2x2_y= torch.einsum('ijab,klcd->ikjlacbd',h_y,id2)
        h2x2_y= h2x2_y + h2x2_y.permute(1,0,3,2,5,4,7,6)

        h2x2_nn= h2x2_x.contiguous() + h2x2_y.contiguous()

        Ham = (h2x2_on/4.0 + h2x2_nn/2.0)/3.0
        #Ham = h2x2_nn/2.0
        return Ham

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id3= torch.eye(8**3,dtype=self.dtype,device=self.device)
        id3= id3.view(8,8,8,8,8,8).contiguous()

        # h_up : on site hamiltonian in d=8
        expr_kron = 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        SS= SS.contiguous()
        expr_kron = 'ijab,kc->ijkabc'
        SSS= torch.einsum(expr_kron,SS,s2.I())
        SSS = SSS + SSS.permute(2,0,1,5,3,4) + SSS.permute(1,2,0,4,5,3) # -A-B- + -B-C- + -C-A-
        h_down= SSS.view(8,8).contiguous() 
        h2x2_down= torch.einsum('ia,jklbcd->ijklabcd',h_down,id3)

        h2x2_down= h2x2_down + h2x2_down.permute(3,0,1,2,7,4,5,6) + h2x2_down.permute(2,3,0,1,6,7,4,5)\
            + h2x2_down.permute(1,2,3,0,5,6,7,4)## -A1- + -A2- + -A3- + -A4-

        h2x2_down= h2x2_down.contiguous()

        # h_down : made of 3 kinds of bonds
        # h_x:   corresponds to h_bc terms on up triangle
        # h_y:   corresponds to h_ca terms on up triangle
        # h_nnn: correspomds to h_ba terms on up triangle
        id4= torch.eye(16, dtype=self.dtype, device=self.device)
        id4= id4.view(2,2,2,2,2,2,2,2).contiguous()
        h_bc= torch.einsum('ijab,klmncdef->kilmnjcadefb', SS, id4)
        h_ca= torch.einsum('ijab,klmncdef->klijmncdabef', SS, id4)
        h_ba= torch.einsum('ijab,klmncdef->kiljmncadbef', SS, id4)

        h_x= h_bc.contiguous().view(8,8,8,8)
        h_y= h_ca.contiguous().view(8,8,8,8)
        h_nnn= h_ba.contiguous().view(8,8,8,8)


        id2= torch.eye(8**2, dtype=self.dtype, device=self.device)
        id2= id2.view(8,8,8,8).contiguous()

        h2x2_x= torch.einsum('ijab,klcd->ijklabcd',h_x,id2)
        #h2x2_x= h2x2_x + h2x2_x.permute(2,3,0,1,6,7,4,5)

        h2x2_y= torch.einsum('ijab,klcd->ikjlacbd',h_y,id2)
        #h2x2_y= h2x2_y + h2x2_y.permute(1,0,3,2,5,4,7,6)

        h2x2_nn= h2x2_x.contiguous() + h2x2_y.contiguous()

        h2x2_nnn= torch.einsum('ijab,klcd->ikljacdb',h_nnn,id2)
        h2x2_nnn= h2x2_nnn.contiguous()

        Ham = (h2x2_down/4.0 + h2x2_nn + h2x2_nnn)/3.0
        #Ham = h2x2_nnn
        return Ham

    def get_h_2x2(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id3= torch.eye(8**3,dtype=self.dtype,device=self.device)
        id3= id3.view(8,8,8,8,8,8).contiguous()

        # h_up : on site hamiltonian in d=8
        expr_kron = 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        SS= SS.contiguous()
        expr_kron = 'ijab,kc->ijkabc'
        SSS= torch.einsum(expr_kron,SS,s2.I())
        SSS = SSS + SSS.permute(2,0,1,5,3,4) + SSS.permute(1,2,0,4,5,3) # -A-B- + -B-C- + -C-A-
        h_down= SSS.view(8,8).contiguous() 
        h2x2_down= torch.einsum('ia,jklbcd->ijklabcd',h_down,id3)

        h2x2_down= h2x2_down + h2x2_down.permute(3,0,1,2,7,4,5,6) + h2x2_down.permute(2,3,0,1,6,7,4,5)\
            + h2x2_down.permute(1,2,3,0,5,6,7,4)## -A1- + -A2- + -A3- + -A4-

        h2x2_down= h2x2_down.contiguous()

        # h_down : made of 3 kinds of bonds
        # h_x:   corresponds to h_bc terms on up triangle
        # h_y:   corresponds to h_ca terms on up triangle
        # h_nnn: correspomds to h_ba terms on up triangle
        id4= torch.eye(16, dtype=self.dtype, device=self.device)
        id4= id4.view(2,2,2,2,2,2,2,2).contiguous()
        h_bc= torch.einsum('ijab,klmncdef->kilmnjcadefb', SS, id4)
        h_ca= torch.einsum('ijab,klmncdef->klijmncdabef', SS, id4)
        h_ba= torch.einsum('ijab,klmncdef->kiljmncadbef', SS, id4)

        h_x= h_bc.contiguous().view(8,8,8,8)
        h_y= h_ca.contiguous().view(8,8,8,8)
        h_nnn= h_ba.contiguous().view(8,8,8,8)

        id2= torch.eye(8**2, dtype=self.dtype, device=self.device)
        id2= id2.view(8,8,8,8).contiguous()

        h2x2_x= torch.einsum('ijab,klcd->ijklabcd',h_x,id2)
        h2x2_x= h2x2_x + h2x2_x.permute(2,3,0,1,6,7,4,5)

        h2x2_y= torch.einsum('ijab,klcd->ikjlacbd',h_y,id2)
        h2x2_y= h2x2_y + h2x2_y.permute(1,0,3,2,5,4,7,6)

        h2x2_nn= h2x2_x.contiguous() + h2x2_y.contiguous()

        h2x2_nnn= torch.einsum('ijab,klcd->ikljacdb',h_nnn,id2)
        h2x2_nnn= h2x2_nnn.contiguous()


        return h2x2_down, h2x2_nn, h2x2_nnn

    def get_obs_ops(self):
        obs_ops = dict()
        expr_kron1 = 'ij,ab->iajb'
        expr_kron2 = 'ijab,kc->ijkabc'
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz_A"]= torch.einsum( expr_kron2, torch.einsum(expr_kron1,s2.SZ(),s2.I()), s2.I()).view(8,8).contiguous() 
        obs_ops["sp_A"]= torch.einsum( expr_kron2, torch.einsum(expr_kron1,s2.SP(),s2.I()), s2.I()).view(8,8).contiguous() 
        obs_ops["sm_A"]= torch.einsum( expr_kron2, torch.einsum(expr_kron1,s2.SM(),s2.I()), s2.I()).view(8,8).contiguous() 

        obs_ops["sz_B"]= torch.einsum( expr_kron2, torch.einsum(expr_kron1,s2.I(),s2.SZ()), s2.I()).view(8,8).contiguous() 
        obs_ops["sp_B"]= torch.einsum( expr_kron2, torch.einsum(expr_kron1,s2.I(),s2.SP()), s2.I()).view(8,8).contiguous() 
        obs_ops["sm_B"]= torch.einsum( expr_kron2, torch.einsum(expr_kron1,s2.I(),s2.SM()), s2.I()).view(8,8).contiguous() 

        obs_ops["sz_C"]= torch.einsum( expr_kron2, torch.einsum(expr_kron1,s2.I(),s2.I()), s2.SZ()).view(8,8).contiguous() 
        obs_ops["sp_C"]= torch.einsum( expr_kron2, torch.einsum(expr_kron1,s2.I(),s2.I()), s2.SP()).view(8,8).contiguous() 
        obs_ops["sm_C"]= torch.einsum( expr_kron2, torch.einsum(expr_kron1,s2.I(),s2.I()), s2.SM()).view(8,8).contiguous() 


        return obs_ops

    def energy_2x2_1site(self,state,env):

        rdm2x2= rdm.rdm2x2((0,0),state,env)
        energy_per_site= torch.einsum("ijklabcd,ijklabcd", rdm2x2, self.Ham)

        return energy_per_site

    def energy_2x2_4site(self,state,env):
        r"""

        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        We assume iPEPS with 2x2 unit cell containing four tensors A, B, C, and D with
        simple PBC tiling::

            A B A B
            C D C D
            A B A B
            C D C D
    
        Taking the reduced density matrix :math:`\rho_{2x2}` of 2x2 cluster given by 
        :py:func:`ctm.generic.rdm.rdm2x2` with indexing of sites as follows 
        :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::
        
            s0--s1
            |   |
            s2--s3

        and without assuming any symmetry on the indices of the individual tensors a set
        of four :math:`\rho_{2x2}`'s are needed over which :math:`h2` operators 
        for the nearest and next-neaerest neighbour pairs are evaluated::  

            A3--1B   B3--1A   C3--1D   D3--1C
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            C3--1D & D3--1C & A3--1B & B3--1A
        """
        energy_down=0
        energy_nn=0
        energy_nnn=0
        for coord in state.sites.keys():
            tmp_rdm= rdm.rdm2x2(coord,state,env)
            energy_down += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_down)
            energy_nn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nn)
            energy_nnn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nnn)
        energy_down_tri = self.j1*energy_down/16.0
        energy_up_tri = self.j1*energy_nn/16.0*2.0 + self.j1*energy_nnn/4.0

        return (energy_up_tri + energy_down_tri)/3.0

    def energy_2x2_9site(self,state,env):
        r"""

        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        We assume iPEPS with 2x2 unit cell containing four tensors A, B, C, and D with
        simple PBC tiling::

            A B A B
            C D C D
            A B A B
            C D C D
    
        Taking the reduced density matrix :math:`\rho_{2x2}` of 2x2 cluster given by 
        :py:func:`ctm.generic.rdm.rdm2x2` with indexing of sites as follows 
        :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::
        
            s0--s1
            |   |
            s2--s3

        and without assuming any symmetry on the indices of the individual tensors a set
        of four :math:`\rho_{2x2}`'s are needed over which :math:`h2` operators 
        for the nearest and next-neaerest neighbour pairs are evaluated::  

            A3--1B   B3--1A   C3--1D   D3--1C
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            C3--1D & D3--1C & A3--1B & B3--1A
        """
        energy_down=0
        energy_nn=0
        energy_nnn=0
        for coord in state.sites.keys():
            tmp_rdm= rdm.rdm2x2(coord,state,env)
            energy_down += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_down)
            energy_nn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nn)
            energy_nnn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nnn)
        energy_down_tri = self.j1*energy_down/36.0
        energy_up_tri = self.j1*energy_nn/36.0*2.0 + self.j1*energy_nnn/9.0

        return (energy_up_tri + energy_down_tri)/3.0

    def eval_obs(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. average magnetization over the unit cell,
            2. magnetization for each site in the unit cell
            3. :math:`\langle S^z \rangle,\ \langle S^+ \rangle,\ \langle S^- \rangle` 
               for each site in the unit cell

        where the on-site magnetization is defined as
        
        .. math::
            
            \begin{align*}
            m &= \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }
            =\sqrt{\langle S^z \rangle^2+1/4(\langle S^+ \rangle+\langle S^- 
            \rangle)^2 -1/4(\langle S^+\rangle-\langle S^-\rangle)^2} \\
              &=\sqrt{\langle S^z \rangle^2 + 1/2\langle S^+ \rangle \langle S^- \rangle)}
            \end{align*}

        Usual spin components can be obtained through the following relations
        
        .. math::
            
            \begin{align*}
            S^+ &=S^x+iS^y               & S^x &= 1/2(S^+ + S^-)\\
            S^- &=S^x-iS^y\ \Rightarrow\ & S^y &=-i/2(S^+ - S^-)
            \end{align*}
        """
        # TODO optimize/unify ?
        # expect "list" of (observable label, value) pairs ?
        obs= dict()
        with torch.no_grad():
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}_A"]= sqrt(abs(obs[f"sz_A{coord}"]**2 + obs[f"sp_A{coord}"]*obs[f"sm_A{coord}"]))
                obs[f"m{coord}_B"]= sqrt(abs(obs[f"sz_B{coord}"]**2 + obs[f"sp_B{coord}"]*obs[f"sm_B{coord}"]))
                obs[f"m{coord}_C"]= sqrt(abs(obs[f"sz_C{coord}"]**2 + obs[f"sp_C{coord}"]*obs[f"sm_C{coord}"]))
                obs[f"sx_A{coord}"]=0.5 * ( obs[f"sp_A{coord}"] + obs[f"sm_A{coord}"] )
                obs[f"sx_B{coord}"]=0.5 * ( obs[f"sp_B{coord}"] + obs[f"sm_B{coord}"] )
                obs[f"sx_C{coord}"]=0.5 * ( obs[f"sp_C{coord}"] + obs[f"sm_C{coord}"] )

        # prepare list with labels and values
        obs_labels=[f"m{coord}_A" for coord in state.sites.keys()] + [f"m{coord}_B" for coord in state.sites.keys()] + [f"m{coord}_C" for coord in state.sites.keys()] \
            +[f"sz_A{coord}" for coord in state.sites.keys()] + [f"sz_B{coord}" for coord in state.sites.keys()] + [f"sz_C{coord}" for coord in state.sites.keys()] \
            +[f"sx_A{coord}" for coord in state.sites.keys()] + [f"sx_B{coord}" for coord in state.sites.keys()] + [f"sx_C{coord}" for coord in state.sites.keys()]

        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,coord,direction,state,env,dist):
   
        # function allowing for additional site-dependent conjugation of op
        def conjugate_op(op):
            #rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            rot_op= torch.eye(self.phys_dim, dtype=self.dtype, device=self.device)
            op_0= op
            op_rot= torch.einsum('ki,kl,lj->ij',rot_op,op_0,rot_op)
            def _gen_op(r):
                #return op_rot if r%2==0 else op_0
                return op_0
            return _gen_op

        op_sx= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        op_isy= -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"]) 

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, self.obs_ops["sz"], \
            conjugate_op(self.obs_ops["sz"]), dist)
        Sx0sxR= corrf.corrf_1sO1sO(coord,direction,state,env, op_sx, conjugate_op(op_sx), dist)
        nSy0SyR= corrf.corrf_1sO1sO(coord,direction,state,env, op_isy, conjugate_op(op_isy), dist)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res  
    
