import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import corrf
from math import sqrt
import itertools

class COUPLEDLADDERS():
    def __init__(self, alpha=0.0, global_args=cfg.global_args):
        r"""
        :param alpha: nearest-neighbour interaction
        :param global_args: global configuration
        :type alpha: float
        :type global_args: GLOBALARGS

        Build Hamiltonian of spin-1/2 coupled ladders

        .. math:: H = \sum_{i=(x,y)} h2_{i,i+\vec{x}} + \sum_{i=(x,2y)} h2_{i,i+\vec{y}}
                   + \alpha \sum_{i=(x,2y+1)} h2_{i,i+\vec{y}}

        on the square lattice. The spin-1/2 ladders are coupled with strength :math:`\alpha`::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._a__a__a__a_...
            ..._|__|__|__|_...
            ..._a__a__a__a_...   
            ..._|__|__|__|_...   
                :  :  :  :      (a = \alpha) 

        where

        * :math:`h2_{ij} = \mathbf{S}_i.\mathbf{S}_j` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.alpha=alpha

        self.h2 = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        expr_kron = 'ij,ab->iajb'
        SS = torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        return SS

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_1x1c4v(self,ipeps):
        pass

    def energy_2x2(self,ipeps):
        pass

    # assuming reduced density matrix of 2x2 cluster with indexing of DOFs
    # as follows rdm2x2=rdm2x2(s0,s1,s2,s3;s0',s1',s2',s3')
    def energy_2x2(self,rdm2x2):
        energy = torch.einsum('ijklabkl,ijab',rdm2x2,self.h)
        return energy

    def energy_2x1_1x2(self,state,env):
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

        Taking the reduced density matrix :math:`\rho_{2x1}` (:math:`\rho_{1x2}`) 
        of 2x1 (1x2) cluster given by :py:func:`rdm.rdm2x1` (:py:func:`rdm.rdm1x2`) 
        with indexing of sites as follows :math:`s_0,s_1;s'_0,s'_1` for both types
        of density matrices::

            rdm2x1   rdm1x2

            s0--s1   s0
                     |
                     s1

        and without assuming any symmetry on the indices of individual tensors a following
        set of terms has to be evaluated in order to compute energy-per-site::

               0       0       0
            1--A--3 1--B--3 1--A--3
               2       2       2
               0       0       0
            1--C--3 1--D--3 1--C--3
               2       2       2             A--3 1--B,      A  B  C  D
               0       0                     B--3 1--A,      2  2  2  2
            1--A--3 1--B--3                  C--3 1--D,      0  0  0  0
               2       2             , terms D--3 1--C, and  C, D, A, B
        """
        energy=0.
        for coord,site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord,state,env)
            rdm1x2 = rdm.rdm1x2(coord,state,env)
            ss = torch.einsum('ijab,ijab',rdm2x1,self.h2)
            energy += ss
            if coord[1] % 2 == 0:
                ss = torch.einsum('ijab,ijab',rdm1x2,self.h2)
            else:
                ss = torch.einsum('ijab,ijab',rdm1x2,self.alpha * self.h2)
            energy += ss

        # return energy-per-site
        energy_per_site=energy/len(state.sites.items())
        return energy_per_site

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
            4. :math:`\mathbf{S}_i.\mathbf{S}_j` for all non-equivalent nearest neighbour
               bonds

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
        obs= dict({"avg_m": 0.})
        with torch.no_grad():
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())
        
            for coord,site in state.sites.items():
                rdm2x1 = rdm.rdm2x1(coord,state,env)
                rdm1x2 = rdm.rdm1x2(coord,state,env)
                obs[f"SS2x1{coord}"]= torch.einsum('ijab,ijab',rdm2x1,self.h2)
                obs[f"SS1x2{coord}"]= torch.einsum('ijab,ijab',rdm1x2,self.h2)

        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
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

    def eval_corrf_DD_H(self,coord,direction,state,env,dist,verbosity=0):
        # function generating properly S.S operator
        def _gen_op(r):
            return self.h2
        
        D0DR= corrf.corrf_2sOH2sOH_E1(coord, direction, state, env, self.h2, _gen_op,\
            dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res

    def eval_corrf_DD_V(self,coord,direction,state,env,dist,verbosity=0):
        r"""
        Evaluates correlation functions of two vertical dimers
        DD_v(r)= <(S(0).S(y))(S(r*x).S(y+r*x))>
             or= <(S(0).S(x))(S(r*y).S(x+r*y))> 
        """
        # function generating properly S.S operator
        def _gen_op(r):
            return self.h2
        
        D0DR= corrf.corrf_2sOV2sOV_E2(coord, direction, state, env, self.h2, _gen_op,\
            dist, verbosity=verbosity)
        
        res= dict({"dd": D0DR})
        return res

class COUPLEDLADDERS_D2_BIPARTITE():
    def __init__(self, alpha=0.0, global_args=cfg.global_args):
        r"""
        :param alpha: nearest-neighbour interaction
        :param global_args: global configuration
        :type alpha: float
        :type global_args: GLOBALARGS

        Build Hamiltonian of spin-1/2 coupled ladders

        .. math:: H = \sum_{i=(x,y)} h2_{i,i+\vec{x}} + \sum_{i=(x,2y)} h2_{i,i+\vec{y}}
                   + \alpha \sum_{i=(x,2y+1)} h2_{i,i+\vec{y}}

        on the square lattice. The spin-1/2 ladders are coupled with strength :math:`\alpha`::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._a__a__a__a_...
            ..._|__|__|__|_...
            ..._a__a__a__a_...   
            ..._|__|__|__|_...   
                :  :  :  :      (a = \alpha) 

        where

        * :math:`h2_{ij} = \mathbf{S}_i.\mathbf{S}_j` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.alpha=alpha

        self.h2, self.h2_rot = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        expr_kron = 'ij,ab->iajb'
        SS = torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        rot_op= s2.BP_rot()
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,SS,rot_op)
        return SS, SS_rot

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_2x1_1x2(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float
        
        We assume iPEPS with 1x2 unit cell containing two tensors A, B
        simple PBC tiling::

            A  Au A  Au
            Bu B  Bu B
            A  Au A  Au
            Bu B  Bu B
    
        where unitary "u" operates on every site of "odd" sublattice to realize
        AFM correlations.
        Taking the reduced density matrix :math:`\rho_{2x1}` (:math:`\rho_{1x2}`) 
        of 2x1 (1x2) cluster given by :py:func:`rdm.rdm2x1` (:py:func:`rdm.rdm1x2`) 
        with indexing of sites as follows :math:`s_0,s_1;s'_0,s'_1` for both types
        of density matrices::

            rdm2x1   rdm1x2

            s0--s1   s0
                     |
                     s1

        and without assuming any symmetry on the indices of individual tensors a following
        set of terms has to be evaluated in order to compute energy-per-site::

               0       0       0
            1--A--3 1--A--3 1--A--3
               2       2       2
               0       0       0
            1--B--3 1--B--3 1--B--3
               2       2       2                             A  B
               0       0                                     2  2
            1--A--3 1--A--3                  A--3 1--A,      0  0
               2       2             , terms B--3 1--B, and  B, A,
        """
        energy=0.
        for coord,site in state.sites.items():
        # for coord in [(0,0),(0,1),(1,0),(1,1)]:
            rdm2x1 = rdm.rdm2x1(coord,state,env)
            rdm1x2 = rdm.rdm1x2(coord,state,env)
            ss = torch.einsum('ijab,ijab',rdm2x1,self.h2_rot)
            energy += ss
            if coord[1] % 2 == 0:
                ss = torch.einsum('ijab,ijab',rdm1x2,self.h2_rot)
            else:
                ss = torch.einsum('ijab,jiba',rdm1x2,self.alpha * self.h2_rot)
            energy += ss

        # return energy-per-site
        energy_per_site=energy/len(state.sites.items())
        return energy_per_site

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
            4. :math:`\mathbf{S}_i.\mathbf{S}_j` for all non-equivalent nearest neighbour
               bonds

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
        obs= dict({"avg_m": 0.})
        with torch.no_grad():
            rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                if coord[1] % 2 == 0: rdm1x1= rot_op@rdm1x1@rot_op.t()

                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())
        
            # for coord,site in state.sites.items():
            for coord in [(0,0),(0,1),(1,0),(1,1)]:
                rdm2x1 = rdm.rdm2x1(coord,state,env)
                rdm1x2 = rdm.rdm1x2(coord,state,env)
                if (coord[1] % 2 == 0) ^ (coord[0] % 2 == 0):
                    obs[f"SS1x2{coord}"]= torch.einsum('ijab,ijab',rdm1x2,self.h2_rot)
                else:
                    obs[f"SS1x2{coord}"]= torch.einsum('ijab,jiba',rdm1x2,self.h2_rot)
                if (coord[0] % 2 == 0) ^ (coord[0] % 2 == 0):
                    obs[f"SS2x1{coord}"]= torch.einsum('ijab,ijab',rdm2x1,self.h2_rot)
                else:
                    obs[f"SS2x1{coord}"]= torch.einsum('ijab,jiba',rdm2x1,self.h2_rot)

        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in [(0,0),(0,1),(1,0),(1,1)]]
        obs_labels += [f"SS1x2{coord}" for coord in [(0,0),(0,1),(1,0),(1,1)]]
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

    def eval_corrf_DD_H(self,coord,direction,state,env,dist,verbosity=0):
        # function generating properly S.S operator
        def _gen_op(r):
            return self.h2
        
        D0DR= corrf.corrf_2sOH2sOH_E1(coord, direction, state, env, self.h2, _gen_op,\
            dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res

    def eval_corrf_DD_V(self,coord,direction,state,env,dist,verbosity=0):
        r"""
        Evaluates correlation functions of two vertical dimers
        DD_v(r)= <(S(0).S(y))(S(r*x).S(y+r*x))>
             or= <(S(0).S(x))(S(r*y).S(x+r*y))> 
        """
        # function generating properly S.S operator
        def _gen_op(r):
            return self.h2
        
        D0DR= corrf.corrf_2sOV2sOV_E2(coord, direction, state, env, self.h2, _gen_op,\
            dist, verbosity=verbosity)
        
        res= dict({"dd": D0DR})
        return res