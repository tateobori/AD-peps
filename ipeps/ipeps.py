import torch
from collections import OrderedDict
import json
import itertools
import math
import config as cfg

# TODO drop constrain for aux bond dimension to be identical on 
# all bond indices

class IPEPS():
    def __init__(self, sites, vertexToSite=None, lX=None, lY=None, peps_args=cfg.peps_args,\
        global_args=cfg.global_args):
        r"""
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : torch.tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Member ``sites`` is a dictionary of non-equivalent on-site tensors
        indexed by tuple of coordinates (x,y) within the elementary unit cell.
        The index-position convetion for on-site tensors is defined as follows::

               u s 
               |/ 
            l--a--r  <=> a[s,u,l,d,r]
               |
               d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.
        Member ``vertexToSite`` is a mapping function from any vertex (x,y) on a square lattice
        passed in as tuple(int,int) to a corresponding vertex within elementary unit cell.
        
        On-site tensor of an IPEPS object ``wfc`` at vertex (x,y) is conveniently accessed 
        through the member function ``site``, which internally uses ``vertexToSite`` mapping::
            
            coord= (0,0)
            a_00= wfc.site(coord)

        By combining the appropriate ``vertexToSite`` mapping function with elementary unit 
        cell specified through ``sites``, various tilings of a square lattice can be achieved:: 
            
            # Example 1: 1-site translational iPEPS
            
            sites={(0,0): a}
            def vertexToSite(coord):
                return (0,0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   a  a a a a
            # -1   a  a a a a
            #  0   a  a a a a
            #  1   a  a a a a
            # Example 2: 2-site bipartite iPEPS
            
            sites={(0,0): a, (1,0): b}
            def vertexToSite(coord):
                x = (coord[0] + abs(coord[0]) * 2) % 2
                y = abs(coord[1])
                return ((x + y) % 2, 0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  b a b a
            # -1   B  a b a b
            #  0   A  b a b a
            #  1   B  a b a b
        
            # Example 3: iPEPS with 3x2 unit cell with PBC 
            
            sites={(0,0): a, (1,0): b, (2,0): c, (0,1): d, (1,1): e, (2,1): f}
            wfc= IPEPS(sites,lX=3,lY=2)
            
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   b  c a b c
            # -1   e  f d e f
            #  0   b  c a b c
            #  1   e  f d e f

        where in the last example a default setting for ``vertexToSite`` is used, which
        maps square lattice into elementary unit cell of size ``lX`` x ``lY`` assuming 
        periodic boundary conditions (PBC) along both X and Y directions.
        """
        self.dtype= global_args.dtype
        self.device= global_args.device

        self.sites= OrderedDict(sites)
        
        # TODO we infer the size of the cluster from the keys of sites. Is it OK?
        # infer the size of the cluster
        if lX is None or lY is None:
            min_x = min([coord[0] for coord in sites.keys()])
            max_x = max([coord[0] for coord in sites.keys()])
            min_y = min([coord[1] for coord in sites.keys()])
            max_y = max([coord[1] for coord in sites.keys()])
            self.lX = max_x-min_x + 1
            self.lY = max_y-min_y + 1
        else:
            self.lX = lX
            self.lY = lY

        if vertexToSite is not None:
            self.vertexToSite = vertexToSite
        else:
            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*self.lX)%self.lX, (y + abs(y)*self.lY)%self.lY )
            self.vertexToSite = vertexToSite

    def site(self, coord):
        """
        :param coord: tuple (x,y) specifying vertex on a square lattice
        :type coord: tuple(int,int)
        :return: on-site tensor corresponding to the vertex (x,y)
        :rtype: torch.tensor
        """
        return self.sites[self.vertexToSite(coord)]

    def get_parameters(self):
        return self.sites.values()

    def get_checkpoint(self):
        return self.sites

    def load_checkpoint(self,checkpoint_file):
        checkpoint= torch.load(checkpoint_file)
        self.sites= checkpoint["parameters"]

    def write_to_file(self,outputfile,aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        write_ipeps(self,outputfile,aux_seq=aux_seq, tol=tol, normalize=normalize)

    def add_noise(self,noise):
        r"""
        :param noise: magnitude of the noise
        :type noise: float

        Take IPEPS and add random uniform noise with magnitude ``noise`` to all on-site tensors
        """
        for coord in self.sites.keys():
            rand_t = torch.rand( self.sites[coord].size(), dtype=self.dtype, device=self.device)
            self.sites[coord] = self.sites[coord] + noise * rand_t

    def get_aux_bond_dims(self):
        return [d for key in self.sites.keys() for d in self.sites[key].size()[1:]]

    def __str__(self):
        print(f"lX x lY: {self.lX} x {self.lY}")
        for nid,coord,site in [(t[0], *t[1]) for t in enumerate(self.sites.items())]:
            print(f"a{nid} {coord}: {site.size()}")
        
        # show tiling of a square lattice
        coord_list = list(self.sites.keys())
        mx, my = 3*self.lX, 3*self.lY
        label_spacing = 1+int(math.log10(len(self.sites.keys())))
        for y in range(-my,my):
            if y == -my:
                print("y\\x ", end="")
                for x in range(-mx,mx):
                    print(str(x)+label_spacing*" "+" ", end="")
                print("")
            print(f"{y:+} ", end="")
            for x in range(-mx,mx):
                print(f"a{coord_list.index(self.vertexToSite((x,y)))} ", end="")
            print("")
        
        return ""

def read_ipeps(jsonfile, vertexToSite=None, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPS in json format
    :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                         into a vertex within elementary unit cell
    :param aux_seq: array specifying order of auxiliary indices of on-site tensors stored
                    in `jsonfile`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type vertexToSite: function(tuple(int,int))->tuple(int,int)
    :type aux_seq: list[int]
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS
    

    A simple PBC ``vertexToSite`` function is used by default
    
    Parameter ``aux_seq`` defines the expected order of auxiliary indices
    in input file relative to the convention fixed in tn-torch::
    
         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2
        
        for alternative order, eg.
        
         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2] 
         3
    """
    asq = [x+1 for x in aux_seq]
    sites = OrderedDict()
    
    with open(jsonfile) as j:
        raw_state = json.load(j)

        # check for presence of "aux_seq" field in jsonfile
        if "aux_ind_seq" in raw_state.keys():
            asq = [x+1 for x in raw_state["aux_ind_seq"]]

        # Loop over non-equivalent tensor,site pairs in the unit cell
        for ts in raw_state["map"]:
            coord = (ts["x"],ts["y"])

            # find the corresponding tensor (and its elements) 
            # identified by "siteId" in the "sites" list
            t = None
            for s in raw_state["sites"]:
                if s["siteId"] == ts["siteId"]:
                    t = s
            if t == None:
                raise Exception("Tensor with siteId: "+ts["sideId"]+" NOT FOUND in \"sites\"") 

            # 0) find the dimensions of auxiliary indices
            # branch 1: key "auxInds" exists

            # branch 2: key "auxInds" does not exist, all auxiliary 
            # indices have the same dimension
            X = torch.zeros((t["physDim"], t["auxDim"], t["auxDim"], \
                t["auxDim"], t["auxDim"]), dtype=global_args.dtype, device=global_args.device)

            # 1) fill the tensor with elements from the list "entries"
            # which list the non-zero tensor elements in the following
            # notation. Dimensions are indexed starting from 0
            # 
            # index (integer) of physDim, left, up, right, down, (float) Re, Im  
            for entry in t["entries"]:
                l = entry.split()
                X[int(l[0]),int(l[asq[0]]),int(l[asq[1]]),int(l[asq[2]]),int(l[asq[3]])]=float(l[5])

            sites[coord]=X#.

        # Unless given, construct a function mapping from
        # any site of square-lattice back to unit-cell
        # check for legacy keys
        lX = raw_state["sizeM"] if "sizeM" in raw_state else raw_state["lX"]
        lY = raw_state["sizeN"] if "sizeN" in raw_state else raw_state["lY"]

        if vertexToSite == None:
            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*lX)%lX, (y + abs(y)*lY)%lY )

            state = IPEPS(sites, vertexToSite, lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)
        else:
            state = IPEPS(sites, vertexToSite, lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)
    return state

def extend_bond_dim(state, new_d):
    r"""
    :param state: wavefunction to modify
    :param new_d: new enlarged auxiliary bond dimension
    :type state: IPEPS
    :type new_d: int
    :return: wavefunction with enlarged auxiliary bond dimensions
    :rtype: IPEPS

    Take IPEPS and enlarge all auxiliary bond dimensions of all on-site tensors up to 
    size ``new_d``
    """
    new_state = state
    for coord,site in new_state.sites.items():
        dims = site.size()
        size_check = [new_d >= d for d in dims[1:]]
        if False in size_check:
            raise ValueError("Desired dimension is smaller than following aux dimensions: "+str(size_check))

        new_site = torch.zeros((dims[0],new_d,new_d,new_d,new_d), dtype=state.dtype, device=state.device)
        new_site[:,:dims[1],:dims[2],:dims[3],:dims[4]] = site
        new_state.sites[coord] = new_site
    return new_state

def write_ipeps(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param aux_seq: array specifying order in which the auxiliary indices of on-site tensors 
                    will be stored in the `outputfile`
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPS
    :type ouputfile: str or Path object
    :type aux_seq: list[int]
    :type tol: float
    :type normalize: bool

    Parameter ``aux_seq`` defines the order of auxiliary indices relative to the convention 
    fixed in tn-torch in which the tensor elements are written out::
    
         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2
        
        for alternative order, eg.
        
         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2] 
         3

    """
    asq = [x+1 for x in aux_seq]
    json_state=dict({"lX": state.lX, "lY": state.lY, "sites": []})
    
    site_ids=[]
    site_map=[]
    for nid,coord,site in [(t[0], *t[1]) for t in enumerate(state.sites.items())]:
        if normalize:
            site= site/torch.max(torch.abs(site))

        json_tensor=dict()
        
        tdims = site.size()
        tlength = tdims[0]*tdims[1]*tdims[2]*tdims[3]*tdims[4]
        
        site_ids.append(f"A{nid}")
        site_map.append(dict({"siteId": site_ids[-1], "x": coord[0], "y": coord[1]} ))
        json_tensor["siteId"]=site_ids[-1]
        json_tensor["physDim"]= tdims[0]
        # assuming all auxBondDim are identical
        json_tensor["auxDim"]= tdims[1]
        json_tensor["numEntries"]= tlength
        entries = []
        elem_inds = list(itertools.product( *(range(i) for i in tdims) ))
        for ei in elem_inds:
            entries.append(f"{ei[0]} {ei[asq[0]]} {ei[asq[1]]} {ei[asq[2]]} {ei[asq[3]]}"\
                +f" {site[ei[0]][ei[1]][ei[2]][ei[3]][ei[4]]}")
            
        json_tensor["entries"]=entries
        json_state["sites"].append(json_tensor)

    json_state["siteIds"]=site_ids
    json_state["map"]=site_map

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))

##############################################
class IPESS():
    def __init__(self, sites, vertexToSite=None, lX=None, lY=None, peps_args=cfg.peps_args,\
        global_args=cfg.global_args):
        r"""
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : torch.tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Member ``sites`` is a dictionary of non-equivalent on-site tensors
        indexed by tuple of coordinates (x,y) within the elementary unit cell.
        The index-position convetion for on-site tensors is defined as follows::

               u s 
               |/ 
            l--a--r  <=> a[s,u,l,d,r]
               |
               d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.
        Member ``vertexToSite`` is a mapping function from any vertex (x,y) on a square lattice
        passed in as tuple(int,int) to a corresponding vertex within elementary unit cell.
        
        On-site tensor of an IPEPS object ``wfc`` at vertex (x,y) is conveniently accessed 
        through the member function ``site``, which internally uses ``vertexToSite`` mapping::
            
            coord= (0,0)
            a_00= wfc.site(coord)

        By combining the appropriate ``vertexToSite`` mapping function with elementary unit 
        cell specified through ``sites``, various tilings of a square lattice can be achieved:: 
            
            # Example 1: 1-site translational iPEPS
            
            sites={(0,0): a}
            def vertexToSite(coord):
                return (0,0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   a  a a a a
            # -1   a  a a a a
            #  0   a  a a a a
            #  1   a  a a a a
            # Example 2: 2-site bipartite iPEPS
            
            sites={(0,0): a, (1,0): b}
            def vertexToSite(coord):
                x = (coord[0] + abs(coord[0]) * 2) % 2
                y = abs(coord[1])
                return ((x + y) % 2, 0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  b a b a
            # -1   B  a b a b
            #  0   A  b a b a
            #  1   B  a b a b
        
            # Example 3: iPEPS with 3x2 unit cell with PBC 
            
            sites={(0,0): a, (1,0): b, (2,0): c, (0,1): d, (1,1): e, (2,1): f}
            wfc= IPEPS(sites,lX=3,lY=2)
            
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   b  c a b c
            # -1   e  f d e f
            #  0   b  c a b c
            #  1   e  f d e f

        where in the last example a default setting for ``vertexToSite`` is used, which
        maps square lattice into elementary unit cell of size ``lX`` x ``lY`` assuming 
        periodic boundary conditions (PBC) along both X and Y directions.
        """
        self.dtype= global_args.dtype
        self.device= global_args.device

        self.sites= OrderedDict(sites)
        
        # TODO we infer the size of the cluster from the keys of sites. Is it OK?
        # infer the size of the cluster
        if lX is None or lY is None:
            min_x = min([coord[0] for coord in sites.keys()])
            max_x = max([coord[0] for coord in sites.keys()])
            min_y = min([coord[1] for coord in sites.keys()])
            max_y = max([coord[1] for coord in sites.keys()])
            self.lX = max_x-min_x + 1
            self.lY = max_y-min_y + 1
        else:
            self.lX = lX
            self.lY = lY

        if vertexToSite is not None:
            self.vertexToSite = vertexToSite
        else:
            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*self.lX)%self.lX, (y + abs(y)*self.lY)%self.lY )
            self.vertexToSite = vertexToSite

    def site(self, coord):
        """
        :param coord: tuple (x,y) specifying vertex on a square lattice
        :type coord: tuple(int,int)
        :return: on-site tensor corresponding to the vertex (x,y)
        :rtype: torch.tensor
        """
        return self.sites[self.vertexToSite(coord)]

    def get_parameters(self):
        return self.sites.values()

    def get_checkpoint(self):
        return self.sites

    def load_checkpoint(self,checkpoint_file):
        checkpoint= torch.load(checkpoint_file)
        self.sites= checkpoint["parameters"]

    def write_to_file(self,outputfile,aux_seq=[0,1,2], tol=1.0e-14, normalize=False):
        write_ipess(self,outputfile,aux_seq=aux_seq, tol=tol, normalize=normalize)

    def add_noise(self,noise):
        r"""
        :param noise: magnitude of the noise
        :type noise: float

        Take IPESS and add random uniform noise with magnitude ``noise`` to all on-site tensors
        """
        for coord, ten in self.sites.items():
            for i, j in enumerate(ten):

                rand_t = torch.rand( self.sites[coord][i].size(), dtype=self.dtype, device=self.device)
                self.sites[coord][i] = self.sites[coord][i] + noise * rand_t

    def get_aux_bond_dims(self):
        return [d for key in self.sites.keys() for d in self.sites[key].size()[1:]]

    def __str__(self):
        print(self.sites.items())
        return ""

def write_ipess(state, outputfile, aux_seq=[0,1,2], tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param aux_seq: array specifying order in which the auxiliary indices of on-site tensors 
                    will be stored in the `outputfile`
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPS
    :type ouputfile: str or Path object
    :type aux_seq: list[int]
    :type tol: float
    :type normalize: bool

    """
    asq = [x+1 for x in aux_seq]
    json_state=dict({"lX": state.lX, "lY": state.lY, "sites": []})
    
    site_ids=[]
    site_map=[]
    for coord, ten in state.sites.items():
        for nid, local_ten in enumerate(ten):

            if normalize:
                local_ten= local_ten/torch.max(torch.abs(local_ten))

            json_tensor=dict()
        
            tdims = local_ten.size()
            tlength = tdims[0]*tdims[1]*tdims[2]
        
            site_ids.append(f"A{nid}_{coord}")
            site_map.append(dict({"siteId": site_ids[-1], "x": coord[0], "y": coord[1]} ))
            json_tensor["siteId"]=site_ids[-1]
            json_tensor["numEntries"]= tlength
            json_tensor["auxDim"]= tdims[0]
            entries = []
            elem_inds = list(itertools.product( *(range(i) for i in tdims) ))
            for ei in elem_inds:
                entries.append(f"{ei[0]} {ei[1]} {ei[2]}"\
                    +f" {local_ten[ei[0]][ei[1]][ei[2]]}")
                
            json_tensor["entries"]=entries
            json_state["sites"].append(json_tensor)


    json_state["siteIds"]=site_ids
    json_state["map"]=site_map

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))

def read_ipess(jsonfile, vertexToSite=None, aux_seq=[0,1,2], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPS in json format
    :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                         into a vertex within elementary unit cell
    :param aux_seq: array specifying order of auxiliary indices of on-site tensors stored
                    in `jsonfile`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type vertexToSite: function(tuple(int,int))->tuple(int,int)
    :type aux_seq: list[int]
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS
    

    A simple PBC ``vertexToSite`` function is used by default
    
    Parameter ``aux_seq`` defines the expected order of auxiliary indices
    in input file relative to the convention fixed in tn-torch::
    
         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2
        
        for alternative order, eg.
        
         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2] 
         3
    """
    asq = [x for x in aux_seq]
    sites = OrderedDict()
    
    with open(jsonfile) as j:
        raw_state = json.load(j)

        # check for presence of "aux_seq" field in jsonfile
        if "aux_ind_seq" in raw_state.keys():
            asq = [x+1 for x in raw_state["aux_ind_seq"]]

        # Loop over non-equivalent tensor,site pairs in the unit cell
        unit=[]
        for ts in raw_state["map"]:
            coord = (ts["x"],ts["y"])

            # find the corresponding tensor (and its elements) 
            # identified by "siteId" in the "sites" list
            t = None
            for s in raw_state["sites"]:
                if s["siteId"] == ts["siteId"]:
                    t = s
            if t == None:
                raise Exception("Tensor with siteId: "+ts["sideId"]+" NOT FOUND in \"sites\"") 

            # 0) find the dimensions of auxiliary indices
            # branch 1: key "auxInds" exists

            # branch 2: key "auxInds" does not exist, all auxiliary 
            # indices have the same dimension
            X = torch.zeros((t["auxDim"],2,t["auxDim"]), dtype=global_args.dtype, device=global_args.device)
            Y = torch.zeros((t["auxDim"],t["auxDim"],t["auxDim"]), dtype=global_args.dtype, device=global_args.device)

            # 1) fill the tensor with elements from the list "entries"
            # which list the non-zero tensor elements in the following
            # notation. Dimensions are indexed starting from 0

            try:
                for entry in t["entries"]:
                    l = entry.split()
                    X[int(l[asq[0]]),int(l[asq[1]]),int(l[asq[2]])]=float(l[3])
                unit.append(X)

            except:
                for entry in t["entries"]:
                    l = entry.split()
                    Y[int(l[asq[0]]),int(l[asq[1]]),int(l[asq[2]])]=float(l[3])
                unit.append(Y)
        sites[coord]=unit

        # Unless given, construct a function mapping from
        # any site of square-lattice back to unit-cell
        # check for legacy keys
        lX = 1
        lY = 1

        if vertexToSite == None:
            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*lX)%lX, (y + abs(y)*lY)%lY )

            state = IPESS(sites, vertexToSite, lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)
        else:
            state = IPESS(sites, vertexToSite, lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)
    return state