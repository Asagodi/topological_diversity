#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:32:10 2019

@author: abel
"""
import numpy as np 
import math
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
from copy import deepcopy
import itertools
from collections import Counter


class Combinatorial_Dynamical_System(object):
    "Combinatorial representation of datapoints from samples (from a dynamical system)"
    def __init__(self, delta):
        """
        f is map that generated data
        delta: diameter of cubes
        self.cubes are the cubes in the grid that have a datapoint in them, \mathcal{X} is a set of full cubes
        a cube is determined by the coordinates of its centre
        self.tuplecubes cubes as tuples
        self.cube_ind: the indices for the cubes in the order as they come in the data
        self.cube_ind_dict: dictionary between the cubes and the indices
        self.index_cube_dict: dictionary between the indices and the cubes
        the combinatorial cubical multivalued map \mathcal{F}=graph G 
            is a representation of f (f(Q)\subset |\mathcal{F}(Q)|) \forall Q\in\mathcal{X}
        """
#         self.data = data
#         self.data_length_list = data_length_list
        self.delta = delta
        self.cubes = []
        self.tuplecubes = []
        self.cube_ind = [-1]
        self.cube_ind_dict = {}
        self.index_cube_dict = {}
        ncubes = 0
        #this can get too big:
#         self.A=lil_matrix((ncubes, ncubes), dtype=np.int8)
        self.G = nx.DiGraph()
        self.G.add_nodes_from([i for i in range(ncubes)])
        
    def initialise_with_data(self, data, data_length_list):
        cube_ind = self.get_cubes(data)
        self.update_graph(cube_ind, data_length_list)
        
        
    def get_cubes(self, data):
        """Creates cubes for all the data with elementary cubes with diameter self.delta
        data has shape (n, d) with n: number of samples, d:dimension of space
        also updates"""
        #should data be translated here? (data is supposed to be positive in homcubes).
        max_cubeind = max(self.cube_ind)
        cube_ind = [max_cubeind]
        dim = data.shape[1]
        sigdelta = str(self.delta)[::-1].find('.')+1
        deltadiv2 = round(self.delta/2., sigdelta)
        for i,t in enumerate(data[:-1, 0]):
            cube = []
            for j in range(dim):
                coord = round(heaviside(data[i, j])*(self.delta*np.floor_divide(data[i,j],self.delta)+deltadiv2), sigdelta)
                cube.append(coord)
            if cube in self.cubes:
                ind = self.cubes.index(cube)
                cube_ind.append(ind)
            else:
                max_cubeind += 1
                cube_ind.append(max_cubeind)
                self.cubes.append(cube)
                self.cube_ind_dict[tuple(cube)] = max_cubeind
        try:
            cube_ind.remove(max(self.cube_ind))
        except:
            0
        self.cube_ind.extend(cube_ind)
        self.index_cube_dict = {v: k for k, v in self.cube_ind_dict.items()}
        self.tuplecubes = []
        for cube in self.cubes:
            self.tuplecubes.append(tuple(cube))
        return cube_ind
    
    def get_cubesandgraph(self, data, nbins, minval=None, maxval=None, calc_matrix=False):
        if minval==None:
            minval = np.min(data)
            self.minval = minval
        if maxval==None:
            maxval = np.max(data)
            self.maxval = maxval
        
        bins = np.linspace(0, maxval, nbins)
        digitized = np.digitize(data, bins)
        
        digitized_t = digitized.reshape((-1,digitized.shape[-1]))
        tuple_codewords = map(tuple, digitized_t)
        freq_ = Counter(tuple_codewords)
        self.bins = list(freq_.keys())
        self.nbins = len(list(freq_.keys()))
        self.cube_ind_dict = {k: i for i,k in enumerate(list(freq_.keys()))}
        self.index_cube_dict = {v: k for k, v in self.cube_ind_dict.items()}
        if calc_matrix:
            self.A=lil_matrix((self.nbins, self.nbins), dtype=np.int8)

        # digitized = digitized.reshape((data_length_list[0],-1,digitized.shape[-1]))
        self.G = nx.DiGraph()
        # self.G.add_nodes_from(self.bins)
        i=0
        # bin1=digitized[0]
        # bin2=digitized[1]
        # self.G.add_edge(tuple(bin1), tuple(bin2))
        for xid in range(digitized.shape[1]):
            dig_traj = digitized[:,xid,:]
            for t in range(dig_traj.shape[0]-1):
                bin1=tuple(list(dig_traj[t,  :]))
                bin2=tuple(list(dig_traj[t+1,:]))
                self.G.add_edge(tuple(bin1), tuple(bin2)) 
                if calc_matrix:
                    ci1 = self.cube_ind_dict[tuple(bin1)]
                    ci2 = self.cube_ind_dict[tuple(bin2)]
                    self.A[ci1, ci2] = 1
            
    
#     def get_cubesandgraph(self, data, nbins, data_length_list, minval=None, maxval=None, calc_matrix=False):
#         if minval==None:
#             minval = np.min(data)
#             self.minval = minval
#         if maxval==None:
#             maxval = np.max(data)
#             self.maxval = maxval
        
#         bins = np.linspace(0, maxval, nbins)
#         digitized = np.digitize(data, bins)
#         tuple_codewords = map(tuple, digitized)
#         freq_ = Counter(tuple_codewords)
#         self.bins = list(freq_.keys())
#         self.nbins = len(list(freq_.keys()))
#         self.cube_ind_dict = {k: i for i,k in enumerate(list(freq_.keys()))}
#         self.index_cube_dict = {v: k for k, v in self.cube_ind_dict.items()}
#         if calc_matrix:
#             self.A=lil_matrix((self.nbins, self.nbins), dtype=np.int8)

#         self.G = nx.DiGraph()
#         self.G.add_nodes_from(self.bins)
#         i=0
#         bin1=digitized[0]
#         # bin2=digitized[1]
#         # self.G.add_edge(tuple(bin1), tuple(bin2))
#         s=-1
#         for t,bin2 in enumerate(digitized[1:-1]):
#             s+=1
#             if s % data_length_list[i] == 0 and s!=0:
#                 i+=1
#                 s=-1
#                 continue
#             if s % data_length_list[i] == 1:
#                 bin1=digitized[t]
#                 bin2=digitized[t+1]

#             if calc_matrix:
#                 ci1 = self.cube_ind_dict[tuple(bin1)]
#                 ci2 = self.cube_ind_dict[tuple(bin2)]
#                 self.A[ci1, ci2] = 1
#             self.G.add_edge(tuple(bin1), tuple(bin2))
#             bin1 = bin2
    
    def update_graph(self, cube_ind, data_length_list, calc_matrix=False):
        """
        Updates graph G and transition matrix A from the cube_ind
        takes data_length_list as the list of the lengths 
        of the samples of the dynamics
        """
        ncubes = len(self.cubes)
#         self.G = nx.DiGraph()
#         self.G.add_nodes_from([i for i in range(ncubes)])
        i=0
        ci1=0
        s=-1
        for t,ci2 in enumerate(cube_ind):
            s+=1
            if s % data_length_list[i] == 0:
                ci1=cube_ind[t+1]
                if s == data_length_list[i]:
                    i+=1
                    s=0
                continue
            if s % data_length_list[i] == 1:
                continue
            #how to change size A?
            if calc_matrix:
                self.A[ci1, ci2] = 1.
            self.G.add_edge(ci1, ci2)
            ci1 = ci2

    def update_cubesandgraph(self, data, data_length_list):
        "Update cube set and graph for new data set"
        cube_ind = self.get_cubes(data)
        self.update_graph(cube_ind, data_length_list)

    def get_recurrent_components(self, includeselfedges=False):
        """
        Calculates recurrent components/strongly connected components
        these correnspond to the Morse sets
        """
        scc=nx.strongly_connected_components(self.G)
        RCs = []
        for cc in scc:
            if len(list(self.G.subgraph(cc).nodes()))>1:
                RCs.append(list(self.G.subgraph(cc).nodes()))
        if includeselfedges == True:
            for edge in self.G.edges():
                alreadyincluded = False
                if edge[0]==edge[1]:
                    for RC in RCs:
                        if edge[0] in RC:
                            alreadyincluded = True
                    if alreadyincluded == False:
                        RCs.append([edge[0]])

        self.RCs = RCs
        return RCs
    
    # def edge_in_subgraph(self, edge):
        

    def calc_F(self, U):
        "Calculates \mathcal{F}(U) for graph \mathcal{F}=self.G and subset U"
        Uprime = set()
        for zeta in U:
            for edge in self.G.out_edges(zeta):
                Uprime.add(edge[1])
        return Uprime

    def calc_F_inv(self, U):
        "Calculates \mathcal{F}^{-1}(U) for graph \mathcal{F}=self.G and subset U"
        Uprime = set()
        for zeta in U:
            for edge in self.G.in_edges(zeta):
                Uprime.add(edge[0])
        return Uprime
    
    def get_forward_invariant_subset(self, U, limit=1000):
        "Iterates until an invariant set is reached"
        A = set(U)
        #what should be the limit?
        for i in range(limit):

            Aprime = set()
            for zeta in A:
                for newzeta in self.G.in_edges(zeta):
                    Aprime.add(newzeta[0])

            if A == Aprime:
                break
            else:
                for r in A:
                    Aprime.add(r)
                A = Aprime.copy()
        return A
    
    def get_backward_invariant_subset(self, U, limit=1000):
        "Iterates back in time until an invariant set is reached"
        R = set(U)
        for i in range(limit):
            Rprime = set()
            for zeta in R:
                for newzeta in self.G.in_edges(zeta):
                    Rprime.add(newzeta[0])

            if R == Rprime:
                #check whether really invariant:
    #            if calc_F_inv(R, G) == R:
#                 repellers.append(R)
                break
            else:
                for r in R:
                    Rprime.add(r)
                R = Rprime.copy()
        return R
    

    def evaluate(self, F, U):
        "Calculates F(U) for give grap F and subset U"
        Uprime = set()
        for zeta in U:
            for edge in F.out_edges(zeta):
                Uprime.add(edge[1])
        return Uprime
    
    def evaluate_inv(self, F, U):
        "Calculates F^-1(U) for give grap F and subset U"
        Uprime = set()
        for zeta in U:
            for edge in F.in_edges(zeta):
                Uprime.add(edge[0])
        return Uprime

    def maximal_closed_subgraph(self):
        "Takes subgraph of all the nodes that have connetions both forwards and backwards"
        nnodes = len(self.G.nodes())
        Hp = deepcopy(self.G)
        Hpp = deepcopy(self.G)
        while True:
            for i in Hpp.nodes():
                if len(Hp.in_edges(i)) == 0:
                    Hp.remove_node(i)
                elif len(Hp.out_edges(i)) == 0:
                    Hp.remove_node(i)
            if len(Hp.nodes())==nnodes:
                break
            else:
                nnodes = len(Hp.nodes())
            Hpp = deepcopy(Hp)
        self.max_closed = Hp
        return Hp

    def invariantPart(self, N):
        """Combinatorial invariant set S inside set N and graph G=\mathcal{F}
        returns the set Inv(N , F)"""
        H = deepcopy(nx.subgraph(self.G, N)) #self.restricted_map(N)
        S = set(N).copy()
        while True:  
            Sp = S
            Spp = S.intersection(self.evaluate(H, S))
            S = Spp.intersection(self.evaluate_inv(H, S))
            if S == Sp:
                break
        return S
    
    def convert_cubes_to_indices(self, cubicalset):
        "Gets indices of set of cubes"
        indexset = set()
        for cube in cubicalset:
            indexset.add(self.cube_ind_dict[cube]) 
        return indexset
            
    def convert_indices_to_cubes(self, indexset):
        "Gets coordinates of the centres of cubes of set of cube indices (inverse of convert_cubes_to_indices)"
        cubicalset = set()
        for cube in indexset:
            cubicalset.add(self.index_cube_dict[cube])
        return cubicalset
    
    def index_pair(self, N):
        """N is set of coordinates of the centres of cubes
        returns index pair of N, if it is an isolating neighbourhood,
        otherwise returns Failure
        """
        S = self.invariantPart(N)
        M = self.cubical_wrap(S).intersection(self.G.nodes())
        if M.issubset(N):
            F = self.restricted_map(M)
            C = self.get_neighours_cubicalset(S)#collar(S)
            C = C.intersection(self.G.nodes())
            P0 = set(self.evaluate(F, S)).intersection(C)
            while True:
                lastP0 = P0
                P0 = self.evaluate(F, P0).intersection(C)
                P0 = P0.union(lastP0)
                if P0 == lastP0:
                    break
            P1 = S.union(P0)
            Pbar1 = self.evaluate(F, P1)
            Pbar0 = Pbar1 - S
            return P1, P0, Pbar1, Pbar0
        else:
            return "Failure"
        
    def get_neighours_cube(self, cube, kmax):
        """Takes the neighbours of a cube
        returns 3**kmax """
        
        L = set(cube)
        for some in itertools.product([-1,0,1], repeat=kmax):
            face = list(deepcopy(cube))
            for i,s in enumerate(some):
                face[i] = cube[i]+s
            L.add(tuple(face))
        L.remove(cube)

        return L
    
    def cubicalwrap_cube(self, cube, kmax):
        """takes the neighbours of a cube"""
        L = set()
        for some in itertools.product([-1,0,1], repeat=kmax):
            face = list(deepcopy(cube))
            for i,s in enumerate(some):
                face[i] = cube[i]+s
            L.add(tuple(face))
        return L

    def get_neighours_cubicalset(self, S):
        """takes the neighbours of a cubical set,
        same as collar in Computational Homology (Kaczynski, 2006)"""
        N = self.cubical_wrap(S)
        return N-set(S)

    def cubical_wrap(self, S):
        """takes the wrap of a cubical set"""
        #now only works for cubes of dim kmax
        maxdim = 0
        dim = len(list(S)[0])
        N = set()
        for cube in S:
            L = self.cubicalwrap_cube(cube, dim)
            for l in L:
                N.add(l)
        return N
        
    def restricted_map(self, N):
        """For the graph self.G takes the subgraph with nodes in list of nodes N"""
        return nx.subgraph(self.G, N)
    
    def convert_to_interval_representation(self, cubes):
        sigdelta = str(self.delta)[::-1].find('.')+3
        intervalcubes = set()
        dim = len(list(cubes)[0])
#         print(dim)
        for cube in cubes:
            newcube = []
            for j in range(dim):
                interval = tuple([round(heaviside(cube[j])*self.delta*np.floor_divide(cube[j],self.delta), sigdelta),
                                  round(heaviside(cube[j])*self.delta*np.floor_divide(cube[j],self.delta) + self.delta, sigdelta)])
                newcube.append(interval)
            intervalcubes.add(tuple(newcube))
        return intervalcubes


    def write_map_chompformat(self):
        """"""
        cubefile=''
        mapfile=''
        for i,j in enumerate(self.G.nodes()):
            c=self.convert_to_integertupleformat(self.convert_indices_to_cubes([j]))[0]
            cubefile+=str(c)+'\n'
            outedges = []
            for edge in self.G.out_edges(j):
                outedges.append(edge[1])
            outedges = self.convert_indices_to_cubes(outedges)
            intformat = self.convert_to_integertupleformat(outedges)
            if list(outedges)!=[]:
                infs = ''
                for cint in intformat:
                    infs+=str(cint)+' '
                mapfile+=str(c)+' -> {' + infs[:-1] + '}\n'
        return cubefile, mapfile

    def convert_to_integertupleformat(self, cubical_set):
        """Takes the cubical_set in coordindate representation to another representation for chomp
        (0.5, 1.5, 5.5)->[(0) (1) (5)] if delta=1"""
        itf = []
        for cube in cubical_set:
            new = tuple(np.array(np.round((np.array(cube)/self.delta), decimals=1),dtype='int'))
            itf.append(new)
        return itf
    
    def make_isolated(self, N, maxsteps = 10):
        I = self.invariantPart(N)
        for st in range(maxsteps):
            I_wrap = self.cubical_wrap(I).intersection(self.G.nodes())
#             M = self.cubical_wrap(I).intersection(self.G.nodes())
            I = self.invariantPart(M) 
        if I.issubset(I_wrap):
            return I
        else:
            print("Did not find isolated neighbourhood with given maximal steps")
    
# def write_mapandcubes(graph, delta, cds):
#     """"""
#     cubefile=''
#     mapfile=''
#     for i,j in enumerate(graph.nodes()):
#         c=convert_to_integertupleformat(convert_indices_to_cubes([j], cds), delta)[0]
#         cubefile+=str(c)+'\n'
#         outedges = []
#         for edge in graph.out_edges(j):
#             outedges.append(edge[1])
#         outedges = convert_indices_to_cubes(outedges, cds)
#         intformat = convert_to_integertupleformat(outedges, delta)
#         if list(outedges)!=[]:
#             infs = ''
#             for cint in intformat:
#                 infs+=str(cint)+' '
#             mapfile+=str(c)+' -> {' + infs[:-1] + '}\n'
#     return cubefile, mapfile

def write_mapandcubes(graph, delta, cds):
    """"""
    cubefile=''
    mapfile=''
    for i,node in enumerate(graph.nodes()):
#         print(node)
        cubefile+=str(node)+'\n'
        outedges = []
        for edge in graph.out_edges(node):
            outedges.append(edge[1])
#         print(outedges)
        if list(outedges)!=[]:
            infs = ''
            for cint in outedges:
                infs+=str(cint)+' '
            mapfile+=str(node)+' -> {' + infs[:-1] + '}\n'
        else:
            mapfile+=str(node)+' -> {}\n'
    return cubefile, mapfile

def convert_to_integertupleformat(cubical_set, delta):
    """Takes the cubical_set in coordindate representation to another representation for chomp
    (0.5, 1.5, 5.5)->[(0) (1) (5)] if delta=1"""
    itf = []
    for cube in cubical_set:
        new = tuple(np.array(np.round((np.array(cube)/delta), decimals=1),dtype='int'))
        itf.append(new)
    return itf

def convert_indices_to_cubes(indexset, cds):
    "Gets coordinates of the centres of cubes of set of cube indices (inverse of convert_cubes_to_indices)"
    cubicalset = set()
    for cube in indexset:
        cubicalset.add(cds.index_cube_dict[cube])
    return cubicalset


####....
def heaviside(x):
    if x<0:
        return -1
    if x>=0:
        return 1
    
def embed(data, dim, tau, shift=0):
    datalist = []
    for d in range(dim-1):
        datalist.append(data[shift+d*tau:-(dim-d-1)*tau])
    datalist.append(data[shift+(dim-1)*tau:])
    return np.array(datalist).T

def make_grid(dim, low, high, delta):
    "makes grid in dim dimensions with evenly spaced (hyper)cubes"
    n=int((high-low)/delta)+1 #+1?

    grid = []
    for coords in itertools.product(range(n),repeat=dim):
        gc = []
        for i in coords:
            gc.append(low+i*delta)
        grid.append(gc)
    return grid

def convert_to_chomp_format(cubical_set, delta):
    """Takes the cubical_set in coordindate representation to the representation for chomp
    (0.5, 1.5, 5.5)->[(0,1) (1,2) (5,6)] if delta=1"""
    filetxt = ""
    for cube in cubical_set:
        
        filetxt += '['
        for coord in cube:
#             new = np.array(np.round((np.array(coord)/delta)),dtype='int')
            filetxt+=str(tuple(np.array(np.round((np.array(coord)/delta)),dtype='int')))+ ' '
        filetxt = filetxt[:-1]+']\n'
    return filetxt

def get_dim(cube, delta):
    "returns dimension of cube"
    return np.where( np.abs(np.mod(cube, delta)-delta/2.) < 10**(np.log2(delta)) )[0].shape[0]

def primaryFaces(cube, delta):
    "cube = list(coordinate of centre)"
    "return primary faces of cube"
    relcoord = np.where( np.abs(np.mod(cube, delta)-delta/2.) < 10**(np.log2(delta)) )[0]
    k = relcoord.shape[0]
    L = set()

    for i in range(k):
        for j in range(2):
            face = list(deepcopy(cube))
            face[relcoord[i]] = round(cube[relcoord[i]]+(-1)**j*delta/2., int(-np.log2(delta)))
            L.add(tuple(face))
    return L


def boundaryOperator(cube, delta):
    "calculates the boundary operator for an elementary cube"
    sgn = 1
    chain = {}
    relcoord = np.where( np.abs(np.mod(cube, delta)-delta/2.) < 10**(np.log2(delta)) )[0]
    k = relcoord.shape[0]

    for i in range(k):
        for j in range(2):
            face = list(deepcopy(cube))
            face[relcoord[i]] = cube[relcoord[i]]+(-1)**j*delta/2.
            chain[tuple(face)] = (-1)**j*sgn
        sgn = -sgn
    return chain


def cubicalChainGroups(K, delta, maxdim=None):
    "K: cubical set (list of its maximal faces)"
    "returns E, the groups of cubical chains of a cubical set"
    E = []

    while K != set():
        Q = list(K)[0]
        K = set(list(K)[1:])
        k = get_dim(Q, delta)
        L = primaryFaces(Q, delta)
        if L != set():
            K = K.union(L)
        try:
            E[k] = E[k].union([Q])
        except:
            maxk = len(E)
            for ktok in range(k-maxk+1):
                E.append(set())
            E[k] = E[k].union([Q])
        if L != set():
            E[k-1] = E[k-1].union(L)
    return E

def canonicalCoordinates(chain, K):
    """returns the vector representation of a chain
    K is list of all elementary cubes for a certain dimension (can be calc by unrolledE)"""
    v = np.zeros(len(K))
    for i in range(len(K)):
        try:
            v[i] = chain[tuple(list(K)[i])]
        except:
            0
    return v


def boundaryOperatorMatrix(E, delta):
    "Calculates the boundary operator for a cubical chain E"
    D = [np.array([]) for k in range(len(E))]
    for k in range(1, len(E)): 
        m = len(E[k-1])
        l = len(E[k])
        D[k] = np.zeros((m, l))
        for j in range(l):
            chain = boundaryOperator(list(E[k])[j], delta)
            v = canonicalCoordinates(chain, E[k-1])
            D[k][:,j] = v
    D[0] = np.array([[0]*len(E[0])])
    return D

def simultaneousReduce(A, B):
   if A.shape[1] != B.shape[0]:
      raise Exception("Matrices have the wrong shape.")
 
   numRows, numCols = A.shape # col reduce A
 
   i,j = 0,0
   while True:
      if i >= numRows or j >= numCols:
         break
 
      if A[i][j] == 0:
         nonzeroCol = j
         while nonzeroCol < numCols and A[i,nonzeroCol] == 0:
            nonzeroCol += 1
 
         if nonzeroCol == numCols:
            j += 1
            continue
 
         colSwap(A, j, nonzeroCol)
         rowSwap(B, j, nonzeroCol)
 
      pivot = A[i,j]
      scaleCol(A, j, 1.0 / pivot)
      scaleRow(B, j, 1.0 / pivot)
 
      for otherCol in range(0, numCols):
         if otherCol == j:
            continue
         if A[i, otherCol] != 0:
            scaleAmt = -A[i, otherCol]
            colCombine(A, otherCol, j, scaleAmt)
            rowCombine(B, j, otherCol, -scaleAmt)
 
      i += 1; j+= 1
 
   return A,B

def simultaneousReduce(A, B):
    if A.shape[1] != B.shape[0]:
        raise Exception("Matrices have the wrong shape.")

    numRows, numCols = A.shape # col reduce A
 
    i,j = 0,0
    while True:
        if i >= numRows or j >= numCols:
            break
 
        if A[i][j] == 0:
            nonzeroCol = j
            while nonzeroCol < numCols and A[i,nonzeroCol] == 0:
                nonzeroCol += 1
 
            if nonzeroCol == numCols:
                i += 1
                continue
 
            colSwap(A, j, nonzeroCol)
            rowSwap(B, j, nonzeroCol)
 
        pivot = A[i,j]
        scaleCol(A, j, 1.0 / pivot)
        scaleRow(B, j, 1.0 / pivot)

        for otherCol in range(0, numCols):
            if otherCol == j:
                continue
            if A[i, otherCol] != 0:
                scaleAmt = -A[i, otherCol]
                colCombine(A, otherCol, j, scaleAmt)
                rowCombine(B, j, otherCol, -scaleAmt)
 
        i += 1; j+= 1
 
    return A,B

def singleReduce(A):
    numRows, numCols = A.shape 

    i,j = 0,0
    while True:
        if i >= numRows or j >= numCols:
            break
 
        if A[i][j] == 0:
            nonzeroCol = j
            while nonzeroCol < numCols and A[i,nonzeroCol] == 0:
                nonzeroCol += 1
 
            if nonzeroCol == numCols:
                i += 1
                continue
 
            colSwap(A, j, nonzeroCol)
 
        pivot = A[i,j]
        scaleCol(A, j, 1.0 / pivot)
 
        for otherCol in range(0, numCols):
            if otherCol == j:
                continue
            if A[i, otherCol] != 0:
                scaleAmt = -A[i, otherCol]
                colCombine(A, otherCol, j, scaleAmt)
 
        i += 1; j+= 1
 
    return A

def bettiNumber(k, d_k, d_kplus1):
    """
    returns k-th betti number
    d_k is k-th boundary operator matrix
    d_kplus1 is (k+1)-th boundary operator matrix
    """
    A, B = np.copy(d_k), np.copy(d_kplus1)
    if k == 0:
        B = singleReduce(B.T).T

    simultaneousReduce(A, B)

    dimKChains = A.shape[1]
    kernelDim = dimKChains - numPivotCols(A)
    imageDim = min(numPivotRows(B), B.shape[1])
   
    return kernelDim - imageDim

def get_bettiNumbers_of_cubicalset(tuplecubes, delta):
    """
    returns betti numbers up to maximal dimension of cubes in tuplecubes
    tuplecubes is the set of cubes as tuples of the coordinates of the center
    """
    E = cubicalChainGroups(tuplecubes, delta)
    D = boundaryOperatorMatrix(E, delta)
    bettinums = []
    for i in range(len(D)-1):
        bn = bettiNumber(i, D[i], D[i+1])
        bettinums.append(bn)
    return bettinums

def scaleCol(A, i, c):
    A[:, i] *= c*np.ones(A.shape[0])

def colCombine(A, addTo, scaleCol, scaleAmt):
    A[:, addTo] += scaleAmt * A[:, scaleCol]
    
def colSwap(A, i, j):
    temp = np.copy(A[:, i])
    A[:, i] = A[:, j]
    A[:, j] = temp
    
def rowSwap(A, i, j):
    temp = np.copy(A[i, :])
    A[i, :] = A[j, :]
    A[j, :] = temp
    
def scaleRow(A, i, c):
    A[i, :] *= c*np.ones(A.shape[1])
    
def rowCombine(A, addTo, scaleRow, scaleAmt):
    A[addTo, :] += scaleAmt * A[scaleRow, :]

def numPivotCols(A):
    z = np.zeros(A.shape[0])
    return [np.all(A[:, j] == z) for j in range(A.shape[1])].count(False)
 
def numPivotRows(A):
    z = np.zeros(A.shape[1])
    return [np.all(A[i, :] == z) for i in range(A.shape[0])].count(False)

def smallestNonzero(v, k):
    alpha = np.min(np.abs(v[k:][np.where(v[k:]!=0)]))
    i_0 = np.where(np.abs(v[k:]) == alpha)
    return alpha, i_0

def minNonzero(B, k):
    v = zeros(B.shape[1])
    q = zeros(B.shape[1])
    for i in range(B.shape[0]):
        if i >= k:
            v[i], q[i] = smallestNonzero(B[i,:], k)
    alpha, i_0 = smallestNonzero(v, k)
    return alpha, i_0, q[i_0]

def rowExchangeOperation(B, Q, Qtil, i, j):
    B = rowSwap(B, i, j)
    Q = rowSwap(Q, i, j)
    Qtil = colSwap(Qtil, i, j)
    return B, Q, Qtil

def columnExchangeOperation(B, Q, Qtil, i, j):
    B = colSwap(B, i, j)
    Q = colSwap(Q, i, j)
    Qtil = rowSwap(Qtil, i, j)
    return B, Q, Qtil

def moveMinNonzero(B, Q, Qtil, R, Rtil, k):
#     print(B.shape, k)
    alpha, i, j = minNonzero(B, k)
    B, Q, Qtil = rowExchangeOperation(B, Q, Qtil, i, j)
    B, R, Rtil = columnExchangeOperation(B, R, Rtil, i, j)
    return B, Q, Qtil, R, Rtil

def rowPrepare(B,Q,Qtil,k,l):
    m = B.shape[0]
    alpha, i_0 = smallestNonzero(B[:,l], k)
    B, Q, Qtil = rowExchangeOperation(B,Q,Qtil,k,i_0)
    return B,Q,Qtil


def rowReduce(B,Q,Qtil,k,l):
    m = B.shape[0]
    while np.all(B[k:,l]!=0): #any or all?
        B,Q,Qtil = rowPrepare(B,Q,Qtil,k,l)
        B,Q,Qtil = partRowReduce(B,Q,Qtil,k,l)
    return B,Q,Qtil

def partRowReduce(B, Q, Qtil, k, l):
    for i in range(k, B.shape[0]):
        q = np.floor((B[i, l]/B[k, l]))
        B[i, :] = B[i, :] - q*B[k, :]
        Qtil[i, :] = Qtil[i, :] - q*Qtil[k, :]
        Q[i, :] = Q[i, :] + q*Q[k, :]
    return B, Q, Qtil

def checkForDivisibility(B, k):
    for i in range(k+1, B.shape[0]):
        for j in range(k+1, B.shape[1]):
            q = np.floor(B[i, j]/B[k, k])
            
            if q*B[k,k] != B[i,j]:
                return False, i, j, q
    return True, 0, 0, 0

def partSmithForm(B, Q, Qtil, R, Rtil, k):
    B, Q, Qtil, R, Rtil = moveMinNonzero(B, Q, Qtil, R, Rtil, k)
    m = B.shape[0]
    n = B.shape[1]
    divisible = True
    while divisible:
        B, Q, Qtil, R, Rtil = moveMinNonzero(B, Q, Qtil, R, Rtil, k)
        B, Q, Qtil = partRowReduce(B, Q, Qtil, k, k)
        if B[k+1:, k] != 0:
            continue
        if B[k, k + 1:] != 0:
            continue
        divisible, i, j, q = checkForDivisibility(B, k)
        if not divisible:
            B, Q, Qtil = rowAddOperation(B, Q, Qtil, i, k, l)
            B, R, Rtil = columnAddOperation(B, R, Rtil, k, j, -q)
    return B, Q, Qtil, R, Rtil

def rowMultiply(B, i):
    B[i, :] = -B[i, :]
    return B

def columnMultiply(B, i):
    B[:, i] = -B[:, i]
    return B

def rowMultiplyOperation(B, Q, Qtil, i):
    B = rowMultiply(B, i)
    Qtil = rowMultiply(Qtil, i)
    B = columnMultiply(B, i)
    return B, Q, Qtil

def smithForm(B):
    m = B.shape[0]
    n = B.shape[1]
    Q, Qtil = eye(m), eye(m)
    R, Rtil = eye(n), eye(n)
    s, t = 0, 0
    while np.all(B[t:, t:] != 0):
        t+=1
        B, Q, Qtil, R, Rtil = partSmithForm(B, Q, Qtil, R, Rtil, t)
        if B[t, t] < 0:
            B, Q, Qtil = rowMultiplyOperation(B,Q,Qtil)
        if B[t, t] == 1:
            s = s + 1
    return B, Q, Qtil, R, Rtil, s, t

def make_wc(W, ps, dt=0.1, inits=[]):
    N = W.shape[0]
    eqstring = ''
    wi = 0
    for i in range(1,W.shape[0]+1):
        eqstring+='x'+str(i)+"'=-x"+str(i)+"+f("+'w'+str((i-1)*N+i-1)+"*x"+str(i)+'+p%s+'%(i)#+'-b*y'+str(i)
        for j in range(1,W.shape[1]+1):
            if i!=j:
                eqstring+='w'+str(wi)+"*x"+str(j)+'+'
            wi+=1
        eqstring=eqstring[:-1]+')\n'

    weighstring = 'p '
    wi=0
    for i in range(0,W.shape[0]):
        for j in range(0,W.shape[1]):
            if wi%25==0 and wi!=0:
                weighstring=weighstring[:-1]+'\np '
            weighstring+='w'+str(wi)+'='+str(round(W[i,j], 2))+','
            wi+=1

    inputstring = 'par '
    for i in range(0,W.shape[0]):
        inputstring+='p'+str(i+1)+'='+str(ps[i])+','

    initstring = 'init '
    ##np.random.rand(2*N)
    if inits==[]:
        inits = [0.]*N
        for i in range(W.shape[0]):
            initstring+='x'+str(int(i+1))+'='+str(inits[i])+','
    else:
        for i in range(W.shape[0]):
            initstring+='x'+str(int(i+1))+'='+str(inits[i])+','

    wcstring = "# the wilson-cowan equations\n"
#     wcstring += "f(u)=u/(sqrt(1+u**2))\n"##1/(1+exp(-u))
    wcstring += "f(u)=1/(1+exp(-u))\n"##
    wcstring += eqstring[:-1] + "\n"
    wcstring += weighstring[:-1] + "\n"
    wcstring += inputstring[:-1] + " \n"
    wcstring += initstring[:-1] + "\n"
    wcstring += '@ xp=x1,yp=x2,xlo=-.125,ylo=-.125,xhi=1,yhi=1,dt=%s\n'%dt
    wcstring += "done"
    return wcstring

def make_ctln(W, bs, dt=0.1):
    N = W.shape[0]
    eqstring = ''
    wi = 0
    for i in range(1,W.shape[0]+1):
        eqstring+='x'+str(i)+"'=-x"+str(i)+"+f("+'b%s+'%(i)#+'-b*y'+str(i)
        for j in range(1,W.shape[1]+1):
            if i!=j:
                eqstring+='w'+str(wi)+"*x"+str(j)+'+'
            wi+=1
        eqstring=eqstring[:-1]+')\n'

    weighstring = 'p '
    wi=0
    for i in range(0,W.shape[0]):
        for j in range(0,W.shape[1]):
            if wi%25==0 and wi!=0:
                weighstring=weighstring[:-1]+'\np '
            if i!=j:
                weighstring+='w'+str(wi)+'='+str(round(W[i,j], 2))+','
            wi+=1

    inputstring = 'par '
    for i in range(0,W.shape[0]):
        inputstring+='b'+str(i+1)+'='+str(bs[i])+','

    initstring = 'init '
    inits = [0.]*N#np.random.rand(2*N)
    for i in range(W.shape[0]):
            initstring+='x'+str(int(i+1))+'='+str(inits[i])+','

    wcstring = "# ctln\n"
    wcstring += "f(x)=if(x<0)then(0)else(x)\n"##
    wcstring += eqstring[:-1] + "\n"
    wcstring += weighstring[:-1] + "\n"
    wcstring += inputstring[:-1] + " \n"
    wcstring += initstring[:-1] + "\n"
    wcstring += '@ xp=x1,yp=x2,xlo=-.125,ylo=-.125,xhi=1,yhi=1,dt=%s\n'%dt
    wcstring += "done"
    return wcstring

def make_wc_mayu(W, ps, sigmoids, others, dt=0.05):
#     W, ps, k
    N = W.shape[0]
    eqstring = ''
    wi = 0
    for i in range(1,W.shape[0]+1):
        eqstring+='x'+str(i)+"'=-x"+str(i)+"+(1-x"+str(i)+")*"+"f%s("%(np.mod(i+1,2))+'w'+str((i-1)*N+i-1)+"*x"+str(i)+'+p%s+'%(i)#+'-b*y'+str(i)
        for j in range(1,W.shape[1]+1):
            if i!=j:
                eqstring+='w'+str(wi)+"*x"+str(j)+'+'
            wi+=1
        eqstring=eqstring[:-1]+')\n'

    weighstring = 'p '
    wi=0
    for i in range(0,W.shape[0]):
        for j in range(0,W.shape[1]):
            weighstring+='w'+str(wi)+'='+str(round(W[i,j], 2))+','
            wi+=1

    inputstring = 'par '
    for i in range(0,W.shape[0]):
        inputstring+='p'+str(i+1)+'='+str(ps[i])+','
        
    for par in others:
        inputstring+=par+'='+str(others[par])+','

    initstring = 'init '
    inits = [0.]*N#np.random.rand(2*N)
    for i in range(W.shape[0]):
            initstring+='x'+str(int(i+1))+'='+str(inits[i])+','

    wcstring = "# the wilson-cowan equations\n"
#     wcstring += "f(u)=u/(sqrt(1+u**2))\n"##1/(1+exp(-u))
    for sigmoid in sigmoids:
        wcstring += sigmoid+"\n"
    wcstring += eqstring[:-1] + "\n"
    wcstring += weighstring[:-1] + "\n"
    wcstring += inputstring[:-1] + " \n"
    wcstring += initstring[:-1] + "\n"
    wcstring += '@ xp=x1,yp=x2,xlo=-.125,ylo=-.125,xhi=1,yhi=1,dt=%s\n'%dt
    wcstring += "done"
    return wcstring

def jac(x,W,ps):
    n=W.shape[0]
    J = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            ins = np.dot(W[i,:], x)+ps[i]
            J[i,j] = W[i,j]*sigmoid(ins)*(1-sigmoid(ins))
            if i==j:
                J[i,j]+=-1
    return J

def eval_wcode(x, W, ps):
    return -x+sigmoid(W.dot(x)+np.array(ps))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def F(x):
    return -x+sigmoid(W.dot(x)+np.array(ps))

def G(x):
    -x+sigmoid(W[0,0]*x+ps[0])
    
    
    
