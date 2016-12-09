import numpy as np
import sys
import scipy.sparse as sp
import scipy.sparse.linalg as spla

######################################
#          Class: network            #
######################################
class network_class:
    """Contains: parameters of the model, namely size, boundary
    conditions, position of the point contacts, angle alpha
    of the scattering matrix at nodes of type A,B.
    In the future, also disorder.
    Network is made out of plaquettes ij with labeling of edges as:
    
      /\
     0  1   
    /    \
    \    /
     3  2
      \/
 
    arrows: 0,3 upgoing, 1,2 downgoing.

    TODOs: 
    - Implement Z4 symmetry to reduce by 4 the dimension of linear system
    - csc format since we want to extract the pc_ind column when solving
      the linear system

    Parameters
    ----------
    my_geometry = dictiornary containing the following:
                  bc = string among 'torus', 'cylinder', 'rectangle'
                  W = circumference of the cylinder
                  L = length of the cylinder
    my_pc = list of integers containing the positions of the point contacts 
            as [[x1,y1,a1],[x2,y2,a2],...]
    my_scatt_angles = dictionary with 'A', 'B'
    disorder = boolean specifying if random phases or not

    Returns
    -------
    object of this type.

    """

    #
    # Class constructor
    #
    def __init__(self, my_geometry, my_pcs, my_scatt_angles, disorder):
        print("Inside class network_class constructor")
        # Save model and couplings for later use
        self.geometry = my_geometry
        self.pcs = my_pcs
        self.num_pcs = len(self.pcs)
        self.scatt_angles = my_scatt_angles
        self.num_plaq = self.geometry['L']*self.geometry['W']
        self.num_links = 4*self.num_plaq
        self.disorder = disorder
        # Random phases are initialized every time construct_T_csr is called.
        if disorder == False: # do not set random phases
            self.dtype = float
        else: # disorder, standard behavior
            self.dtype = complex  # complex type due to phases

    #
    #
    #
    def construct_T_csr(self):
        """
        Returns T as a sparse matrix with format csr,
        where the column indices for row i are stored in
        indices[indptr[i]:indptr[i+1]] and their corresponding values
        are stored in data[indptr[i]:indptr[i+1]].

        indptr : size num_links, = [0,2,4, ... , 2*num_links - 1]

        Important: loops are vectorized, slicing and reshaping to set 
        arrays instead of for loops which are extremely slow in python...

        Allocate an instance of disorder each time called.

        The implementation of cylinder is compact but not efficient.
        TODO: Implement cylinder by reducing nnz
              Also, implement csc

        Parameters
        ----------
        self : 

        Returns
        -------
        indptr : size num_links
        indices : size 2*num_links
        data : size 2*num_links    """

        L = self.geometry['L']
        W = self.geometry['W']
        NP= self.num_plaq #number of plaquettes
        NL= self.num_links #number of links
        links=np.arange(NL,dtype=np.intp)
        # phases:
        if self.disorder == False: # do not set random phases
            phi = np.zeros(self.num_links) # float
        else: # disorder, standard behavior
            phi = 2*np.pi*np.random.rand(self.num_links) # float
        # indexptr is easy (always assume torus!)
        indexptr = np.arange(0,2*(NL+1),2,dtype=np.intp)
        indices=np.zeros(2*NL,dtype=np.intp) #intp: type for indexing
        # set indices
        # first contribution to (ij0) : (ij3)
        indices[0::8]=links[3::4]
        # second contribution to (ij0), i==0: (L-1,j,1); i>0:
        # (i-1,j,1).  first reshape into a matrix of the type of the
        # network. Then move last column at the beginning, then
        # reshape back
        tmp=links[1::4].reshape(W,L)
        indices[1::8]=np.hstack((tmp[:,L-1:],tmp[:,:L-1])).reshape(L*W)
        # first contribution to (ij1) : (ij0)
        indices[2::8]=links[0::4]
        # second contribution to (ij1), j==W-1: (i,0,2); j<W-1:
        # (i,j+1,2).  first reshape into a matrix of the type of the
        # network. Then move first row to the end, then reshape back
        tmp=links[2::4].reshape(W,L)
        indices[3::8]=np.vstack((tmp[1:,:],tmp[0,:])).reshape(L*W)
        # first contribution to (ij2) : (ij1)
        indices[4::8]=links[1::4]
        # second contribution to (ij2), i==L-1: (0,j,3); i<L-1:
        # (i+1,j,3).  first reshape into a matrix of the type of the
        # network. Then move first column to the end, then reshape
        # back
        tmp=links[3::4].reshape(W,L)
        indices[5::8]=np.hstack((tmp[:,1:],tmp[:,:1])).reshape(L*W)
        # first contribution to (ij3) : (ij2)
        indices[6::8]=links[2::4]
        # second contribution to (ij3), j==0: (i,W-1,0); j>0:
        # (i,j-1,0).  first reshape into a matrix of the type of the
        # network. Then move last row to the beginning, then reshape
        # back
        tmp=links[0::4].reshape(W,L)
        indices[7::8]=np.vstack((tmp[W-1,:],tmp[:W-1,:])).reshape(L*W)

        # Set parameters a,b,g,d (B); ap,bp,gp,dp (A)
        ap = np.cos(self.scatt_angles['A'])
        bp = np.cos(self.scatt_angles['A'])
        cp = np.sin(self.scatt_angles['A'])
        dp = -np.sin(self.scatt_angles['A'])
        a = np.cos(self.scatt_angles['B'])
        b = np.cos(self.scatt_angles['B'])
        c = np.sin(self.scatt_angles['B'])
        d = -np.sin(self.scatt_angles['B'])
        #
        data=np.zeros(2*NL, dtype = self.dtype)
        cphi=phi[0::4]
        data[0::8]=a*np.exp(1j*cphi)
        data[1::8]=d*np.exp(1j*cphi)
        cphi=phi[1::4]
        data[2::8]=cp*np.exp(1j*cphi)
        data[3::8]=ap*np.exp(1j*cphi)
        cphi=phi[2::4]
        data[4::8]=b*np.exp(1j*cphi)
        data[5::8]=c*np.exp(1j*cphi)
        cphi=phi[3::4]
        data[6::8]=dp*np.exp(1j*cphi)
        data[7::8]=bp*np.exp(1j*cphi)

        if self.geometry['bc'] == 'cylinder':
            # left boundary
            data[1::8][0::L]=np.zeros(W)
            # right boundary
            data[5::8][L-1::L]=np.zeros(W)
        
        # Q : set to zero the row corresponding to point contacts
        for x in range(self.num_pcs):
            row = self.get_index(self.pcs[x][0],self.pcs[x][1],self.pcs[x][2])
            data[indexptr[row]:indexptr[row+1]] = np.zeros(indexptr[row+1]-indexptr[row])

        return indexptr, indices, data

    #
    # get_instance_of_psic
    #
    def get_instance_of_psic(self):
        """
        Compute and return the scattering states psic

        Parameters:
        ----------
        
        Returns:
        psics : list of psic, one for each point contact. 
                shape=(num_pcs,num_links)
        """
        # Define an instance of sparse matrix T in csr format
        indptr, indices, data = self.construct_T_csr()
        T_csr = sp.csr_matrix( (data,indices,indptr), \
                               shape=(self.num_links, self.num_links) )
        # Define the sparse csr matrix 1-T
        one_minus_T_csr = sp.eye(self.num_links)-T_csr

        psics = np.array([], dtype=self.dtype) #list of psic to be ret
        for x in range(self.num_pcs):
            pc_ind = self.get_index(self.pcs[x][0],self.pcs[x][1],self.pcs[x][2])
            #print('Computing psi_c = (1-T)^{-1}T|c> for |c> at ', pcs[i]) 
            # Apply T on v[pc_ind]=1, other = 0 gives pc_ind -
            # th column of T efficient in csc format not in csr...
            b = T_csr.getcol(pc_ind)
            # direct solver of sparse linear system
            psic = spla.spsolve(one_minus_T_csr, b)
            # check:
            #print('check:',np.allclose(one_minus_T_csr * psic , b.toarray().reshape(net.num_links)))
            psics = np.append(psics, psic)

        return psics.reshape(self.num_pcs, self.num_links)

    #
    # class module: get_index
    #
    def get_index(self, x,y,a):
        """Returns the index of a-th link of a plaquette with 
        coordinates x,y for a rectangular network L \times W

        Parameters
        ----------
        self :
        x : 
        y :
        a :
        
        Returns
        -------
        index
        
        """
        return 4*(x+self.geometry['L']*y)+a

    #
    # class module: check consistency
    #
    def check_consistency(self):
        """
        Check unitarity of T if no pcs

        """
        if self.num_pcs == 0:
            # If no pc, create a matrix and check unitarity:
            # Define an instance of sparse matrix T in csr format
            indptr, indices, data = self.construct_T_csr()
            T_csr = sp.csr_matrix( (data,indices,indptr), \
                                       shape=(self.num_links, self.num_links) )
            T_mat = T_csr.toarray()
            print('T^dagger allclose T^-1 : ', 
                  np.allclose(np.linalg.inv(T_mat), np.conj(np.transpose(T_mat))))
            print('abs(eigs) allclose 1 : ',
                  np.allclose(np.abs(np.linalg.eig(T_mat)[0]), np.ones(self.num_links)))
        else:
            print('In check_consistency: No test implemented for case with pc')


    #
    # create_file_name
    #
    def create_file_name(self, obs):
        """
        Create the common part of file name:

        Parameters
        ----------
        self :
        obs :
        
        Returns
        -------
        L_W_pcs_x_y_a_..._obs_x1_x2_y1_y2_..._
        
        """
        L = self.geometry['L']
        W = self.geometry['W']
        bc = self.geometry['bc']
        pcs = self.pcs
        file_name = 'L'+str(L)+'_W'+str(W)+'_'+bc+'_pcs_'
        # pcs
        for c in pcs:
            [x,y,a]=c
            file_name += str(x)+'_'+str(y)+'_'+str(a)+'_'
        # obs
        file_name += 'obs_'
        for r in obs:
            [[x1,x2],[y1,y2],[a1,a2]]=r
            file_name += str(x1)+'_'+str(x2)+'_'+str(y1)+'_'+str(y2)+'_'\
                         +str(a1)+'_'+str(a2)+'_'

        return file_name

    ### End of the class network_class

#
# save_to_file: now part of compute_psics
#
# def save_psics_to_file(psics, net, ndis, obs, path=None):
#     """
#     Function that saves psics (supposed to be ndarray with shape
#     = (ndis, net.num_pcs, net.num_links) to file using numpy save
#     obs is the observation region, supposed to be 
#     rectangle(s) [[[x1,x2],[y1,y2]],...] with x2>x1, y2>y1 etc:
#      y2------------------------
#           |               | 
#           |     obs       | 
#           |               |
#      y1---|--------------------
#           |               |
#           x1              x2 
#     path: directory where file saved

#     Parameters:
#     -----------
#     psics :
#     net :
#     path :
#     obs :

#     Returns:
#     --------
#     Nothing
#     """

#     if path==None: # if not provided use the data/ folder
#         path='data/'
#     file_name = path+'/psics_'
#     file_name += net.create_file_name(obs)
#     file_name += 'ndis'+str(ndis)+'_'

#     # append an index, given by smallest available integer 
#     text=sorted(glob.glob(file_name+"*"))
#     new_ind = len(text) + 1
#     print('In save_psics_to_file: new_ind = ',  new_ind)
#     file_name += 'ind'+str(new_ind)+'.npy'
#     f = open(file_name, 'w')
#     # Save psics
#     np.save(file_name, psics)
#     print('saved to ', file_name)
#     f.close()

#    # end of the routine

