"""compute_psum_3pf.py: a python program to compute the partial sum of
wave functions saved in a given file.  Requires two observation
regions.  Calls the module network.

Created by Roberto Bondesan, Oct 20, 2016.

"""

import numpy as np
import network
import argparse
import glob, re
import warnings, sys
import os

#
# ret_psum_3pf
#
def ret_psum_3pf(net, obs, ind_obs1, ind_obs2, q1, q2, in_file_name, path):
    """
    Returns partial sums 
    psum = sum(|psi(l)|^(2 q1) |psi(l')|^(2 q2),disorder)
    for l in observation ind_obs, l' in observation ind_obs2
    
    save the results on file

    only if pcs=1 at the moment. psis_abs shape = (ndis, nlinks)
    
    """
    o=np.array(obs) # easier to manipulate
    assert(ind_obs1 < ind_obs2) # Assumes ind_obs1 < ind_obs2
    # size of psic for a given disorder realization and point contact
    size_Rs = (o[:,0,1]-o[:,0,0])*(o[:,1,1]-o[:,1,0])*(o[:,2,1]-o[:,2,0])
    start_obs1 = sum(size_Rs[0:ind_obs1]) # index of psi where obs 1 reg starts
    end_obs1 = sum(size_Rs[0:ind_obs1+1]) # index of psi where obs 1 reg ends
    start_obs2 = sum(size_Rs[0:ind_obs2]) # index of psi where obs 2 reg starts
    end_obs2 = sum(size_Rs[0:ind_obs2+1]) # index of psi where obs 2 reg ends
    size_psi_c_fixed = sum(size_Rs) 
    print(size_Rs)
    # reshape so that we can vstack
    psics = np.array([])
    psics = psics.reshape(0,net.num_pcs,size_psi_c_fixed) 
    # first find how many disorder realizations and
    # ndump for this file
    try:
        try:
            print(in_file_name)
            ndis = int(re.search('ndis(.+?)_', in_file_name).group(1))
            ndump = int(re.search('ndump(.+?)_', in_file_name).group(1))
            # Since there can be cylinder in string, match only if ind preceeded by _
            ind = int(re.search('(?<=_)ind(.+?).npy', in_file_name).group(1))            
        except AttributeError:
            print('Cannot determine ndis, ndump, ind: set to 1')
            ndis = 1 # apply error handling
            ndump = 1
            ind = 1
        # Use numpy routines to load
        if ndis % ndump != 0:
            print('ndump not multiple of ndis, skip ', in_file_name)
            sys.exit(1)

        # Files to save data
        if path==None: # if not provided use the data/ folder  
            path='data/'
        file_name = path+'/psum_psi_q1_'+str(q1)+'_psi_q2_'+str(q2)+'_av_'
        file_name += net.create_file_name(obs)
        file_name += 'indobs_'+str(ind_obs1)+'_'+str(ind_obs2)+'_'
        file_name += 'ndis'+str(ndis)+'_ind'+str(ind)+'.npy'
        if os.path.isfile(file_name):
            # Then nothing to do...
            print('In ret_psum_3pf: file',file_name, \
                  ' already exists, exit')
            return 
        # else:
        narrays = int(ndis/ndump)
        with open(in_file_name, 'rb') as f:                    
            for i in range(narrays):
                cur_psics=np.load(f)
                psics=np.vstack((psics, cur_psics))
        print('In load_psics_from_files, loaded ', in_file_name)
        psis_abs = np.squeeze(np.abs(psics))
        # Double check that the ndis in file name corresponds to
        # actual saved data (fails if interrupted)
        assert(ndis == psis_abs.shape[0])
        # Compute the partial sum of |psi(l)|^q1 |psi(l')|^q2 Here l
        # in obs1, l' in obs2
        psis_abs_R1 = psis_abs[:,start_obs1:end_obs1]
        psis_abs_R2 = psis_abs[:,start_obs2:end_obs2]
        psis_R1_q1 = np.power(psis_abs_R1,2*q1)
        psis_R2_q2 = np.power(psis_abs_R2,2*q2)
        psum = np.einsum('ij,ik', psis_R1_q1, psis_R2_q2)
        np.save(file_name, psum)
        print('In compute_psi_av: saved psum (|psi(R1)|^2q1 |psi(R2)|^2q2) to ',
              file_name)
    except IOError:
        print('In load_psics_from_files: Problem with reading file ',\
              in_file_name)
    # return nothing

#
# main
#
def main(L, W, bc, pcs, obs, ind_obs1, ind_obs2, q1, q2, in_file_name, path):
    # Set the parameters. See help of network_class for explanations
    geo = {'L' : L, 'W' : W, 'bc' : bc}
    ang = {'A' : np.pi/4., 'B' : np.pi/4.} # typically not to be modified
    # and create an instance of network class
    net = network.network_class(geo, pcs, ang, disorder=True)
    # only 1 pc at the moment
    # TODO: if npcs > 1, one needs to take the symmetric polynomial...
    if net.num_pcs > 1:
        print('npcs > 1 not implemented yet, exit')
        sys.exit(1)
        
    ret_psum_3pf(net, obs, ind_obs1, ind_obs2, q1, q2, in_file_name, path)
    
    print('End of the program.')
    # return nothing    

# When it is executed from command line
if __name__ == '__main__':
    # Get from command line
    parser = argparse.ArgumentParser(description='compute_corr')
    parser.add_argument('-L', '--length', type=int, help='length of the network')
    parser.add_argument('-W', '--width', type=int, help='width of the network')
    parser.add_argument('-b', '--bc', type=str, help='boundary condition')
    parser.add_argument('-c', '--pcs', nargs='+', type=int,
                        help='position of point contacts as x1 y1 a1 ... ')
    parser.add_argument('-R', '--obs', nargs='+', type=int,
                        help='observation regions: one observation region is \
                        rectangle x1 x2 y1 y2 ...')
    parser.add_argument('-i', '--indobs', type=int, nargs=2,
                        help='which observation region2 of the ones provided\
                        shall be used for the two wave functions')
    parser.add_argument('-f', '--fname', type=str, help='in file name')
    parser.add_argument('-p', '--path', type=str, 
                        help='path to directory where to save',
                        default='data/')
    parser.add_argument('-q', '--powers', nargs=2, type=float, 
                        help='q1,q2 st E(|psi|^q1 * |psi|^q2). For moment, float')

    args = parser.parse_args()
    L = vars(args)['length']
    W = vars(args)['width']
    bc = vars(args)['bc']
    tmp = vars(args)['pcs']
    pcs = [tmp[i:i + 3] for i in range(0, len(tmp), 3)]
    path = vars(args)['path']
    tmp = vars(args)['obs']
    tmp = [tmp[i:i + 2] for i in range(0, len(tmp), 2)]
    obs = [tmp[i:i + 3] for i in range(0, len(tmp), 3)]
    path = vars(args)['path']
    [ind_obs1,ind_obs2] = vars(args)['indobs']
    in_file_name = vars(args)['fname']
    [q1,q2] = vars(args)['powers']

    print(obs,ind_obs1,ind_obs2,q1,q2,in_file_name)
    
    main(L, W, bc, pcs, obs, ind_obs1, ind_obs2, q1, q2, in_file_name, path)
