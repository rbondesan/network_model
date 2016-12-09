"""compute_psum_2pf.py: a python program to compute the partial sum
of |psi(l)|^{2q} saved in a given file. 
Does this for l in some observation region
Calls the module network.

Created by Roberto Bondesan, Oct 20, 2016.

"""

import numpy as np
import network
import argparse
import glob, re
import warnings, sys
import os

#
# ret_psum
#
def ret_psum_2pf(net, obs, ind_obs, q, in_file_name, path):
    """
    Returns partial sum
    psum = sum(|psi(l)|^(2q),disorder) for l in a given observation region only,
    whose index is ind_obs.
    
    save the results on file

    only if pcs=1 at the moment. psis_abs shape = (ndis, nlinks)
    
    """
    o=np.array(obs) # easier to manipulate
    # size of psic for a given disorder realization and point contact
    size_Rs = (o[:,0,1]-o[:,0,0])*(o[:,1,1]-o[:,1,0])*(o[:,2,1]-o[:,2,0])
    start_obs = sum(size_Rs[0:ind_obs]) # index of psi where obs reg starts
    end_obs = sum(size_Rs[0:ind_obs+1]) # index of psi where obs reg ends
    # reshape so that we can vstack
    psics = np.array([])
    psics = psics.reshape(0,net.num_pcs,size_Rs[ind_obs]) 
    # first find how many disorder realizations and
    # ndump for this file
    try:
        try:
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
        file_name = path+'/psum_psi_q_'+str(q)+'_av_'
        file_name += net.create_file_name(obs)
        file_name += 'indobs_'+str(ind_obs)+'_'
        file_name += 'ndis'+str(ndis)+'_ind'+str(ind)+'.npy'
        if os.path.isfile(file_name):
            # Then nothing to do...
            print('In ret_psum_2pf: file', file_name, \
                  ' already exists, exit')
            return 
        # else:
        narrays = int(ndis/ndump)
        with open(in_file_name, 'rb') as f:                    
            for i in range(narrays):
                cur_psics=np.load(f)
#                print('c',cur_psics.shape)
                # Here cur_psics.shape = (ndump, 1, sum(size_Rs)),
                # extract the right observation region:
                cur_psics=cur_psics[:,:,start_obs:end_obs]
                psics=np.vstack((psics, cur_psics))
#                print(psics.shape)
        print('In compute_psum_2pf, loaded ', in_file_name)
        # At this point psics has shape
        # (ndis, 1, size obs1 + size obs 2 + ...=size_psi_c_fixed)
        # compute partial sums and store number of elements summed
        psis_abs = np.squeeze(np.abs(psics))
        # Double check that the ndis in file name corresponds to
        # actual saved data (fails if interrupted)
        assert(ndis == psis_abs.shape[0])
        psis_q = np.power(psis_abs,2*q)
        # where to save the partial sum of |psi(l)|^q
        psum = np.sum(psis_q, axis=0)
        # Save to file psum
        np.save(file_name, psum)
        print('In compute_psum_2pf: saved psum (|psi|^2q) to ', file_name)
    except IOError:
        print('In compute_psum_2pf: Problem with reading file ',\
              in_file_name)

    # return nothing

#
# main
#
def main(L, W, bc, pcs, obs, ind_obs, q, in_file_name, path):
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

    ret_psum_2pf(net, obs, ind_obs, q, in_file_name, path)
    
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
    parser.add_argument('-i', '--indobs', type=int,
                        help='which observation region of the ones provided\
                        shall be used')
    parser.add_argument('-f', '--fname', type=str, help='in file name')
    parser.add_argument('-p', '--path', type=str, 
                        help='path to directory where to save',
                        default='data/')
    parser.add_argument('-q', '--power', type=float, 
                        help='if q s.t. E(|psi|^q). For moment, float',
                        default=1.0)

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
    ind_obs = vars(args)['indobs']
    path = vars(args)['path']
    in_file_name = vars(args)['fname']
    q = vars(args)['power']

    print(obs,ind_obs,q,in_file_name)

    main(L, W, bc, pcs, obs, ind_obs, q, in_file_name, path)
