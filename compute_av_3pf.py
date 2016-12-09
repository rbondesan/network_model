"""compute_av_3pf.py: a python program that sums over the partial sums
and normalize by ndis. Calls the module network.

Created by Roberto Bondesan, Oct 20, 2016.

"""

import numpy as np
import network
import argparse
import glob, re
import warnings, sys

#
# ret_psi_av_3pf
#
def ret_psi_av_3pf(net, obs, ind_obs1, ind_obs2, q1, q2, path):
    """
    Compute the average  E(|psi|^(2q1)|psi|^(2q2)) by summing the psums

    only if pcs=1 at the moment. psis_abs shape = (ndis, nlinks)
    
    save to file
    """
    if path==None: # if not provided use the data/ folder
        path='data/'
    file_name = path+'/psum_psi_q1_'+str(q1)+'_psi_q2_'+str(q2)+'_av_'
    file_name += net.create_file_name(obs)
    file_name += 'indobs_'+str(ind_obs1)+'_'+str(ind_obs2)+'_'

    # list to save the number of disorder realizations summed over in partial sum
    ndis = 0 # tot dis realizations
    ret = 0 # sum of psums
    for in_file_name in sorted(glob.glob(file_name+"*")):
        try:
            # first find how many disorder realizations
            try:
                cur_ndis = int(re.search('ndis(.+?)_', \
                                         in_file_name).group(1))
            except AttributeError:
                print('Cannot determine ndis, skip ', in_file_name)
                continue
            # else: 
            # cur_psiq1_psiq2 is a size_R0 x size_R1 tensor given by summing 
            # over cur_ndis realizations
            cur_psiq1_psiq2 = np.load(in_file_name)
            ret += cur_psiq1_psiq2
            ndis += cur_ndis
            print('In ret_psi_av_3pf, loaded ', in_file_name)
        except IOError:
            print('In ret_psi_av_3pf: Problem with reading file ',\
                  in_file_name)

    # Compute E(|psi|^2q1 |psi|^2q2)
    ret = ret/ndis
    # Save to file
    file_name = path+'/psi_q1_'+str(q1)+'_psi_q2_'+str(q2)+'_av_'
    file_name += net.create_file_name(obs)
    file_name += 'indobs_'+str(ind_obs1)+'_'+str(ind_obs2)+'_'
    file_name += 'ndis'+str(ndis)+'.npy'
    np.save(file_name, ret)
    print('In compute_psi_av_3pf: saved E(|psi|^2q1 |psi|^2q2) to ', file_name)

    # return nothing

#
# main
#
def main(L, W, bc, pcs, obs, ind_obs1, ind_obs2, q1, q2, path):
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

    ret_psi_av_3pf(net, obs, ind_obs1, ind_obs2, q1, q2, path)
    
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
    [q1,q2] = vars(args)['powers']
    [ind_obs1,ind_obs2] = vars(args)['indobs']

    print(obs,ind_obs1,ind_obs2,q1,q2)
    
    main(L, W, bc, pcs, obs, ind_obs1, ind_obs2, q1, q2, path)
