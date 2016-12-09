"""#!/usr/bin/python"""

"""compute_psics.py: a python program to compute the scattering
wave function in the network model with point contacts.
Calls the module network.

Created by Roberto Bondesan, Oct 20, 2016.

"""

import numpy as np
import network
import argparse
import glob
import warnings, sys
# DEBUG
import time

def main(L, W, bc, pcs, ndis, ndump, obs, path):
    # Set the parameters. See help of network_class for explanations
    geo = {'L' : L, 'W' : W, 'bc' : bc}
    ang = {'A' : np.pi/4., 'B' : np.pi/4.} # typically not to be modified
    # and create an instance of network class
    net = network.network_class(geo, pcs, ang, disorder=True)

    # Determine size of psic for a given disorder realization and point contact
    o=np.array(obs) # easier to manipulate
    size_psi_c_fixed = sum( (o[:,0,1]-o[:,0,0])*(o[:,1,1]-o[:,1,0])*(o[:,2,1]-o[:,2,0]) ) 

    # Open file to save output
    if path==None: # if not provided use the data/ folder
        path='data/'
    file_name = path+'/psics_'
    file_name += net.create_file_name(obs)
    file_name += 'ndis'+str(ndis)+'_'
    # Check if ndis is a multiple of ndump, otherwise warning
    if ndis % ndump != 0 :
        print(ndis % ndump)
        warnings.warn("In compute_psics.main: ndump not multiple of ndis!")
        sys.exit(1)
    file_name += 'ndump'+str(ndump)+'_'
    # append an index, given by smallest available integer 
    text=sorted(glob.glob(file_name+"*"))
    file_name += 'ind'+str(len(text) + 1)+'.npy'
    f = open(file_name, 'wb')

    # loop over the number of disorder realizations
    mypsics = np.array([[]],dtype=net.dtype)
    for i in range(1,ndis+1): # so that i = 1,...,ndis
        psic = net.get_instance_of_psic()
        # reshape to a more handy object
        psic = psic.reshape(net.num_pcs,W,L,4)            
        for r in obs:
            [[x1,x2],[y1,y2],[a1,a2]]=r
            # append: psic[:,y1:y2,x1:x2,a1:a2] is reshaped to a vector 
            # of dim n_pc * (y2-y1) * (x2-x1) * (a2-a1)
            mypsics=np.append(mypsics, psic[:,y1:y2,x1:x2,a1:a2])

        # Dump every ndump times
        if i % ndump == 0:
            mypsics=mypsics.reshape(ndump,net.num_pcs,size_psi_c_fixed)
            # mypsics[i,c,:] is the i-th disorder realization,c-th point contact
            np.save(f, mypsics)
            # clear mypsics
            mypsics = np.array([[]],dtype=net.dtype)

    print('In compute_psics.main: saved to ', file_name)
    f.close()

    print('End of the program.')
    # return nothing    

# When it is executed from command line
if __name__ == '__main__':
    # Get from command line
    parser = argparse.ArgumentParser(description='compute_psics')
    parser.add_argument('-L', '--length', type=int, help='length of the network')
    parser.add_argument('-W', '--width', type=int, help='width of the network')
    parser.add_argument('-b', '--bc', type=str, help='boundary condition')
    parser.add_argument('-c', '--pcs', nargs='+', type=int,
                        help='position of point contacts as x1 y1 a1 ... ')
    parser.add_argument('-u', '--ndump', type=int, 
                        help='iterations after which dump data to file',
                        default=1)
    # Set ndump = number of disorder realizations after whcih dump data to file
    # keep it small in order not to load too much memory, also append is costly
    parser.add_argument('-d', '--ndis', type=int, 
                        help='number of disorder realizations',
                        default=1)
    parser.add_argument('-p', '--path', type=str, 
                        help='path to directory where to save',
                        default='data/')
    parser.add_argument('-R', '--obs', nargs='+', type=int,
                        help='observation regions: one observation region is a\
                        rectangle x1 x2 y1 y2 a1 a2, a slice psi[y1:y2,x1:x2,a1:a2]')
                        # where   
                        #   y2------------------------     \
                        #        |               |         \
                        #        |     obs       |         \
                        #        |               |         \
                        #   y1---|--------------------     \
                        #        |               |         \
                        #        x1              x2
    args = parser.parse_args()
    L = vars(args)['length']
    W = vars(args)['width']
    bc = vars(args)['bc']
    tmp = vars(args)['pcs']
    pcs = [tmp[i:i + 3] for i in range(0, len(tmp), 3)]
    print(pcs)
    ndis = vars(args)['ndis']
    ndump = vars(args)['ndump']
    path = vars(args)['path']
    tmp = vars(args)['obs']
    tmp = [tmp[i:i + 2] for i in range(0, len(tmp), 2)]
    obs = [tmp[i:i + 3] for i in range(0, len(tmp), 3)]
    print(obs)
    path = vars(args)['path']

    main(L, W, bc, pcs, ndis, ndump, obs, path)
