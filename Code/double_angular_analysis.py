# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:20:28 2024

@author: abel_
"""
import os, sys
import glob
import pickle
import yaml
from pathlib import Path, PurePath
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir) 
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

import numpy as np

from angular_loss_analysis import load_net_path, simulate_rnn_with_task, get_rnn_ode, find_periodic_orbits
from tasks import double_angularintegration_task
from odes import recttanh_jacobian_point

exp_path = parent_dir + '/experiments/double_angular_test/N128_T128_noisy/recttanh/';
exp_i=0
params_path = glob.glob(exp_path + '/param*.yml')[0]; training_kwargs = yaml.safe_load(Path(params_path).read_text()); 
exp_list = glob.glob(exp_path + "/res*")
exp = exp_list[exp_i]
net, result = load_net_path(exp, which='post')
training_kwargs = result['training_kwargs']
n_rec = net.dims[1]
wi, wrec, wo, brec, h0, oth = result['weights_last']
net.noise_std=0;
h_init='random'

##########simulate trajectories
T=1*128; batch_size=4096;
dt=0.1
task = double_angularintegration_task(T=T, dt=dt, speed_range=[0.,0.], sparsity=1, random_angle_init='equally_spaced', constant_speed=True);
input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size); output_angle1 = np.arctan2(output[...,1], output[...,0]); output_angle2 = np.arctan2(output[...,3], output[...,2])

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
trajectories_flat = trajectories[:,:,:].reshape((-1,trajectories.shape[-1]))
pca.fit(trajectories_flat)
trajectories_flat = trajectories[:,30:500:100,:].reshape((-1,trajectories.shape[-1]));
trajectories_pca = pca.fit_transform(trajectories_flat)
trajectories_pca_time = trajectories_pca.reshape((batch_size,-1,10))

trajectories_start_flat = trajectories[:,0,:].reshape((-1,trajectories.shape[-1]))



#########MSE analysis
angles = np.linspace(-np.pi, np.pi, int(np.sqrt(batch_size)))
theta, phi = np.meshgrid(angles, angles)
angle_error1 = np.abs(output_angle1 - theta.flatten()[:, np.newaxis])
angle_error2 = np.abs(output_angle2 - phi.flatten()[:, np.newaxis])


#eigenspectrum
max_eigv = []
max_eigv_2nd = []
eigv_rest = []
init_states = trajectories[:,0,:]
for i,x in enumerate(init_states[::16,:]):
    ########replace accordingly
    get_rnn_ode(training_kwargs['nonlinearity'])

    J = recttanh_jacobian_point(wrec,brec,1,x)
    eigenvalues, eigenvectors = np.linalg.eig(J)
    eigenvalues = sorted(np.real(eigenvalues))
    max_eigv.append(eigenvalues[-1])
    max_eigv_2nd.append(eigenvalues[-2])
    eigv_rest.extend(eigenvalues[:-2])

plt.hist(max_eigv,color='b',alpha=.5, density=True);plt.hist(max_eigv_2nd,color='purple',alpha=.5, density=True);plt.hist(eigv_rest,color='k',alpha=.5, density=True);

max_eigv = []
max_eigv_2nd = []
for i,x in enumerate(init_states[::16,:]):
    J = recttanh_jacobian_point(wrec,brec,1,x)
    eigenvalues, eigenvectors = np.linalg.eig(J)
    eigenvalues = sorted(np.real(eigenvalues))
    max_eigv.append(eigenvalues[-1])
    max_eigv_2nd.append(eigenvalues[-2])
    plt.scatter([i]*n_rec, eigenvalues, s=1, c='k', marker='o', alpha=0.5); 
plt.plot(max_eigv, 'b', label='1st')
plt.plot(max_eigv_2nd, 'purple', label='2nd')
plt.xlabel(r'$\theta$')
plt.ylabel('eigenvalue spectrum')
plt.xticks([0,len(csx)], [0,r'$2\pi$'])
plt.ylim([-1.5,0.2])
plt.legend(loc='lower right')
plt.hlines(0, 0,len(csx), 'r', linestyles='dotted');
plt.savefig(exp_path+f'/eigenvalue_spectrum_{exp_i}.pdf')




################VF uniform norm
rnn_ode = get_rnn_ode(training_kwargs['nonlinearity'])

diff = np.zeros((trajectories_start_flat.shape[0]))
diff1 = np.zeros((trajectories_start_flat.shape[0]))
diff2 = np.zeros((trajectories_start_flat.shape[0]))
for i in range(trajectories_start_flat.shape[0]):
    x,y,z,a=np.dot(wo.T,trajectories_start_flat[i,:])
    u,v,n,m=np.dot(wo.T, rnn_ode(0, trajectories_start_flat[i], wrec, brec, tau=10))
    diff[i] = np.linalg.norm(np.array([u,v,n,m]))
    
    diff1[i] = np.linalg.norm(np.array([u,v]))
    diff2[i] = np.linalg.norm(np.array([n,m]))
    
vf_norm_1 = np.max(diff1)
vf_norm_2 = np.max(diff2)




#####################FPS
from sklearn.cluster import DBSCAN
recurrences, recurrences_pca = find_periodic_orbits(trajectories, trajectories_pca, limcyctol=1e-2, mindtol=1e-10)
fxd_pnts = np.array([recurrence[0] for recurrence in recurrences if len(recurrence)==1 or len(recurrence)==11]).reshape((-1,n_rec));
lcs =[recurrence for recurrence in recurrences if len(recurrence)!=1 and len(recurrence)>11];
epsilon = 0.2  # Distance threshold
db = DBSCAN(eps=epsilon, min_samples=1).fit(fxd_pnts);
unique_indices = np.unique(db.labels_, return_index=True)[1]
unique_points = fxd_pnts[unique_indices]

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111); 
ax.grid(True)
#ax.scatter(theta[:,:],phi[:,:],s=1); 
#ax.scatter(angle1[:,-10:],angle2[:,-10:],s=5,c='orange',alpha=0.5)
max_eigs = []
for fp in unique_points:
    rec_out = np.dot(np.array(fp),wo)
    rec_angle1 = np.mod(np.arctan2(rec_out[...,1], rec_out[...,0]),2*np.pi)
    rec_angle2 = np.mod(np.arctan2(rec_out[...,3], rec_out[...,2]),2*np.pi)

    J = recttanh_jacobian_point(wrec,brec,1,fp)    
    eigenvalues, eigenvectors = np.linalg.eig(J)
    eigenvalues = sorted(np.real(eigenvalues))
    max_eigs.append(eigenvalues[-1])
    if eigenvalues[-1]>0.:
        ax.scatter(rec_angle1,rec_angle2, c='r', s=5, alpha=1,zorder=10)
    else:
        ax.scatter(rec_angle1,rec_angle2, c='g', s=5, alpha=1,zorder=10)
    
ticks = np.linspace(0, 2 * np.pi, 9)  # 9 ticks from 0 to 2π
ax.set_xticks(ticks)
ax.set_xticklabels(['0' if tick == 0 else r'$2\pi$' if tick == 2 * np.pi else '' for tick in ticks])
ax.set_yticks(ticks)
ax.set_yticklabels(['0' if tick == 0 else r'$2\pi$' if tick == 2 * np.pi else '' for tick in ticks])
plt.savefig(exp_path+'/output_2D_fps.pdf');



fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(121); 
for i in range(batch_size):
    ax.plot(trajectories_pca_time[i,:,0], trajectories_pca_time[i,:,2],'k',zorder=-10,alpha=.3);
ax.scatter(trajectories_pca_time[:,-1,0], trajectories_pca_time[:,-1,2],s=1);
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax = fig.add_subplot(122);
for i in range(batch_size):
    ax.plot(trajectories_pca_time[i,:,1], trajectories_pca_time[i,:,3],'k',zorder=-10,alpha=.3);
ax.scatter(trajectories_pca_time[:,-1,1], trajectories_pca_time[:,-1,3],s=1);
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True)); plt.savefig(exp_path+'/torus_exp0_02_13.pdf');