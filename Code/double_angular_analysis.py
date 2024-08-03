# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:20:28 2024

@author: abel_
"""


exp_path = parent_dir + '/experiments/double_angular_test/N128_T128_noisy/recttanh/';
exp_i=0
params_path = glob.glob(exp_path + '/param*.yml')[0]; training_kwargs = yaml.safe_load(Path(params_path).read_text()); 
exp_list = glob.glob(exp_path + "/res*")
exp = exp_list[exp_i]
net, result = load_net_path(exp, which='post')
n_rec = net.dims[1]
wi, wrec, wo, brec, h0, oth = result['weights_last']
net.noise_std=0;
h_init='random'



T=1*128; batch_size=4096;
task = double_angularintegration_task(T=T, dt=dt, speed_range=[0.,0.], sparsity=1, random_angle_init='equally_spaced', constant_speed=True);
input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size); output_angle1 = np.arctan2(output[...,1], output[...,0]); output_angle2 = np.arctan2(output[...,3], output[...,2])

max_eigv = []
max_eigv_2nd = []
eigv_rest = []
for i,x in enumerate(init_states[::16,:]):
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