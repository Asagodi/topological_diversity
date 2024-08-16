# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:56:45 2024

@author: abel_
"""


import numpy as np
from scipy.spatial.distance import cdist



########SPHERE

def sample_points_on_sphere(num_points, r=1):
    points = []
    for _ in range(num_points):
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)

        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)

        x = r*np.sin(phi) * np.cos(theta)
        y = r*np.sin(phi) * np.sin(theta)
        z = r*np.cos(phi)

        points.append((x, y, z))

    return np.array(points)

def uniform_points_on_sphere(num_points_on_equator=10, r=1):
    points = []
    for i in range(num_points_on_equator+1):
        phi = i*np.pi/num_points_on_equator
        if np.sin(phi)==0:
            num_points_on_phi_ring=1
        else:
            num_points_on_phi_ring = abs(int(num_points_on_equator/r*np.sin(phi)))
        if i==num_points_on_equator:
            num_points_on_phi_ring=1
        for j in range(num_points_on_phi_ring):
            theta = j*2*np.pi/num_points_on_phi_ring
            x = r*np.sin(phi) * np.cos(theta)
            y = r*np.sin(phi) * np.sin(theta)
            z = r*np.cos(phi)
            points.append((x, y, z))
            
    return np.array(points)

def grid_on_sphere():
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Then, stack these flattened arrays together and transpose to get (N, 3)
    points = np.vstack((x_flat, y_flat, z_flat)).T
    return points





########TORUS

def torus_4d_to_3d_thetas(theta1, theta2, r1=1, r2=.25):
    # Convert angles to 4D Cartesian coordinates
    x = (r1+r2*np.cos(theta1))*np.cos(theta2);
    y = (r1+r2*np.cos(theta1))*np.sin(theta2);
    z = r2*np.sin(theta1);

    return x,y,z


def get_manifold_from_closest_projections_torus(trajectories_flat, wo, npoints=128):

    n_rec = wo.shape[0]
    xs = np.arange(-np.pi, np.pi, 2*np.pi/npoints)
    xs = np.append(xs, -np.pi)
    ys = np.dot(trajectories_flat.reshape((-1,n_rec)), wo)
    circle_points = np.array([np.cos(xs), np.sin(xs)]).T
    torus_points = np.array(np.meshgrid(circle_points,circle_points))
    dists = cdist(torus_points.reshape((-1,4)), ys)
    csx2 = []
    for i in range(xs.shape[0]):
        csx2.append(trajectories_flat[np.argmin(dists[i,:]),:])
    csx2 = np.array(csx2)
    csx2_proj2 = np.dot(csx2, wo)
    return csx2, csx2_proj2




