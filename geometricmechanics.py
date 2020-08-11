"""
Geometric Mechanics Module
__author__ = "Gil Miranda"
__credits__ = ["Alejandro Cabrera, Iago Leal"]
__license__ = "MIT"
__version__ = "1.0.1"
__email__ = "gil@matematica.ufrj.br"
__status__ = "Development"
"""

import numpy as np
import pyquaternion as pyQ
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import juggle_axes
import matplotlib.animation as animation

class deformableBody:
    def __init__(self, r_0, masses, p_0, rot, n, body_lines):
        self.r_0 = r_0
        self.masses = np.array(masses)
        self.p_0 = p_0
        self.q_0 = [1,0,0,0]
        self.rot = rot
        self.body_lines = body_lines
        self.r = np.copy(self.r_0)
        self.n = n
        self.positions = np.empty([len(masses), 3, self.n])
     
    def translation_CM(self):
        r_cm = Physics.CM(self.r, self.masses)
        self.r -= r_cm
        
    def set_positions(self, q, r, tmax):
        for k, rot in enumerate(np.transpose(q)):
            t = k*tmax/self.n
            p = Physics.particles(r, t)
            for i in range(0, len(self.masses)):
                q = pyQ.Quaternion(rot)
                self.positions[i, : ,k] = q.rotate(p[i, :])

class Physics:
    
    @staticmethod
    def get_InertialTensor(r, m):
        I = np.empty((3,3))

        I[0][0] = np.dot(m,np.sum(r[:,1:3]**2, axis = 1))
        I[1][1] = np.dot(m,np.sum(r[:,[True, False, True]]**2, axis = 1))
        I[2][2] = np.dot(m,np.sum(r[:,0:2]**2, axis = 1))

        I[0][1] = np.dot(m,-np.multiply(r[:,0],r[:,1]))
        I[0][2] = np.dot(m,-np.multiply(r[:,0],r[:,2]))
        I[1][2] = np.dot(m,-np.multiply(r[:,1],r[:,2]))
        I[1][0] = I[0][1]
        I[2][0] = I[0][2]
        I[2][1] = I[1][2]
        return I

    @staticmethod
    def get_InternalAngularMomentum(Q, v, m):
        s = m @ np.cross(Q, v)
        return s
    
    @staticmethod
    def CM(r, masses):
        cm = r.T @ masses
        total_m = np.sum(masses)
        cm /= total_m
        return cm
    
    @staticmethod
    def particles(rot, t):
        particle = np.array([r(t) for r in rot])
        return particle
    
    @staticmethod
    def eqOfMotion(x, t, rot, masses):
        # q -> quaternion for rotation
        # p -> body angular momentum
        q = pyQ.Quaternion(x[:4])
        p = x[4:]
        # Position
        pos = Physics.particles(rot, t)

        #Velocities
        ep = 1e-7
        v = (Physics.particles(rot, t+ep) - Physics.particles(rot, t-ep))/(2*ep)

        # Tensor of Inertia
        I = Physics.get_InertialTensor(pos, masses)
        Iinv = np.linalg.inv(I)

        # Angular Momentum
        L = Physics.get_InternalAngularMomentum(pos, v, masses)
        dumb = np.dot(Iinv, (p-L))
        dp_dt = np.cross(p,dumb)

        omega_i = np.dot(Iinv, (p-L))
        omega = pyQ.Quaternion(0, omega_i[0], omega_i[1], omega_i[2])
        dq_dt = 0.5 * q * omega
        return [dq_dt[0],dq_dt[1],dq_dt[2],dq_dt[3]] + [dp_dt[0], dp_dt[1], dp_dt[2]]
    
    @staticmethod
    def solve_eq(p_0, t, rot, masses):
        args = (rot, masses)
        q = scipy.integrate.odeint(Physics.eqOfMotion,[1,0,0,0] + p_0,t, args)
        q = np.transpose(q)
        q = q[:4,:]
        return q   

class Graphics:

    @staticmethod
    def update_plot(num, positions, body_lines, ani_lines, sc):
        for line, (sp, ep) in zip(ani_lines, body_lines):
            line._verts3d = positions[[sp,ep], :, num].T.tolist()
        sc._offsets3d = juggle_axes(positions[:,0, num], positions[:,1, num], positions[:,2, num], 'z')
        return sc

    @staticmethod
    def create_plot(positions, body_lines, time, name = 'cubeQ', save = True):
        #Create figure object and set animation
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d([-1.0, 1.0])
        ax.set_ylim3d([-1.0, 1.0])
        ax.set_zlim3d([-1.0, 1.0])
        ani_lines = [ax.plot([], [], [], 'k-')[0] for _ in body_lines]


        sc = ax.scatter3D(positions[:,0, 0], positions[:,1, 0], positions[:,2, 0], marker='o', c='red', s = 60)


        ani = FuncAnimation(fig, Graphics.update_plot,  frames=len(time), interval=2,
                fargs=(positions, body_lines, ani_lines, sc), repeat=True)
        plt.show()
        if save == True:
            ani.save('./'+ name + '.gif', writer='imagemagick')