# -*- coding: utf-8 -*-
"""
N-body problem simulation for matplotlab. Based on
https://fiftyexamples.readthedocs.org/en/latest/gravity.html
"""
import argparse
import itertools
from math import atan2, sin, cos
from time import sleep

import numpy as np

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


G = 6.67428e-11
AU = (149.6e6 * 1000)
SUN_MASS = 1.98892 * 10**30
ONE_DAY = 24*3600


class Body(object):

    def __init__(self, name, mass, p, v=(0.0, 0.0, 0.0)):
        self.name = name
        self.mass = mass
        self.p = p
        self.v = v
        self.f = np.array([0.0, 0.0, 0.0])

    def __str__(self):
        return 'Body {}'.format(self.name)

    def attraction(self, other):
        """Calculate the force vector between the two bodies"""
        assert self is not other
        diff_vector = other.p - self.p
        distance = norm(diff_vector)
        print distance
        assert np.abs(distance) > 10**4, 'Bodies collided!'
        f_tot = G * self.mass * other.mass / (distance**2)
        f = f_tot * diff_vector / norm(diff_vector)
        return f


def norm(x):
    return np.sqrt(np.sum(x**2))


def move(bodies, timestep):
    pairs = itertools.combinations(bodies, 2)
    # Initialize force vectors
    for b in bodies:
        b.f = np.array([0.0, 0.0, 0.0])
    # Calculate force vectors
    for b1, b2 in pairs:
        f = b1.attraction(b2)
        b1.f += f
        b2.f -= f
    # Update velocities based on force, update positions based on velocity
    for body in bodies:
        body.v += body.f / body.mass * timestep
        body.p += body.v * timestep
        print body.name, body.p, body.v, body.f
    print ''


def points_for_bodies(bodies):
    x0 = np.array([body.p[0] for body in bodies])
    y0 = np.array([body.p[1] for body in bodies])
    z0 = np.array([body.p[2] for body in bodies])
    return x0, y0, z0


def norm_forces_for_bodies(bodies, norm_factor):
    u0 = np.array([body.f[0] for body in bodies])
    v0 = np.array([body.f[1] for body in bodies])
    w0 = np.array([body.f[2] for body in bodies])
    return u0/norm_factor, v0/norm_factor, w0/norm_factor


class AnimatedScatter(object):

    def __init__(self, bodies, axis_range, timescale):
        self.bodies = bodies
        self.axis_range = axis_range
        self.timescale = timescale
        self.stream = self.data_stream()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.fig = fig
        self.ax = ax
        self.force_norm_factor = None
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=10,
                                           init_func=self.setup_plot, blit=False)

    def setup_plot(self):
        xi, yi, zi, ui, vi, wi = next(self.stream)
        c = ['r', 'g', 'b']
        self.scatter = self.ax.scatter(xi, yi, zi, c=c, s=200)
        self.quiver = self.ax.quiver(xi, yi, zi, ui, vi, wi, length=1)

        FLOOR = self.axis_range[0]
        CEILING = self.axis_range[1]
        self.ax.set_xlim3d(FLOOR, CEILING)
        self.ax.set_ylim3d(FLOOR, CEILING)
        self.ax.set_zlim3d(FLOOR, CEILING)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        return self.scatter, self.quiver

    def quiver_force_norm_factor(self):
        axis_length = np.abs(self.axis_range[1]) + np.abs(self.axis_range[0])
        return np.amax(np.array([b.f for b in self.bodies]))/(axis_length/10)

    def data_stream(self):
        while True:
            move(self.bodies, self.timescale)
            if not self.force_norm_factor:
                self.force_norm_factor = self.quiver_force_norm_factor()
                print 'factor ', self.force_norm_factor
            x, y, z = points_for_bodies(self.bodies)
            u, v, w = norm_forces_for_bodies(self.bodies, self.force_norm_factor)
            yield x, y, z, u, v, w

    def update(self, i):
        x_i, y_i, z_i, u_i, v_i, w_i = next(self.stream)

        self.scatter._offsets3d = (x_i, y_i, z_i)

        segments = np.array([(b.p, b.p + b.f/self.force_norm_factor) for b in self.bodies])
        self.quiver.set_segments(segments)

        plt.draw()
        return self.scatter, self.quiver

    def show(self):
        plt.show()


def parameters_for_simulation(simulation_name):

    if simulation_name == 'home3':
        sun = Body('sun', mass=SUN_MASS, p=np.array([0.0, 0.0, 0.0]))
        earth = Body('earth', mass=5.9742 * 10**24, p=np.array([-1*AU, 0.0, 0.0]), v=np.array([0.0, 29.783 * 1000, 0.0]))
        venus = Body('venus', mass=4.8685 * 10**24, p=np.array([0.723 * AU, 0.0, 0.0]), v=np.array([0.0, -35.0 * 1000, 0.0]))
        axis_range = (-1.2*AU, 1.2*AU)
        timescale = ONE_DAY
        return (sun, earth, venus), axis_range, timescale

    elif simulation_name == 'misc3':
        """
        This is basically mostly just a demonstration that this
        simulation doesn't respect conservation laws.
        """
        sun1 = Body('A', mass=6.0 * 10**30, p=np.array([4.0*AU, 0.5*AU, 0.0]), v=np.array([-10.0 * 1000, -1.0 * 1000, 0.0]))
        sun2 = Body('B', mass=8.0 * 10**30, p=np.array([-6.0*AU, 0.0, 3.0*AU]), v=np.array([5.0 * 1000, 0.0, 0.0]))
        sun3 = Body('C', mass=10.0 * 10**30, p=np.array([0.723 * AU, -5.0 * AU, -1.0*AU]), v=np.array([-10.0 * 1000, 0.0, 0.0]))
        axis_range = (-20*AU, 20*AU)
        timescale = ONE_DAY
        return (sun1, sun2, sun3), axis_range, timescale

    elif simulation_name == 'centauri3':
        """
        Based on known orbit dimensions and masses.
        Not working, not sure why. They shouldn't get farther than 36AU
        or about 5e12m away from each other.
        """
        p_a = np.array([-7.5711 * 10**11, 0.0, 0.0])
        p_b = np.array([9.1838 * 10**11, 0.0, 0.0])
        v_a = np.array([0.0, 1.212 * 10**4, 0.0])
        v_b = np.array([0.0, -1.100 * 10**4, 0.0])
        alphaA = Body('Alpha A', mass=1.100*SUN_MASS, p=p_a, v=v_a)
        alphaB = Body('Alpha B', mass=0.907*SUN_MASS, p=p_b, v=v_b)
        axis_range = (-10.0**13, 10.0**13)
        timescale = ONE_DAY * 5
        return (alphaA, alphaB), axis_range, timescale

    else:
        raise ValueError('No simulation named {}'.format(simulation_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation', help='home3, misc3, or centauri3')
    args = parser.parse_args()
    bodies, axis_range, timescale = parameters_for_simulation(args.simulation)
    a = AnimatedScatter(bodies, axis_range, timescale)
    a.show()

if __name__ == '__main__':
    main()
