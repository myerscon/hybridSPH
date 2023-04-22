import numpy as np
import matplotlib.pyplot as plt

# PySPH base and carray imports
from pysph.base.utils import get_particle_array as gpa

# Riemann imports
from pysph.examples.gas_dynamics.riemann_2d_config import R2DConfig
from custom_application import CustomApplication

# current case from the al possible unique cases
case = 4

# config for current case
config = R2DConfig(case)
#gamma = 1.4
#gamma1 = gamma - 1
#kernel_factor = 1.5
#dt = 1e-4
dim = 2
 
class Riemann2D(CustomApplication):
    def __init__(self,gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params) -> None:
        #update
        self.pfreq = pfreq
        self.tf = tf
        self.dt = dt
        super().__init__(gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)

class CustomRiemann(Riemann2D):
    def __init__(self,dx,xmin,xmax,ymin,ymax,gamma,kf,DoDomain,mirror_x,mirror_y,xcntr,ycntr,r_init,gaussian,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params) -> None:
        #update
        self.dx = dx
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.gamma = gamma
        self.gamma1 = gamma - 1
        self.kernel_factor = kf
        self.xcntr = xcntr
        self.ycntr = ycntr
        self.r_init = r_init # unneccessary in this problem
        self.gaussian = gaussian
        super().__init__(gamma,DoDomain,mirror_x,mirror_y,adaptive,cfl,pfreq,tf,dt,scheme,scheme_params)
    
    def create_particles(self):
        # taken from create_particles_constant_volume
        dx = self.dx
        dx2 = dx * 0.5
        vol = dx * dx

        xmin = self.xmin #config.xmin
        ymin = self.ymin #config.ymin
        xmax = self.xmax #config.xmax
        ymax = self.ymax #config.ymax
        xmid = self.xcntr #config.xmid
        ymid = self.ycntr #config.ymid

        self.rho1, self.u1, self.v1, self.p1 = config.rho1, config.u1, config.v1, config.p1
        self.rho2, self.u2, self.v2, self.p2 = config.rho2, config.u2, config.v2, config.p2
        self.rho3, self.u3, self.v3, self.p3 = config.rho3, config.u3, config.v3, config.p3
        self.rho4, self.u4, self.v4, self.p4 = config.rho4, config.u4, config.v4, config.p4
        x, y = np.mgrid[xmin+dx2:xmax:dx, ymin+dx2:ymax:dx]

        x = x.ravel()
        y = y.ravel()

        u = np.zeros_like(x)
        v = np.zeros_like(x)

        # density and mass
        rho = np.ones_like(x)
        p = np.ones_like(x)

        for i in range(x.size):
            if x[i] <= xmid:
                if y[i] <= ymid:  # w3
                    rho[i] = self.rho3
                    p[i] = self.p3
                    u[i] = self.u3
                    v[i] = self.v3
                else:            # w2
                    rho[i] = self.rho2
                    p[i] = self.p2
                    u[i] = self.u2
                    v[i] = self.v2
            else:
                if y[i] <= ymid:  # w4
                    rho[i] = self.rho4
                    p[i] = self.p4
                    u[i] = self.u4
                    v[i] = self.v4
                else:            # w1
                    rho[i] = self.rho1
                    p[i] = self.p1
                    u[i] = self.u1
                    v[i] = self.v1

        # thermal energy
        e = p/(self.gamma1 * rho)

        # mass
        m = vol * rho

        # smoothing length
        h = self.kernel_factor * (m/rho)**(1./dim)

        # Add extra props
        #additional_props = ['dwdh', 'ah', 'arho', 'grhoz', 'omega', 'div', 'grhox', 'converged', 'grhoy','cs','orig_idx','ae',
        #                    'uy', 'px', 'wx', 'wy', 'vx', 'ux', 'py', 'uz', 'wz', 'vz', 'pz', 'vy']

        # create the particle array
        pa = gpa(name='fluid', x=x, y=y, m=m, rho=rho, h=h,
                 u=u, v=v, p=p, e=e, h0=h.copy()) # , additional_props=additional_props
        self.scheme.setup_properties([pa])

        print("Riemann 2D with %d particles"
              % (pa.get_number_of_particles()))

        return [pa, ]

    def boundary_set(self):
        """ Sets wall particles to initial state
        Args:
            self
        Returns:
            None
        """
        boundary_ids = np.extract(self.particles[0].particle_type_id == 1.1, self.particles[0].id)

        upper_ids = np.extract(self.particles[0].y[boundary_ids] > self.ycntr, self.particles[0].id[boundary_ids])
        lower_ids = np.extract(self.particles[0].y[boundary_ids] < self.ycntr, self.particles[0].id[boundary_ids])

        upper_right_ids = np.extract(self.particles[0].x[upper_ids] > self.xcntr, self.particles[0].id[upper_ids])
        upper_left_ids = np.extract(self.particles[0].x[upper_ids] < self.xcntr, self.particles[0].id[upper_ids])
        lower_right_ids = np.extract(self.particles[0].x[lower_ids] > self.xcntr, self.particles[0].id[lower_ids])
        lower_left_ids = np.extract(self.particles[0].x[lower_ids] < self.xcntr, self.particles[0].id[lower_ids])

        self.particles[0].rho[upper_right_ids] = self.rho1
        self.particles[0].p[upper_right_ids] = self.p1
        self.particles[0].u[upper_right_ids] = self.u1 # 0
        self.particles[0].v[upper_right_ids] = self.v1

        self.particles[0].rho[upper_left_ids] = self.rho2
        self.particles[0].p[upper_left_ids] = self.p2
        self.particles[0].u[upper_left_ids] = self.u2
        self.particles[0].v[upper_left_ids] = self.v2

        self.particles[0].rho[lower_right_ids] = self.rho4
        self.particles[0].p[lower_right_ids] = self.p4
        self.particles[0].u[lower_right_ids] = self.u4
        self.particles[0].v[lower_right_ids] = self.v4
        
        self.particles[0].rho[lower_left_ids] = self.rho3
        self.particles[0].p[lower_left_ids] = self.p3
        self.particles[0].u[lower_left_ids] = self.u3
        self.particles[0].v[lower_left_ids] = self.v3

    def boundary_trim(self,x_walls=[-0.0,1.0],y_walls=[-0.0,1.0]):
        """ Deletes non-boundary particles in boundary regions
        Args:
            self
        Returns:
            None
        """
        real_ids = np.extract(self.particles[0].particle_type_id == 2, self.particles[0].id)

        b_ids = np.extract(self.particles[0].y[real_ids]<y_walls[0], self.particles[0].id)
        t_ids = np.extract(self.particles[0].y[real_ids]>y_walls[1], self.particles[0].id)

        del_ids = np.concatenate((b_ids,t_ids))

        self.particles[0].particle_type_id[del_ids] = 0
        #self.remove_flagged_particles()

        l_ids = np.extract(self.particles[0].y[real_ids]<x_walls[0], self.particles[0].id)
        r_ids = np.extract(self.particles[0].y[real_ids]>x_walls[1], self.particles[0].id)

        del_ids2 = np.concatenate((l_ids,r_ids))

        self.particles[0].particle_type_id[del_ids2] = 0
        self.remove_flagged_particles()