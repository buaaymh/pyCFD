import abc
import numpy as np

from time_scheme import RK3ForFDM
from equation import Euler1d
import gas

class NumericalSolver(abc.ABC):

    def set_mesh(self, x_min, x_max, x_num):
        self.x_min = x_min
        self.x_max = x_max
        self.x_num = x_num
        self.dx = (x_max - x_min) / x_num
        self.x_vec = np.linspace(start=x_min + self.dx/2,
                                 stop =x_max - self.dx/2, num=x_num)

    def set_time_stepper(self, start, stop, t_num):
        self.t_num = t_num
        self.dt = (stop - start) / t_num
        self.t_vec = np.linspace(start=start, stop=stop, num=t_num+1)

    def set_initial_condition(self, func):
        self.initial_func = func

    def set_boundary_condition(self, boundary="periodic"):
        self.boundary = boundary
    
    def set_riemann_solver(self, riemann):
        self.riemann = riemann

    def eval_rhs(self, u_vec):
        rhs = np.zeros_like(u_vec)
        if (self.boundary == "periodic"):
            self._dual_periodic(u_vec, rhs)
        elif (self.boundary == "free"):
            self._dual_free(u_vec, rhs)
        elif (self.boundary == "reflect"):
            self._dual_reflect(u_vec, rhs)
        return rhs

    def get_x_vec(self):
        return self.x_vec
    
    def get_t_vec(self):
        return self.t_vec
    
    def get_u_mat(self):
        return self.u_mat

class FDMSolver(NumericalSolver):

    def set_mesh(self, x_min, x_max, x_num):
        self.x_min = x_min
        self.x_max = x_max
        self.x_num = x_num
        self.dx = (x_max - x_min) / x_num
        self.x_vec = np.linspace(start=x_min + self.dx/2,
                                 stop =x_max - self.dx/2, num=x_num)

    def run_with_transient(self):
        self.u_mat = np.zeros((self.x_num, self.riemann.num_equations()))
        for i in range(self.x_num):
            self.u_mat[i] = self.initial_func(self.x_vec[i])
        # time march
        self._time_scheme.set_rhs(rhs=lambda u_vec: self.eval_rhs(u_vec))
        for i in range(1, self.t_num+1):
            u_old = self.u_mat
            self.u_mat = self._time_scheme.get_u_new(u_old=u_old, dt=self.dt)
    
    def run_with_animation(self):
        self.u_mat = np.zeros((self.t_num+1, self.x_num, self.riemann.num_equations()))
        for i in range(self.x_num):
            self.u_mat[0, i] = self.initial_func(self.x_vec[i])
        # time march
        self._time_scheme.set_rhs(rhs=lambda u_vec: self.eval_rhs(u_vec))
        for i in range(1, self.t_num+1):
            self.u_mat[i] = self._time_scheme.get_u_new(u_old=self.u_mat[i-1], dt=self.dt)

class FirstOrderScheme(FDMSolver):

    def __init__(self):
        self._time_scheme = RK3ForFDM()
    
    def _dual_periodic(self, u_vec, rhs):
        flux = np.zeros_like(u_vec)
        for i in range(len(u_vec)):
            flux[i] = self.riemann.eval_flux(u_vec[i-1], u_vec[i])
        for i in range(len(u_vec)):
            rhs[i-1] = flux[i-1] - flux[i]
        rhs /= self.dx

    def _dual_free(self, u_vec, rhs):
        flux = np.zeros((len(u_vec)+1, self.riemann.num_equations()))
        flux[0] = self.riemann.flux(u_vec[0])
        flux[-1] = self.riemann.flux(u_vec[-1])
        for i in range(1, len(u_vec)):
            flux[i] = self.riemann.eval_flux(u_vec[i-1], u_vec[i])
        for i in range(len(u_vec)):
            rhs[i] = flux[i] - flux[i+1]
        rhs /= self.dx

    def _dual_reflect(self, u_vec, rhs):
        flux = np.zeros((len(u_vec)+1, self.riemann.num_equations()))
        flux[0] = self.riemann.eval_reflected_flux(u_vec[0], -1)
        flux[-1] = self.riemann.eval_reflected_flux(u_vec[-1], 1)
        for i in range(1, len(u_vec)):
            flux[i] = self.riemann.eval_flux(u_vec[i-1], u_vec[i])
        for i in range(len(u_vec)):
            rhs[i] = flux[i] - flux[i+1]
        rhs /= self.dx

class WenoScalarScheme(FDMSolver):

    def __init__(self):
        self._time_scheme = RK3ForFDM()
        self._13_12 = 13.0/12
        self._1_3 = 1.0/3
        self._1_6 = 1.0/6
        self._5_6 = 5.0/6
        self._7_6 = 7.0/6
        self._11_6 = 11.0/6

    def _dual_periodic(self, u_vec, rhs):
        flux = np.zeros_like(u_vec)
        for i in range(len(u_vec)):
            i -= 2
            u_l = self._get_weno_u_l(u_vec, i-1)
            u_r = self._get_weno_u_r(u_vec, i)
            flux[i] = self.riemann.eval_flux(u_l, u_r)
        for i in range(len(u_vec)):
            rhs[i-1] = flux[i-1] - flux[i]
        rhs /= self.dx

    def _dual_free(self, u_vec, rhs):
        pass

    def _dual_reflect(self, u_vec, rhs):
        pass

    def _get_weno_u_l(self, u_vec, i):
        is_1 = (0.250 * (u_vec[i-2] -
                     4 * u_vec[i-1] +
                     3 * u_vec[i]) ** 2 +
                self._13_12 * (u_vec[i-2] -
                     2 * u_vec[i-1] +
                         u_vec[i]) ** 2)
        is_2 = (0.250 * (u_vec[i-1] -
                         u_vec[i+1]) ** 2 +
                self._13_12 * (u_vec[i-1] -
                     2 * u_vec[i] +
                         u_vec[i+1]) ** 2)
        is_3 = (0.250 * (3 * u_vec[i] -
                         4 * u_vec[i+1] +
                             u_vec[i+2]) ** 2 +
                self._13_12 * (u_vec[i] -
                     2 * u_vec[i+1] +
                         u_vec[i+2]) ** 2)
        alpha_1 = 0.1 / (1e-6 + is_1) ** 2
        alpha_2 = 0.6 / (1e-6 + is_2) ** 2
        alpha_3 = 0.3 / (1e-6 + is_3) ** 2
        alpha_sum = alpha_1 + alpha_2 + alpha_3
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        omega_3 = alpha_3 / alpha_sum
        u_1 = ( self._1_3 * u_vec[i-2] -
                self._7_6 * u_vec[i-1] +
               self._11_6 * u_vec[i])
        u_2 = (-self._1_6 * u_vec[i-1] +
                self._5_6 * u_vec[i] +
                self._1_3 * u_vec[i+1])
        u_3 = ( self._1_3 * u_vec[i] +
                self._5_6 * u_vec[i+1] -
                self._1_6 * u_vec[i+2])
        u = u_1 * omega_1 + u_2 * omega_2 + u_3 * omega_3
        return u

    def _get_weno_u_r(self, u_vec, i):
        is_1 = (0.250 * (u_vec[i+2] -
                         u_vec[i+1] * 4 +
                         u_vec[i  ] * 3) ** 2 +
                self._13_12 * (u_vec[i+2] -
                         u_vec[i+1] * 2 +
                         u_vec[i]) ** 2)
        is_2 = (0.250 * (u_vec[i+1] -
                         u_vec[i-1]) ** 2 +
                self._13_12 * (u_vec[i+1] -
                         u_vec[i] * 2 +
                         u_vec[i-1]) ** 2)
        is_3 = (0.250 * (u_vec[i] * 3 -
                         u_vec[i-1] * 4 +
                         u_vec[i-2]) ** 2 +
                self._13_12 * (u_vec[i] -
                         u_vec[i-1] * 2 +
                         u_vec[i-2]) ** 2)
        alpha_1 = 0.1 / (1e-6 + is_1) ** 2
        alpha_2 = 0.6 / (1e-6 + is_2) ** 2
        alpha_3 = 0.3 / (1e-6 + is_3) ** 2
        alpha_sum = alpha_1 + alpha_2 + alpha_3
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        omega_3 = alpha_3 / alpha_sum
        u_1 = (self._1_3 * u_vec[i+2] -
               self._7_6 * u_vec[i+1] +
              self._11_6 * u_vec[i])
        u_2 = (-self._1_6 * u_vec[i+1] +
                self._5_6 * u_vec[i] +
                self._1_3 * u_vec[i-1])
        u_3 = (self._1_3 * u_vec[i] +
               self._5_6 * u_vec[i-1] -
               self._1_6 * u_vec[i-2])
        u = u_1 * omega_1 + u_2 * omega_2 + u_3 * omega_3
        return u

class WenoEulerScheme(FDMSolver):

    def __init__(self):
        self._time_scheme = RK3ForFDM()
        self._13_12 = 13.0/12
        self._1_3 = 1.0/3
        self._1_6 = 1.0/6
        self._5_6 = 5.0/6
        self._7_6 = 7.0/6
        self._11_6 = 11.0/6

    def _dual_periodic(self, u_vec, rhs):
        pass

    def _dual_free(self, u_vec, rhs):
        flux = np.zeros((len(u_vec)+1, self.riemann.num_equations()))
        U_in = self._get_weno_U_r(u_vec, 0)
        flux[0] = self.riemann.flux(U_in)
        U_out = self._get_weno_U_l(u_vec, len(u_vec)-1)
        flux[-1] = self.riemann.flux(U_out)
        for i in range(1, len(u_vec)):
            U_l = self._get_weno_U_l(u_vec, i-1)
            U_r = self._get_weno_U_r(u_vec, i)
            flux[i] = self.riemann.eval_flux(U_l, U_r)
        for i in range(len(u_vec)):
            rhs[i] = flux[i] - flux[i+1]
        rhs /= self.dx

    def _dual_reflect(self, u_vec, rhs):
        flux = np.zeros((len(u_vec)+1, self.riemann.num_equations()))
        U_in = self._get_weno_U_r(u_vec, 0)
        flux[0] = self.riemann.eval_reflected_flux(U_in, -1)
        U_out = self._get_weno_U_l(u_vec, len(u_vec)-1)
        flux[-1] = self.riemann.eval_reflected_flux(U_out, 1)
        F_positive = np.zeros((len(u_vec), self.riemann.num_equations()))
        F_negative = np.zeros((len(u_vec), self.riemann.num_equations()))
        for i in range(len(u_vec)):
            F_positive[i] = self.riemann.eval_positive_flux(u_vec[i])
            F_negative[i] = self.riemann.eval_negative_flux(u_vec[i])
        for i in range(1, len(u_vec)):
            F_l = self._get_weno_U_l(F_positive, i-1)
            F_r = self._get_weno_U_r(F_negative, i)
            flux[i] = F_l + F_r
        for i in range(len(u_vec)):
            rhs[i] = flux[i] - flux[i+1]
        rhs /= self.dx

    def _get_weno_U_l(self, u_vec, i):
        U = np.zeros(len(u_vec[0]))
        for j in range(len(u_vec[0])):
            U[j] = self._get_U_l(u_vec, i, j)
        return U

    def _get_weno_U_r(self, u_vec, i):
        U = np.zeros(len(u_vec[0]))
        for j in range(len(u_vec[0])):
            U[j] = self._get_U_r(u_vec, i, j)
        return U

    def _get_U_l(self, u_vec, i, j):
        is_1, is_2, is_3 = float("inf"), float("inf"), float("inf")
        if i > 1 and i < len(u_vec): 
          is_1 = (0.250 * (u_vec[i-2,j] - 4 * u_vec[i-1,j] + 3 * u_vec[i,j]) ** 2 +
                  self._13_12 * (u_vec[i-2,j] - 2 * u_vec[i-1,j] + u_vec[i,j]) ** 2)
        if i > 0 and i < len(u_vec) - 1:
          is_2 = (0.250 * (u_vec[i-1,j] - u_vec[i+1,j]) ** 2 +
                  self._13_12 * (u_vec[i-1,j] - 2 * u_vec[i,j] + u_vec[i+1,j]) ** 2)
        if i > -1 and i < len(u_vec) - 2:          
          is_3 = (0.250 * (3 * u_vec[i,j] - 4 * u_vec[i+1,j] + u_vec[i+2,j]) ** 2 +
                  self._13_12 * (u_vec[i,j] - 2 * u_vec[i+1,j] + u_vec[i+2,j]) ** 2)
        alpha_1 = 0.1 / (1e-6 + is_1) ** 2
        alpha_2 = 0.6 / (1e-6 + is_2) ** 2
        alpha_3 = 0.3 / (1e-6 + is_3) ** 2
        alpha_sum = alpha_1 + alpha_2 + alpha_3
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        omega_3 = alpha_3 / alpha_sum
        u_1, u_2, u_3 = 0, 0, 0
        if i > 1 and i < len(u_vec): 
          u_1 = ( self._1_3 * u_vec[i-2,j] - self._7_6 * u_vec[i-1,j] + self._11_6 * u_vec[i,j])
        if i > 0 and i < len(u_vec) - 1:
          u_2 = (-self._1_6 * u_vec[i-1,j] + self._5_6 * u_vec[i,j]   + self._1_3 * u_vec[i+1,j])
        if i > -1 and i < len(u_vec) - 2:          
          u_3 = ( self._1_3 * u_vec[i,j]   + self._5_6 * u_vec[i+1,j] - self._1_6 * u_vec[i+2,j])
        u = u_1 * omega_1 + u_2 * omega_2 + u_3 * omega_3
        return u

    def _get_U_r(self, u_vec, i, j):
        is_1, is_2, is_3 = float("inf"), float("inf"), float("inf")
        if i > -1 and i < len(u_vec) - 2:
            is_1 = (0.250 * (u_vec[i+2,j] - u_vec[i+1,j] * 4 + u_vec[i,j] * 3) ** 2 +
                    self._13_12 * (u_vec[i+2,j] - u_vec[i+1,j] * 2 + u_vec[i,j]) ** 2)
        if i > 0 and i < len(u_vec) - 1:
            is_2 = (0.250 * (u_vec[i+1,j] - u_vec[i-1,j]) ** 2 +
                    self._13_12 * (u_vec[i+1,j] - u_vec[i,j] * 2 + u_vec[i-1,j]) ** 2)
        if i > 1 and i < len(u_vec):
            is_3 = (0.250 * (u_vec[i,j] * 3 - u_vec[i-1,j] * 4 + u_vec[i-2,j]) ** 2 +
                    self._13_12 * (u_vec[i,j] - u_vec[i-1,j] * 2 + u_vec[i-2,j]) ** 2)
        alpha_1 = 0.1 / (1e-6 + is_1) ** 2
        alpha_2 = 0.6 / (1e-6 + is_2) ** 2
        alpha_3 = 0.3 / (1e-6 + is_3) ** 2
        alpha_sum = alpha_1 + alpha_2 + alpha_3
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        omega_3 = alpha_3 / alpha_sum
        u_1, u_2, u_3 = 0, 0, 0
        if i > -1 and i < len(u_vec) - 2:
            u_1 = (self._1_3 * u_vec[i+2,j] - self._7_6 * u_vec[i+1,j] + self._11_6 * u_vec[i,j])
        if i > 0 and i < len(u_vec) - 1:
            u_2 =(-self._1_6 * u_vec[i+1,j] + self._5_6 * u_vec[i,j]   + self._1_3 * u_vec[i-1,j])
        if i > 1 and i < len(u_vec):
            u_3 = (self._1_3 * u_vec[i,j]   + self._5_6 * u_vec[i-1,j] - self._1_6 * u_vec[i-2,j])
        u = u_1 * omega_1 + u_2 * omega_2 + u_3 * omega_3
        return u

    

if __name__ == '__main__':
    # Set Scheme:
    # solver = FirstOrderScheme()
    # solver = WenoScalarScheme()
    solver = WenoEulerScheme()

    # Set Initial Condition:
    from equation import Euler1d
    euler = Euler1d(gamma=1.4)
    # U_l = euler.u_p_rho_to_U(u=0.75, p=1.0, rho=1.0)
    # U_r = euler.u_p_rho_to_U(u=0, p=0.1, rho=0.125)
    U_l = euler.u_p_rho_to_U(u=0.698, p=3.528, rho=0.445)
    U_r = euler.u_p_rho_to_U(u=0, p=0.571, rho=0.5)
    # def initial(x):
    #     if x < 0.5:
    #       return U_l
    #     else:
    #       return U_r
    # def initial(x):
    #     if x < -4:
    #         return euler.u_p_rho_to_U(u=2.629369, p=10.333333, rho=3.857143)
    #     else:
    #         return euler.u_p_rho_to_U(u=0, p=1, rho=1+0.2*np.sin(5*x))
    def initial(x):
        U_l = euler.u_p_rho_to_U(u=0, p=1000, rho=1)
        U_m = euler.u_p_rho_to_U(u=0, p=0.01, rho=1)
        U_r = euler.u_p_rho_to_U(u=0, p=100, rho=1)
        if x < 0.1:
            return U_l
        elif x >= 0.1 and x < 0.9:
            return U_m
        else:
            return U_r

    # Set Mesh:
    x_min = 0.0
    x_max = 1.0
    x_num = 50
    solver.set_mesh(x_min = x_min, x_max = x_max, x_num = x_num)
    solver.set_initial_condition(func=lambda x: initial(x))
    # solver.set_boundary_condition(boundary="free")
    solver.set_boundary_condition(boundary="reflect")
    # solver.set_boundary_condition(boundary="periodic")

    # Set Time Scheme:
    start = 0.0
    stop = 0.0380
    t_num = 190
    solver.set_time_stepper(start = start, stop = stop, t_num = t_num)

    # Set Riemann Problem:
    # from riemann import Linear
    # riemann = Linear(1.0)
    # from riemann import Burgers
    # riemann = Burgers()
    from riemann import EulerFVS
    riemann = EulerFVS()
    solver.set_riemann_solver(riemann = riemann)


    x_vec = solver.get_x_vec()
    t_vec = solver.get_t_vec()

    # solver.run_with_transient()
    # u_mat = solver.get_u_mat()

    # u_vec = np.zeros(len(u_mat))
    # p_vec = np.zeros(len(u_mat))
    # rho_vec = np.zeros(len(u_mat))
    # for i in range(len(u_mat)):
    #     u, p, rho = euler.U_to_u_p_rho(u_mat[i])
    #     u_vec[i], p_vec[i], rho_vec[i] = u, p, rho
    # from displayer import Transient
    # transient = Transient(u_label = "Density")
    # transient.add_plot(x_vec = x_vec, u_vec = rho_vec, type = "k", label = r'$\rho$')
    # # transient.add_plot(x_vec = x_vec, u_vec = u_vec, type = "r.", label = r'$u$')
    # # transient.add_plot(x_vec = x_vec, u_vec = p_vec, type = "b.", label = r'$p$')
    # transient.display()
    # # transient.save_fig("result.pdf")

    solver.run_with_animation()
    U_mat = solver.get_u_mat()
    u_vec = np.zeros((len(U_mat), len(U_mat[0])))
    p_vec = np.zeros((len(U_mat), len(U_mat[0])))
    rho_vec = np.zeros((len(U_mat), len(U_mat[0])))
    for i in range(len(U_mat)):
        for j in range(len(U_mat[0])):
            u, p, rho = euler.U_to_u_p_rho(U_mat[i, j])
            u_vec[i,j], p_vec[i,j], rho_vec[i,j] = u, p, rho
    from displayer import Animation
    animation = Animation(x_vec=x_vec, t_vec=t_vec, u_mat=rho_vec)
    animation.display(type = "--", y_min = 0.0, y_max = 7.4)

    # solver.run_on_charater()
    # u_mat = solver.get_u_mat()
    # v1_vec = np.zeros(len(u_mat))
    # v2_vec = np.zeros(len(u_mat))
    # v3_vec = np.zeros(len(u_mat))
    # for i in range(len(u_mat)):
    #     v1_vec[i], v2_vec[i], v3_vec[i] = u_mat[i]
    # from displayer import Transient
    # transient = Transient(u_label = "Density")
    # transient.add_plot(x_vec = x_vec, u_vec = v1_vec, type = "k", label = r'$\rho$')
    # transient.add_plot(x_vec = x_vec, u_vec = v2_vec, type = "r.", label = r'$u$')
    # transient.add_plot(x_vec = x_vec, u_vec = v3_vec, type = "b.", label = r'$p$')
    # transient.display()



            


