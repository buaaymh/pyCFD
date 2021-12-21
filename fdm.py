import abc
import numpy as np
from time import*

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
            if (i%500 == 0):
              print(i)
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
        self._Coef = np.array([[ 1/3, -7/6, 11/6],
                               [-1/6,  5/6,  1/3],
                               [ 1/3,  5/6, -1/6]])

    def _is_0(self, f):
      is_0 = (f[0]-2*f[1]+f[2])**2*self._13_12 + (f[0]-4*f[1]+3*f[2])**2*0.25
      return is_0
    def _is_1(self, f):
      is_1 = (f[0]-2*f[1]+f[2])**2*self._13_12 + (f[0]-f[2])**2*0.25
      return is_1
    def _is_2(self, f):
      is_2 = (f[0]-2*f[1]+f[2])**2*self._13_12 + (3*f[0]-4*f[1]+f[2])**2*0.25
      return is_2

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
        is_0 = self._is_0([u_vec[i-2],u_vec[i-1],u_vec[i]])
        is_1 = self._is_1([u_vec[i-1],u_vec[i  ],u_vec[i+1]])
        is_2 = self._is_2([u_vec[i  ],u_vec[i+1],u_vec[i+2]])
        alpha_0 = 0.1 / (1e-6 + is_0) ** 2
        alpha_1 = 0.6 / (1e-6 + is_1) ** 2
        alpha_2 = 0.3 / (1e-6 + is_2) ** 2
        alpha_sum = alpha_0 + alpha_1 + alpha_2
        omega_0 = alpha_0 / alpha_sum
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        u_0 = np.dot(np.transpose([u_vec[i-2],u_vec[i-1],u_vec[i]]),   self._Coef[0])
        u_1 = np.dot(np.transpose([u_vec[i-1],u_vec[i  ],u_vec[i+1]]), self._Coef[1])
        u_2 = np.dot(np.transpose([u_vec[i  ],u_vec[i+1],u_vec[i+2]]), self._Coef[2])
        u = u_0 * omega_0 + u_1 * omega_1 + u_2 * omega_2
        return u

    def _get_weno_u_r(self, u_vec, i):
        is_0 = self._is_0([u_vec[i+2],u_vec[i+1],u_vec[i]])
        is_1 = self._is_1([u_vec[i+1],u_vec[i],u_vec[i-1]])
        is_2 = self._is_2([u_vec[i],u_vec[i-1],u_vec[i-2]])
        alpha_0 = 0.1 / (1e-6 + is_0) ** 2
        alpha_1 = 0.6 / (1e-6 + is_1) ** 2
        alpha_2 = 0.3 / (1e-6 + is_2) ** 2
        alpha_sum = alpha_0 + alpha_1 + alpha_2
        omega_0 = alpha_0 / alpha_sum
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        u_0 = np.dot(np.transpose([u_vec[i+2],u_vec[i+1],u_vec[i]]),   self._Coef[0])
        u_1 = np.dot(np.transpose([u_vec[i+1],u_vec[i  ],u_vec[i-1]]), self._Coef[1])
        u_2 = np.dot(np.transpose([u_vec[i  ],u_vec[i-1],u_vec[i-2]]), self._Coef[2])
        u = u_0 * omega_0 + u_1 * omega_1 + u_2 * omega_2
        return u

class WenoEulerScheme(FDMSolver):

    def __init__(self):
        self._time_scheme = RK3ForFDM()
        self._euler = Euler1d(gamma=1.4)
        self._13_12 = 13.0/12
        self._Coef = np.array([[ 1/3, -7/6, 11/6],
                         [-1/6,  5/6,  1/3],
                         [ 1/3,  5/6, -1/6]])
    def _alpha_0(self, f):
      is_0 = (f[0]-2*f[1]+f[2])**2*self._13_12 + (f[0]-4*f[1]+3*f[2])**2*0.25
      alpha_0 = 0.1 / (1e-6 + is_0) ** 2
      return alpha_0
    def _alpha_1(self, f):
      is_1 = (f[0]-2*f[1]+f[2])**2*self._13_12 + (f[0]-f[2])**2*0.25
      alpha_1 = 0.6 / (1e-6 + is_1) ** 2
      return alpha_1
    def _alpha_2(self, f):
      is_2 = (f[0]-2*f[1]+f[2])**2*self._13_12 + (3*f[0]-4*f[1]+f[2])**2*0.25
      alpha_2 = 0.3 / (1e-6 + is_2) ** 2
      return alpha_2

    def _dual_periodic(self, u_vec, rhs):
        pass

    def _dual_free(self, u_vec, rhs):
        flux = np.zeros((len(u_vec)+1, self.riemann.num_equations()))
        U_r = np.dot(np.transpose([u_vec[2],u_vec[1],u_vec[0]]),self._Coef[0])
        flux[0] = self.riemann.flux(U_r)
        U_out = np.dot(np.transpose([u_vec[-3],u_vec[-2],u_vec[-1]]),self._Coef[0])
        flux[-1] = self.riemann.eval_reflected_flux(U_out,1)
        for i in range(1, len(u_vec)):
          L_p, L_n, R = self._get_roe_eigen(u_vec[i-1], u_vec[i])
          V_p = self._get_V_positive(u_vec, i-1, L_p)
          V_n = self._get_V_negative(u_vec, i  , L_n)
          flux[i] = np.dot(R, V_p+V_n)
        for i in range(len(u_vec)):
            rhs[i] = flux[i] - flux[i+1]
        rhs /= self.dx

    def _get_roe_eigen(self, U_l, U_r):
      L, R = np.zeros((3, 3)), np.ones((3, 3))
      u_l, p_l, rho_l = self._euler.U_to_u_p_rho(U_l)
      u_r, p_r, rho_r = self._euler.U_to_u_p_rho(U_r)
      h_l, h_r = (U_l[2] + p_l) / rho_l, (U_r[2] + p_r) / rho_r
      rho_l_sqrt, rho_r_sqrt = np.sqrt(rho_l), np.sqrt(rho_r)
      rho = ((rho_l_sqrt + rho_r_sqrt) * 0.5) ** 2
      u = (u_l * rho_l_sqrt + u_r * rho_r_sqrt) / (rho_l_sqrt + rho_r_sqrt)
      h = (h_l * rho_l_sqrt + h_r * rho_r_sqrt) / (rho_l_sqrt + rho_r_sqrt)
      c = np.sqrt(0.4*(h - u**2*0.5))
      b_1 = 0.4 / c**2
      b_2 = b_1 * u**2 * 0.5
      S = np.zeros((3,3))
      S[0,0], S[1,1], S[2,2] = u-c, u, u+c
      S_m = np.identity(3)*(abs(u)+c)
      S_p, S_n = (S + S_m) * 0.5, (S - S_m) * 0.5

      L[0,0] = (b_2 + u/c) * 0.5
      L[0,1] = -(b_1*u + 1/c) * 0.5
      L[0,2] = b_1 * 0.5
      L[1,0] = 1 - b_2
      L[1,1] = b_1 * u
      L[1,2] = -b_1
      L[2,0] = (b_2 - u/c) * 0.5
      L[2,1] = -(b_1*u - 1/c) * 0.5
      L[2,2] = b_1 * 0.5

      R[1,0] = u - c
      R[1,1] = u
      R[1,2] = u + c
      R[2,0] = h - c*u
      R[2,1] = u**2 * 0.5
      R[2,2] = h + c*u
      return np.dot(S_p, L), np.dot(S_n, L), R

    def _dual_reflect(self, u_vec, rhs):
        flux = np.zeros((len(u_vec)+1, self.riemann.num_equations()))
        U_in = np.dot(np.transpose([u_vec[2],u_vec[1],u_vec[0]]),self._Coef[0])
        flux[0] = self.riemann.eval_reflected_flux(U_in,-1)
        U_out = np.dot(np.transpose([u_vec[-3],u_vec[-2],u_vec[-1]]),self._Coef[0])
        flux[-1] = self.riemann.eval_reflected_flux(U_out,1)
        for i in range(1, len(u_vec)):
          L_p, L_n, R = self._get_roe_eigen(u_vec[i-1], u_vec[i])
          V_p = self._get_V_positive(u_vec, i-1, L_p)
          V_n = self._get_V_negative(u_vec, i  , L_n)
          flux[i] = np.dot(R, V_p+V_n)
        for i in range(len(u_vec)):
            rhs[i] = flux[i] - flux[i+1]
        rhs /= self.dx

    def _get_V_positive(self, U, i, L):
        if i == 0:
          V_vec = np.dot(U[0:4], np.transpose(L))
          V_2 = np.dot(np.transpose([V_vec[0],V_vec[1],V_vec[2]]), self._Coef[2])
          return V_2
        elif i == 1:
          V_vec = np.dot(U[0:5], np.transpose(L))
          V_0 = np.zeros(3)
          V_1 = np.dot(np.transpose([V_vec[0],V_vec[1],V_vec[2]]), self._Coef[1])
          V_2 = np.dot(np.transpose([V_vec[1],V_vec[2],V_vec[3]]), self._Coef[2])
          alpha_0 = np.zeros(3)
          alpha_1 = self._alpha_1([V_vec[1],V_vec[2],V_vec[3]])
          alpha_2 = self._alpha_2([V_vec[2],V_vec[3],V_vec[4]])
        elif i == len(U)-2:
          V_vec = np.dot(U[len(U)-4:len(U)], np.transpose(L))
          V_0 = np.dot(np.transpose([V_vec[0],V_vec[1],V_vec[2]]), self._Coef[0])
          V_1 = np.dot(np.transpose([V_vec[1],V_vec[2],V_vec[3]]), self._Coef[1])
          V_2 = np.zeros(3)
          alpha_0 = self._alpha_0([V_vec[0],V_vec[1],V_vec[2]])
          alpha_1 = self._alpha_1([V_vec[1],V_vec[2],V_vec[3]])
          alpha_2 = np.zeros(3)
        else:
          V_vec = np.dot(U[i-2:i+3], np.transpose(L))
          V_0 = np.dot(np.transpose([V_vec[0],V_vec[1],V_vec[2]]), self._Coef[0])
          V_1 = np.dot(np.transpose([V_vec[1],V_vec[2],V_vec[3]]), self._Coef[1])
          V_2 = np.dot(np.transpose([V_vec[2],V_vec[3],V_vec[4]]), self._Coef[2])
          alpha_0 = self._alpha_0([V_vec[0],V_vec[1],V_vec[2]])
          alpha_1 = self._alpha_1([V_vec[1],V_vec[2],V_vec[3]])
          alpha_2 = self._alpha_2([V_vec[2],V_vec[3],V_vec[4]])

        alpha_sum = alpha_0 + alpha_1 + alpha_2
        omega_0 = alpha_0 / alpha_sum
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        return V_0*omega_0 + V_1*omega_1 + V_2*omega_2
          
    def _get_V_negative(self, U, i, L):
        if i == 1:
          V_vec = np.dot(U[0:4], np.transpose(L))
          V_0 = np.dot(np.transpose([V_vec[3],V_vec[2],V_vec[1]]), self._Coef[0])
          V_1 = np.dot(np.transpose([V_vec[2],V_vec[1],V_vec[0]]), self._Coef[1])
          V_2 = np.zeros(3)
          alpha_0 = self._alpha_0([V_vec[3],V_vec[2],V_vec[1]])
          alpha_1 = self._alpha_1([V_vec[2],V_vec[1],V_vec[0]])
          alpha_2 = np.zeros(3)
        elif i == len(U)-2:
          V_vec = np.dot(U[len(U)-4:len(U)], np.transpose(L))
          V_0 = np.zeros(3)
          V_1 = np.dot(np.transpose([V_vec[3],V_vec[2],V_vec[1]]), self._Coef[1])
          V_2 = np.dot(np.transpose([V_vec[2],V_vec[1],V_vec[0]]), self._Coef[2])
          alpha_0 = np.zeros(3)
          alpha_1 = self._alpha_1([V_vec[3],V_vec[2],V_vec[1]])
          alpha_2 = self._alpha_2([V_vec[2],V_vec[1],V_vec[0]])
        elif i == len(U)-1:
          V_vec = np.dot(U[len(U)-3:len(U)], np.transpose(L))
          V_2 = np.dot(np.transpose([V_vec[2],V_vec[1],V_vec[0]]), self._Coef[2])
          return V_2
        else:
          V_vec = np.dot(U[i-2:i+3], np.transpose(L))
          V_0 = np.dot(np.transpose([V_vec[4],V_vec[3],V_vec[2]]), self._Coef[0])
          V_1 = np.dot(np.transpose([V_vec[3],V_vec[2],V_vec[1]]), self._Coef[1])
          V_2 = np.dot(np.transpose([V_vec[2],V_vec[1],V_vec[0]]), self._Coef[2])
          alpha_0 = self._alpha_0([V_vec[4],V_vec[3],V_vec[2]])
          alpha_1 = self._alpha_1([V_vec[3],V_vec[2],V_vec[1]])
          alpha_2 = self._alpha_2([V_vec[2],V_vec[1],V_vec[0]])
        alpha_sum = alpha_0 + alpha_1 + alpha_2
        omega_0 = alpha_0 / alpha_sum
        omega_1 = alpha_1 / alpha_sum
        omega_2 = alpha_2 / alpha_sum
        return V_0*omega_0 + V_1*omega_1 + V_2*omega_2

if __name__ == '__main__':
    # Set Scheme:
    # solver = FirstOrderScheme()
    # solver = WenoScalarScheme()
    solver = WenoEulerScheme()

    # Set Initial Condition:
    from equation import Euler1d
    euler = Euler1d(gamma=1.4)
    U_l = euler.u_p_rho_to_U(u=0, p=1.0, rho=1.0)
    U_r = euler.u_p_rho_to_U(u=0, p=0.1, rho=0.125)
    # U_l = euler.u_p_rho_to_U(u=0.698, p=3.528, rho=0.445)
    # U_r = euler.u_p_rho_to_U(u=0, p=0.571, rho=0.5)
    def initial(x):
        if x < 0.5:
          return U_l
        else:
          return U_r
    def initial(x):
        if x < 1:
            return euler.u_p_rho_to_U(u=2.629369, p=10.333333, rho=3.857143)
        else:
            return euler.u_p_rho_to_U(u=0, p=1, rho=1+0.2*np.sin(5*x))
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
    x_num = 5000
    solver.set_mesh(x_min = x_min, x_max = x_max, x_num = x_num)
    solver.set_initial_condition(func=lambda x: initial(x))
    solver.set_boundary_condition(boundary="free")
    solver.set_boundary_condition(boundary="reflect")
    # solver.set_boundary_condition(boundary="periodic")

    # Set Time Scheme:
    start = 0.0
    stop = 0.038
    t_num = 15000
    solver.set_time_stepper(start = start, stop = stop, t_num = t_num)

    # Set Riemann Problem:
    # from riemann import Linear
    # riemann = Linear(1.0)
    # from riemann import Burgers
    # riemann = Burgers()
    # from riemann import EulerAusm
    # riemann = EulerAusm()
    from riemann import EulerVanLeer
    riemann = EulerVanLeer()
    solver.set_riemann_solver(riemann = riemann)


    x_vec = solver.get_x_vec()
    t_vec = solver.get_t_vec()

    # from displayer import Animation
    # solver.run_with_animation()
    # u_mat = solver.get_u_mat()
    # animation = Animation(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
    # animation.display(type = "k--", y_min = -1, y_max = 3)

    begin = time()

    solver.run_with_transient()
    end = time()
    print('运行时长:', end-begin)
    u_mat = solver.get_u_mat()

    u_vec = np.zeros(len(u_mat))
    p_vec = np.zeros(len(u_mat))
    rho_vec = np.zeros(len(u_mat))
    for i in range(len(u_mat)):
        u, p, rho = euler.U_to_u_p_rho(u_mat[i])
        u_vec[i], p_vec[i], rho_vec[i] = u, p, rho
    from displayer import Transient
    transient = Transient(u_label = "Density")
    # transient.add_plot(x_vec = x_vec, u_vec = rho_vec, type = "k", label = r'$\rho$')
    # transient.add_plot(x_vec = x_vec, u_vec = u_vec, type = "r.", label = r'$u$')
    # transient.add_plot(x_vec = x_vec, u_vec = p_vec, type = "b.", label = r'$p$')
    # transient.display()
    import data
    filename = "result/Blast/weno_5000_15000.csv"
    data.write_rows(filename, [x_vec, rho_vec])
    

    # solver.run_with_animation()
    # U_mat = solver.get_u_mat()
    # u_vec = np.zeros((len(U_mat), len(U_mat[0])))
    # p_vec = np.zeros((len(U_mat), len(U_mat[0])))
    # rho_vec = np.zeros((len(U_mat), len(U_mat[0])))
    # for i in range(len(U_mat)):
    #     for j in range(len(U_mat[0])):
    #         u, p, rho = euler.U_to_u_p_rho(U_mat[i, j])
    #         u_vec[i,j], p_vec[i,j], rho_vec[i,j] = u, p, rho
    # from displayer import Animation
    # animation = Animation(x_vec=x_vec, t_vec=t_vec, u_mat=rho_vec)
    # animation.display(type = "--", y_min = 0.0, y_max = 7.4)




            


