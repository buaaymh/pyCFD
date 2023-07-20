import abc
import numpy as np

import equation
import euler
import gas

class RiemannSolver(abc.ABC):
    
    @abc.abstractmethod
    def flux(self, u):
        pass

    @abc.abstractmethod
    def geoSource(self, U, x, alpha):
        pass

    @abc.abstractmethod
    def eval_flux(self, u_l, u_r):
        pass

    @abc.abstractmethod
    def num_equations(self):
        pass

class Linear(RiemannSolver):

    def __init__(self, a):
        self.a = a

    def flux(self, u):
        return self.a * u

    def eval_flux(self, u_l, u_r):
        flux = 0.0
        if self.a > 0.0:
            flux = self.flux(u_l)
        else:
            flux = self.flux(u_r)
        return flux

    def num_equations(self):
        return 1


class Burgers(RiemannSolver):

    def __init__(self):
        pass

    def flux(self, u):
        return u**2 / 2

    def eval_flux(self, u_l, u_r):
        u = 0.0
        if u_l > u_r:
            u = self._shockwave(u_l, u_r)
        elif u_l < u_r:
            u = self._rarefaction(u_l, u_r)
        else: # u_l == u_r
            u = u_l
        return self.flux(u)

    def num_equations(self):
        return 1

    def _shockwave(self, u_l, u_r):
        u_mean = (u_l + u_r) / 2
        u = 0.0
        if u_mean > 0:
            u = u_l
        else:
            u = u_r
        return u
    
    def _rarefaction(self, u_l, u_r):
        u = 0.0
        if u_l > 0.0:
            u = u_l
        elif u_r < 0.0:
            u = u_r
        else:
            u = 0.0
        return u

class EulerAusm(RiemannSolver):

    def __init__(self, gamma=1.4):
        self._gas = gas.Ideal(gamma)
        self._equation = equation.Euler1d(gamma)

    def num_equations(self):
        return 3

    def flux(self, U):
        return self._equation.F(U)

    def eval_flux(self, U_l, U_r):
        flux_positive = self.eval_positive_flux(U_l)
        flux_negative = self.eval_negative_flux(U_r)
        return flux_positive+flux_negative

    def eval_reflected_flux(self, U, normal):
        U_b = U
        U_b[1] *= -1
        if normal > 0:
            return self.eval_flux(U, U_b)
        else:
            return self.eval_flux(U_b, U)

    def eval_positive_flux(self, U):
        u, p, rho = self._equation.U_to_u_p_rho(U)
        a = self._gas.p_rho_to_a(p, rho)
        mach = u / a
        mach_positive = mach
        h = a**2 / self._gas.gamma_minus_1() + u**2 * 0.5
        flux = np.asarray([1, u, h])
        if (mach >= -1 and mach <= 1):
            mach_positive = (mach + 1)**2 * 0.25
            p *= (mach + 1) * 0.5
        elif (mach < -1):
            mach_positive = 0.0
            p = 0.0
        flux *= rho * a * mach_positive
        flux[1] += p
        return flux
    
    def eval_negative_flux(self, U):
        u, p, rho = self._equation.U_to_u_p_rho(U)
        a = self._gas.p_rho_to_a(p, rho)
        mach = u / a
        mach_negative = mach
        h = a**2 / self._gas.gamma_minus_1() + u**2 * 0.5
        flux = np.asarray([1, u, h])
        if (mach >= -1 and mach <= 1):
            mach_negative = -(mach - 1)**2 * 0.25
            p *= -(mach - 1) * 0.5
        elif (mach > 1):
            mach_negative = 0.0
            p = 0.0
        flux *= rho * a * mach_negative
        flux[1] += p
        return flux

class EulerVanLeer(RiemannSolver):

    def __init__(self, gamma=1.4):
        self._gas = gas.Ideal(gamma)
        self._equation = equation.Euler1d(gamma)

    def num_equations(self):
        return 3

    def flux(self, U):
        return self._equation.F(U)

    def eval_flux(self, U_l, U_r):
        flux_positive = self.eval_positive_flux(U_l)
        flux_negative = self.eval_negative_flux(U_r)
        return flux_positive+flux_negative

    def geoSource(self, U, x, alpha):
        u = U[1]/U[0]
        p = (U[2] - u*U[1]/2) * self._gas.gamma_minus_1()
        return -(alpha/x)*np.array([U[1], U[1]*u, u*(U[2]+p)])

    def eval_reflected_flux(self, U, normal):
        U_b = U
        U_b[1] *= -1
        if normal > 0:
            return self.eval_flux(U, U_b)
        else:
            return self.eval_flux(U_b, U)

    def eval_positive_flux(self, U):
        u, p, rho = self._equation.U_to_u_p_rho(U)
        a = self._gas.p_rho_to_a(p, rho)
        mach = u / a
        if mach >= 1:
          return self.flux(U)
        elif mach < -1:
          return np.zeros(3)
        flux = np.array([1, 2*a/self._gas.gamma(), 2*a**2 / (self._gas.gamma_plus_1()*self._gas.gamma_minus_1())])
        temp = self._gas.gamma_minus_1() * mach * 0.5 + 1
        flux[1] *= temp
        flux[2] *= temp ** 2
        flux *= rho * a * (1 + mach)**2 * 0.25
        return flux
    
    def eval_negative_flux(self, U):
        u, p, rho = self._equation.U_to_u_p_rho(U)
        a = self._gas.p_rho_to_a(p, rho)
        mach = u / a
        if mach <= -1:
          return self.flux(U)
        elif mach > 1:
          return np.zeros(3)
        flux = np.array([1, 2*a/self._gas.gamma(), 2*a**2/(self._gas.gamma_plus_1()*self._gas.gamma_minus_1())])
        temp = self._gas.gamma_minus_1() * mach * 0.5 - 1
        flux[1] *= temp
        flux[2] *= temp ** 2
        flux *= rho * a * (1 - mach)**2 * -0.25
        return flux
  
if __name__ == '__main__':
    solver = Linear(a=2.0)
    n = solver.num_equations()
    print(n == 1)
    # contact discontinuity
    u_l = 1.0
    u_r = 2.0
    flux = solver.eval_flux(u_l, u_r)
    print("contact", flux, "==", solver.flux(u_l), "\n")
    
    solver = Burgers()
    n = solver.num_equations()
    print(n == 1)
    # expansion wave
    u_l = -1.0
    u_r = 2.0
    flux = solver.eval_flux(u_l, u_r)
    print("middle_rare", flux, "==", solver.flux(0.0))
    u_l = -2.0
    u_r = -1.0
    flux = solver.eval_flux(u_l, u_r)
    print("  left_rare", flux, "==", solver.flux(u_r))
    u_l = 1.0
    u_r = 2.0
    flux = solver.eval_flux(u_l, u_r)
    print(" right_rare", flux, "==", solver.flux(u_l),"\n")

    # shock wave
    u_l = 2.0
    u_r = 1.0
    flux = solver.eval_flux(u_l, u_r)
    print("right_shock", flux, "==", solver.flux(u_l))
    u_l = -1.0
    u_r = -2.0
    flux = solver.eval_flux(u_l, u_r)
    print(" left_shock", flux, "==", solver.flux(u_r))

    solver = EulerVanLeer()
    euler = equation.Euler1d(gamma=1.4)
    # Sod
    print("Sod")
    U_l = euler.u_p_rho_to_U(u=0, p=1.0, rho=1.0)
    U_r = euler.u_p_rho_to_U(u=0, p=0.1, rho=0.125)
    flux = solver.eval_flux(U_l, U_r)
    print(flux)
    U = euler.u_p_rho_to_U(u=+0.927453, p=0.303130, rho=0.426319)
    flux = solver.flux(U)
    print(flux, "\n")
    # ShockCollision
    print("ShockCollision")
    U_l = euler.u_p_rho_to_U(u=19.5975,  p=460.894, rho=5.99924)
    U_r = euler.u_p_rho_to_U(u=-6.19633, p=46.0950, rho=5.99242)
    flux = solver.eval_flux(U_l, U_r)
    print(flux)
    U = euler.u_p_rho_to_U(u=+19.5975, p=460.894, rho=5.99924)
    flux = solver.flux(U)
    print(flux, "\n")
    # BlastFromLeft
    print("BlastFromLeft")
    U_l = euler.u_p_rho_to_U(u=0, p=1000,  rho=1)
    U_r = euler.u_p_rho_to_U(u=0, p=0.01, rho=1)
    flux = solver.eval_flux(U_l, U_r)
    print(flux)
    U = euler.u_p_rho_to_U(u=19.59745, p=460.8938, rho=0.575062)
    flux = solver.flux(U)
    print(flux, "\n")
    # BlastFromRight
    print("BlastFromRight")
    U_l = euler.u_p_rho_to_U(u=0, p=0.01, rho=1)
    U_r = euler.u_p_rho_to_U(u=0, p=100,  rho=1)
    flux = solver.eval_flux(U_l, U_r)
    print(flux)
    U = euler.u_p_rho_to_U(u=-6.196328, p=46.09504, rho=0.575113)
    flux = solver.flux(U)
    print(flux, "\n")
    # AlmostVacuumed
    print("AlmostVacuumed")
    U_l = euler.u_p_rho_to_U(u=-3, p=0.4, rho=1)
    U_r = euler.u_p_rho_to_U(u=+3, p=0.4, rho=1)
    flux = solver.eval_flux(U_l, U_r)
    print(flux)
    U = euler.u_p_rho_to_U(u=0.0, p=0.001894, rho=0.21852)
    flux = solver.flux(U)
    print(flux, "\n")






    