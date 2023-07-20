import abc
import csv
import pandas as pd
import numpy as np
import math
from fv_cell import VrP2Cell
from vr_approach import VrP2Approach

def function(x, k):
    return np.cos(k*x), np.sin(k*x)


class Reconstructer(abc.ABC):

    @abc.abstractmethod
    def du_dx(self, u_vec):
        pass
    
    def sepctral_du_dx_(self, uR_vec, uI_vec):
        return self.du_dx(uR_vec), self.du_dx(uI_vec)

class Upwind1st(Reconstructer):

    def __init__(self, x_vec, dx):
        self.x_vec = x_vec
        self.dx    = dx

    def du_dx(self, u_vec):
        v_vec = np.zeros(len(u_vec))
        for i in range(len(self.x_vec)):
            v_vec[i] = (u_vec[i] - u_vec[i-1]) / self.dx
        return v_vec

class Upwind2nd(Reconstructer):

    def __init__(self, x_vec, dx):
        self.x_vec = x_vec
        self.dx    = dx

    def du_dx(self, u_vec):
        v_vec = np.zeros(len(u_vec))
        for i in range(len(self.x_vec)):
            v_vec[i] = (3*u_vec[i] - 4*u_vec[i-1] + u_vec[i-2]) / (2 * self.dx)
        return v_vec

class Upwind3rd(Reconstructer):

    def __init__(self, x_vec, dx):
        self.x_vec = x_vec
        self.dx    = dx

    def du_dx(self, u_vec):
        v_vec = np.zeros(len(u_vec))
        for i in range(len(self.x_vec)):
            v_vec[i-1] = (2*u_vec[i] + 3*u_vec[i-1] - 6*u_vec[i-2] + u_vec[i-3]) / (6 * self.dx)
        return v_vec

class vanAlbada(Reconstructer):

    def __init__(self, x_vec, dx):
        self.x_vec = x_vec
        self.dx    = dx

    def du_dx(self, u_vec):
        r_vec = np.zeros(len(u_vec))
        for i in range(len(self.x_vec)):
            r_vec[i-1] = self.interpolation(u_vec[i-2], u_vec[i-1], u_vec[i])
        v_vec = np.zeros(len(u_vec))
        for i in range(len(r_vec)):
            v_vec[i] = (r_vec[i] - r_vec[i-1]) / self.dx
        return v_vec
        
    def interpolation(self, u_l, u_m, u_r):
        if ((u_r - u_m)*(u_m - u_l) > 0):
            r = (u_r - u_m) / (u_m - u_l)
            slope = (u_r - u_l) / 2
            return u_m + 0.5 * self.limitFunc(r) * slope
        return u_m
    
    def limitFunc(self, r):
        return (r**2 + r) / (r**2 + 1)

class WENO_JS3(Reconstructer):

    def __init__(self, x_vec, dx):
        self.x_vec = x_vec
        self.dx    = dx

    def du_dx(self, u_vec):
        r_vec = np.zeros(len(u_vec))
        for i in range(len(self.x_vec)):
            r_vec[i-1] = self.interpolation(u_vec[i-2], u_vec[i-1], u_vec[i])
        v_vec = np.zeros(len(u_vec))
        for i in range(len(r_vec)):
            v_vec[i] = (r_vec[i] - r_vec[i-1]) / self.dx
        return v_vec

    def interpolation(self, u_l, u_m, u_r):
        is_0 = (u_l - u_m) ** 2
        is_1 = (u_r - u_m) ** 2
        smooth_0 = 1 / ( 3 * (1e-20 + is_0) ** 2)
        smooth_1 = 2 / ( 3 * (1e-20 + is_1) ** 2)
        w_0 = smooth_0 / (smooth_0 + smooth_1)
        w_1 = smooth_1 / (smooth_0 + smooth_1)
        up2 = 1.5 * u_m - 0.5 * u_l
        cd2 = 0.5 * u_m + 0.5 * u_r
        return w_0 * up2 + w_1 * cd2

class WENO_Z3(Reconstructer):

    def __init__(self, x_vec, dx):
        self.x_vec = x_vec
        self.dx    = dx

    def du_dx(self, u_vec):
        r_vec = np.zeros(len(u_vec))
        for i in range(len(self.x_vec)):
            r_vec[i-1] = self.interpolation(u_vec[i-2], u_vec[i-1], u_vec[i])
        v_vec = np.zeros(len(u_vec))
        for i in range(len(r_vec)):
            v_vec[i] = (r_vec[i] - r_vec[i-1]) / self.dx
        return v_vec

    def interpolation(self, u_l, u_m, u_r):
        is_0 = (u_l - u_m) ** 2
        is_1 = (u_r - u_m) ** 2
        smooth_0 = (1 + abs(is_0 - is_1) / (is_0 + 1e-40)) / 3
        smooth_1 = (1 + abs(is_0 - is_1) / (is_1 + 1e-40)) / 3 * 2
        w_0 = smooth_0 / (smooth_0 + smooth_1)
        w_1 = smooth_1 / (smooth_0 + smooth_1)
        up2 = 1.5 * u_m - 0.5 * u_l
        cd2 = 0.5 * u_m + 0.5 * u_r
        return w_0 * up2 + w_1 * cd2

class ROUNDL(Reconstructer):

    def __init__(self, x_vec, dx):
        self.x_vec = x_vec
        self.dx    = dx

    def du_dx(self, u_vec):
        r_vec = np.zeros(len(u_vec))
        for i in range(len(self.x_vec)):
            r_vec[i-1] = self.interpolation(u_vec[i-2], u_vec[i-1], u_vec[i])
        v_vec = np.zeros(len(u_vec))
        for i in range(len(r_vec)):
            v_vec[i] = (r_vec[i] - r_vec[i-1]) / self.dx
        return v_vec
        
    def interpolation(self, u_l, u_m, u_r):
        a  = (u_m - u_l) / (u_r - u_l)
        a8 = (((1.0+5.0*((-1.0+a)**2))**2)**2)**2
        a4 = ((1.0+12.0*(a**2))**2)**2
        maxal = max(18000.0*(((0.97-a)**2)**2)*(0.97-a)*((-0.55+a)**2)*(-0.55+a),0.0);
        maxbl = max(1100.0*((0.47-a)**2)*(0.47-a)*((-0.05+a)**2)*(-0.05+a),0.0);
        maxsum= 1.0/3.0+5.0*a/6.0+maxal+maxbl;
        r = (0.5+0.5*a)/a8+(1.0-1.0/a8)*(3.0*a/2.0/a4+(1.0-1.0/a4)*maxsum)
        return r * (u_r - u_l) + u_l
    
class Variational3(Reconstructer):

    def __init__(self, x_vec, dx):
        self.x_vec = x_vec
        self.dx    = dx
        self._cells = list()
        self._vr = VrP2Approach()
        self._vr.set_w_array([1, 0.44, 0.18])
        self._vr.set_boundary_condition("periodic")
        for i in range(len(x_vec)):
            head = x_vec[i] - 0.5 * dx
            tail = x_vec[i] + 0.5 * dx
            self._cells.append(VrP2Cell(head, tail, 1))
        self._vr.initialize_mats(self._cells)

    def du_dx(self, u_vec):
        for i in range(len(self.x_vec)):
            self._cells[i].set_u_mean(u_vec[i])
        self._vr.reconstruct_periodic(self._cells)
        v_vec = np.zeros(len(u_vec))
        for i in range(len(self.x_vec)):
            head = self.x_vec[i] - 0.5 * self.dx
            tail = self.x_vec[i] + 0.5 * self.dx
            v_vec[i] = (self._cells[i].u_on_right() - self._cells[i-1].u_on_right()) / self.dx
        return v_vec

if __name__ == '__main__':
    
    # Set Mesh:
    x_min = 0.0
    x_max = 2*np.pi
    x_num = 201
    k_num = 49
    dk    = 2
    dx    = (x_max - x_min) / x_num
    x_vec = np.linspace(start = x_min + dx/2, stop = x_max - dx/2, num = x_num)

    alpha = np.linspace(start=0, stop =k_num*dk*dx, num=k_num)
    Real = np.zeros(k_num)
    Imag = np.zeros(k_num)

    from displayer import Transient
    RealPart = Transient()
    ImagPart = Transient()
    RealPart.add_plot(x_vec = alpha, u_vec = alpha, label = "Exact")
    ImagPart.add_plot(x_vec = alpha, u_vec = np.zeros(k_num), label = "Exact")

    reconstrctors = dict()
    reconstrctors["vanAlbada"] = vanAlbada(x_vec, dx)
    reconstrctors["WENO_Z3"] = WENO_Z3(x_vec, dx)
    reconstrctors["ROUNDL"] = ROUNDL(x_vec, dx)
    reconstrctors["Variational3"] = Variational3(x_vec, dx)

    data = dict()
    data["alpha"] = alpha
    for name, reconstructor in reconstrctors.items():
        for k in range(k_num):
            uR_vec, uI_vec = function(x_vec, k*dk)
            vR_vec, vI_vec = reconstructor.sepctral_du_dx_(uR_vec, uI_vec)
            temp = complex(0, 0)
            for i in range(x_num):
                temp += complex(vR_vec[i], vI_vec[i]) / complex(uR_vec[i], uI_vec[i])
            temp *= dx/x_num * complex(0, -1)
            Real[k] = temp.real
            Imag[k] = temp.imag
        RealPart.add_plot(x_vec = alpha, u_vec = Real, label = name)
        ImagPart.add_plot(x_vec = alpha, u_vec = Imag, label = name)
        data[name + "_Real"] = np.array(Real)
        data[name + "_Imag"] = np.array(Imag)

    test=pd.DataFrame(data)
    test.to_csv("result/spectral.csv") # 如果生成excel，可以用to_excel
    
    RealPart.display()
    ImagPart.display()