import abc
import math
import numpy as np
from scipy.integrate import fixed_quad as quad

class PiecewiseCell(abc.ABC):
  
  @abc.abstractmethod
  def u_on_left(self):
    pass
  @abc.abstractmethod
  def u_on_right(self):
    pass

  def num_equa(self):
    return self._nequa

  def length(self):
    return self._dx

  def set_u_mean(self, u_mean):
    self._u_mean = u_mean
  
  def get_u_mean(self):
    return self._u_mean

  def u_on_left(self):
    return self.get_u(self._head)
  
  def u_on_right(self):
    return self.get_u(self._tail)
  
  def num_coef(self):
    return self._degree

  def get_coef(self):
    return self._coef

  def p0(self):
    return self._p[0]
  
  def set_coef(self, new_coef):
    self._coef = new_coef
  
  def x_left(self):
    return self._head
  
  def x_right(self):
    return self._tail

  # basis func
  def f0_0(self, x):
    return (x - self._x_c) / self._dx
  def f0_1(self, x):
    return 1 / self._dx  
  def f1_0(self, x):
    return self.f0_0(x) ** 2 - self._const_1
  def f1_1(self, x):
    return 2 * self.f0_0(x) / self._dx
  def f1_2(self, x):
    return 2 / (self._dx ** 2)   
  def f2_0(self, x):
    return self.f0_0(x) ** 3 - self._const_2
  def f2_1(self, x):
    return 3 * self.f0_0(x) ** 2 / self._dx
  def f2_2(self, x):
    return 6 * self.f0_0(x) / (self._dx ** 2)
  def f2_3(self, x):
    return 6 / (self._dx ** 3)

  def get_u(self, x):
    u = self._u_mean + np.dot(np.transpose(self.f_p0_vec(x)), self._coef)
    return np.transpose(u)

class LinearCell(PiecewiseCell):

  def __init__(self, head, tail, nequa):
    self._head, self._tail = head, tail
    self._x_c = (self._head + self._tail) / 2
    self._dx = self._tail - self._head
    self._nequa = nequa
    self._degree = 1
    self._coef = np.zeros(self._degree)

  def f_p0_vec(self, x):
    return np.array([self.f0_0(x)])

class VrP1Cell(PiecewiseCell):

  def __init__(self, head, tail, nequa):
    self._head, self._tail = head, tail
    self._x_c = (self._head + self._tail) / 2
    self._dx = self._tail - self._head
    self._nequa = nequa
    self._degree = 1
    self._coef = np.zeros((self._degree, self._nequa))

  def f_p0_vec(self, x):
    return np.array([self.f0_0(x)])
  def f_p1_vec(self, x):
    return np.array([self.f0_1(x)])

  def get_A_mat_inner(self):
    A_mat = self._get_A_mat(self._head, 2) + self._get_A_mat(self._tail, 2)
    return A_mat

  def get_A_mat_end(self, orient):
    if orient < 0:
      return self._get_A_mat(self._head, 1)*2 + self._get_A_mat(self._tail, self._degree+1)
    else:
      return self._get_A_mat(self._head, self._degree+1) + self._get_A_mat(self._tail, 1)*2
    
  def _get_A_mat(self, x, p):
    f = [self.f_p0_vec(x), self.f_p1_vec(x)]
    A_mat = np.zeros((self._degree, self._degree))
    for k in range(p):
      A_mat += f[k] * f[k] * self._p[k]
    return A_mat

  def get_B_mat_inner(self, cell, x_l, x_r):
    B_mat = 0
    fi = [cell.f_p0_vec(x_r), cell.f_p1_vec(x_r)]
    fj = [self.f_p0_vec(x_l), self.f_p1_vec(x_l)]
    for k in range(self._degree+1):
      B_mat += fi[k] * fj[k] * self._p[k]
    return B_mat

  def get_B_mat_end(self, cell, x_l, x_r):
    i0, j0 = cell.f_p0_vec(x_r), self.f_p0_vec(x_l)
    B_mat = i0 * j0 * self._p[0]
    return B_mat

class VrP2Cell(PiecewiseCell):

  def __init__(self, head, tail, nequa):
    self._head, self._tail = head, tail
    self._x_c = (self._head + self._tail) / 2
    self._dx = self._tail - self._head
    self._nequa = nequa
    self._degree = 2
    self._coef = np.zeros((self._degree, self._nequa))
    f1 = lambda x : self.f0_0(x) ** 2
    self._const_1 = quad(f1, self._head, self._tail, n=3)[0] / self._dx

  def f_p0_vec(self, x):
    return np.array([self.f0_0(x), self.f1_0(x)])
  def f_p1_vec(self, x):
    return np.array([self.f0_1(x), self.f1_1(x)])
  def f_p2_vec(self, x):
    return np.array([0           , self.f1_2(x)])

  def get_A_mat_inner(self):
    A_mat = self._get_A_mat(self._head, 3) + self._get_A_mat(self._tail, 3)
    return A_mat

  def get_A_mat_end(self, orient):
    A_mat = np.zeros((self._degree, self._degree))
    if orient < 0:
      A_mat = self._get_A_mat(self._head, 1)*2 + self._get_A_mat(self._tail, self._degree+1)
    else:
      A_mat = self._get_A_mat(self._head, self._degree+1) + self._get_A_mat(self._tail, 1)*2
    return A_mat
    
  def _get_A_mat(self, x, p):
    A_mat = np.zeros((self._degree, self._degree))
    f = [self.f_p0_vec(x), self.f_p1_vec(x), self.f_p2_vec(x)]
    for i in range(self._degree):
      for j in range(self._degree):
        for k in range(p):
          A_mat[i,j] += f[k][i] * f[k][j] * self._p[k]
    return A_mat

  def get_B_mat_inner(self, cell, x_l, x_r):
    B_mat = np.zeros((self._degree, self._degree))
    fi = [cell.f_p0_vec(x_r), cell.f_p1_vec(x_r), cell.f_p2_vec(x_r)]
    fj = [self.f_p0_vec(x_l), self.f_p1_vec(x_l), self.f_p2_vec(x_l)]
    for i in range(self._degree):
      for j in range(self._degree):
        for k in range(self._degree+1):
          B_mat[i,j] += fi[k][i] * fj[k][j] * self._p[k]
    return B_mat

  def get_B_mat_end(self, cell, x_l, x_r):
    B_mat = np.zeros((self._degree, self._degree))
    i0, j0 = cell.f_p0_vec(x_r), self.f_p0_vec(x_l)
    for i in range(self._degree):
      for j in range(self._degree):
        B_mat[i,j] += i0[i] * j0[j] * self._p[0]
    return B_mat

class VrP3Cell(PiecewiseCell):

  def __init__(self, head, tail, nequa):
    self._head, self._tail = head, tail
    self._x_c = (self._head + self._tail) / 2
    self._dx = self._tail - self._head
    self._nequa = nequa
    self._degree = 3
    self._coef = np.zeros((self._degree, self._nequa))
    f1 = lambda x : self.f0_0(x) ** 2
    self._const_1 = quad(f1, self._head, self._tail, n=3)[0] / self._dx
    f2 = lambda x : self.f0_0(x) ** 3
    self._const_2 = quad(f2, self._head, self._tail, n=3)[0] / self._dx

  def f_p0_vec(self, x):
      return np.array([self.f0_0(x), self.f1_0(x), self.f2_0(x)])
  def f_p1_vec(self, x):
      return np.array([self.f0_1(x), self.f1_1(x), self.f2_1(x)])
  def f_p2_vec(self, x):
      return np.array([0           , self.f1_2(x), self.f2_2(x)])
  def f_p3_vec(self, x):
      return np.array([0           ,            0, self.f2_3(x)])
  
  def get_A_mat_inner(self):
    A_mat = self._get_A_mat(self._head, 4) + self._get_A_mat(self._tail, 4)
    return A_mat

  def get_A_mat_end(self, orient):
    A_mat = np.zeros((3, 3))
    if orient < 0:
      A_mat = self._get_A_mat(self._head, 1) + self._get_A_mat(self._tail, 4)
    else:
      A_mat = self._get_A_mat(self._head, 4) + self._get_A_mat(self._tail, 1)
    return A_mat
    
  def _get_A_mat(self, x, p):
    A_mat = np.zeros((self._degree, self._degree))
    f = [self.f_p0_vec(x), self.f_p1_vec(x), self.f_p2_vec(x), self.f_p3_vec(x)]
    for i in range(self._degree):
      for j in range(self._degree):
        for k in range(p):
          A_mat[i,j] += f[k][i] * f[k][j] * self._p[k]
    return A_mat
  
  def get_B_mat_inner(self, cell, x_l, x_r):
    B_mat = np.zeros((self._degree, self._degree))
    fi = [cell.f_p0_vec(x_r), cell.f_p1_vec(x_r), cell.f_p2_vec(x_r), cell.f_p3_vec(x_r)]
    fj = [self.f_p0_vec(x_l), self.f_p1_vec(x_l), self.f_p2_vec(x_l), self.f_p3_vec(x_l)]
    for i in range(self._degree):
      for j in range(self._degree):
        for k in range(self._degree+1):
          B_mat[i,j] += fi[k][i] * fj[k][j] * self._p[k]
    return B_mat

  def get_B_mat_end(self, cell, x_l, x_r):
    B_mat = np.zeros((self._degree, self._degree))
    i0, j0 = cell.f_p0_vec(x_r), self.f_p0_vec(x_l)
    for i in range(self._degree):
      for j in range(self._degree):
        B_mat[i,j] += i0[i] * j0[j] * self._p[0]
    return B_mat

if __name__ == '__main__':
  print("P2 Cell:")
  cell = VrP2Cell(-1, 1, 1)
  cell.set_u_mean(2)
  cell.set_coef([1, 2])
  print("f: ", cell.f_p0_vec(-1))
  print("c: ", cell.get_coef())
  print("u: ", cell.get_u(-1))

  print("P3 Cell:")
  cell = VrP3Cell(-1, 1, 3)
  cell.set_u_mean([2, 4, 6])
  cell.set_coef([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
  print("f: ", cell.f_p0_vec(-1))
  print("c: ", cell.get_coef())
  print("u: ", cell.get_u(-1))
  integrate = quad(cell.get_u, -1, 1, n=5)
  print(integrate)