import abc
import numpy as np
from scipy.integrate import fixed_quad as quad

from equation import Euler1d
import fv_cell
import gas

class Limiter(abc.ABC):

  def get_trouble_vec(self):
    return self._trouble

  def get_trouble_history_vec(self):
    history = np.zeros(len(self._trouble))
    for i in range(len(self._trouble)):
      if (self._trouble[i] > self._indicator.Ck()):
        history[i] = 1
    return history

  def set_indicator(self, indicator):
    self._indicator = indicator

  def limit_periodic(self, cells):
    self._dx, self._p = cells[0].length(), cells[0].num_coef()
    self._trouble = np.zeros(len(cells))
    for i in range(len(cells)):
      self._trouble[i-1] = self._indicator.detect(cells[i-2], cells[i-1], cells[i])
    for i in range(len(cells)):
      if (self._trouble[i-1] > self._indicator.Ck()):
        self._limit(cells[i-2], cells[i-1], cells[i])

  def limit_free_reflect(self, cells):
    self._dx, self._p = cells[0].length(), cells[0].num_coef()
    self._trouble = np.zeros(len(cells))
    for i in range(len(cells)):
      if i == 0:
        self._trouble[i] = self._indicator.detect_end_l(cells[i], cells[i+1])
      elif i == len(cells)-1:
        self._trouble[i] = self._indicator.detect_end_r(cells[i-1], cells[i])
      else:
        self._trouble[i] = self._indicator.detect(cells[i-1], cells[i], cells[i+1])
    for i in range(len(cells)):
      if (self._trouble[i] > self._indicator.Ck()):
        if i == 0:
          self._limit_end_l(cells[i], cells[i+1])
        elif i == len(cells)-1:
          self._limit_end_r(cells[i-1], cells[i])
        else:
          self._limit(cells[i-1], cells[i], cells[i+1])
    if cells[0].num_equa() > 1:
      self._positive_preserve(cells)

  def _positive_preserve(self, cells):
    for i in range(len(cells)):
      mean = cells[i].get_u_mean()
      U_l = cells[i].u_on_left()
      U_r = cells[i].u_on_right()
      U_m = cells[i].get_u((cells[i].x_left()+cells[i].x_right())/2)
      rho_min = min(U_l[0], U_r[0], U_m[0])
      E_min = min(U_l[2], U_r[2], U_m[2])
      if E_min < 0 or rho_min < 0:
        C = cells[i].get_coef()
        cells[i].set_coef(C*0)

  def _get_eigen_vec(self, U):
    L, R = np.zeros((3, 3)), np.ones((3, 3))
    u, p, rho = self._euler.U_to_u_p_rho(U)
    c = self.gas.p_rho_to_a(p, rho)
    b_1 = 0.4 / c**2
    b_2 = b_1 * u**2 * 0.5
    h = (U[2] + p) / rho

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
    return L, R

class BJLimiter(Limiter):

  def __init__(self, M):
    self._M = M

  def _limit(self, cell_l, cell_m, cell_r):
    coef = cell_m.get_coef()
    if cell_m.num_coef() > 1:
      coef[1:] *= 0
      cell_m.set_coef(coef)
    U_l, U_m, U_r = cell_l.get_u_mean(), cell_m.get_u_mean(), cell_r.get_u_mean()
    dU_l, dU_r = U_m-cell_m.u_on_left(), cell_m.u_on_right()-U_m
    dU_f, dU_b = U_m - U_l, U_r - U_m
    dU = self._minmod([dU_l, dU_r, dU_f, dU_b])
    coef[0] = dU * 2
    cell_m.set_coef(coef)

  def _minmod(self, dus):
    mods = np.abs(dus)
    du_min = np.min(mods)
    du_min *= np.sign(dus[0])
    sign = np.sign(dus)
    for i in range(len(dus)):
      if sign[i-1] != sign[i]:
        du_min = 0
    return du_min

class EulerBJLimiter(Limiter):

  def __init__(self, isChara):
    self._isChara = isChara
    self._euler = Euler1d(gamma=1.4)
    self.gas = gas.Ideal(gamma=1.4)

  def _limit(self, cell_l, cell_m, cell_r):
    coef = cell_m.get_coef()
    coef[1:] *= 0
    cell_m.set_coef(coef)
    U_l, U_m, U_r = cell_l.get_u_mean(), cell_m.get_u_mean(), cell_r.get_u_mean()
    dU_l, dU_r = U_m-cell_m.u_on_left(), cell_m.u_on_right()-U_m
    dU_f, dU_b = U_m - U_l, U_r - U_m
    if self._isChara:
      L, R = self._get_eigen_vec(U_m)
      dU_l = np.dot(L, dU_l)
      dU_r = np.dot(L, dU_r)
      dU_f = np.dot(L, dU_f)
      dU_b = np.dot(L, dU_b)
    dU = self._minmod([dU_l, dU_r, dU_f , dU_b])
    if self._isChara:
      dU = np.dot(R, dU)
    coef[0] = dU * 2
    cell_m.set_coef(coef)

  def _limit_end_l(self, cell_m, cell_r):
    coef = cell_m.get_coef()
    coef[1:] *= 0
    cell_m.set_coef(coef)
    U_m, U_r = cell_m.get_u_mean(), cell_r.get_u_mean()
    dU_r = cell_m.u_on_right()-U_m
    dU_b = U_r - U_m
    if self._isChara:
      L, R = self._get_eigen_vec(U_m)
      dU_r = np.dot(L, dU_r)
      dU_b = np.dot(L, dU_b)
    dU = self._minmod([dU_r, dU_b])
    if self._isChara:
      dU = np.dot(R, dU)
    coef[0] = dU * 2
    cell_m.set_coef(coef)

  def _limit_end_r(self, cell_l, cell_m):
    coef = cell_m.get_coef()
    coef[1:] *= 0
    cell_m.set_coef(coef)
    U_l, U_m = cell_l.get_u_mean(), cell_m.get_u_mean()
    dU_l = U_m-cell_m.u_on_left()
    dU_f = U_m - U_l
    if self._isChara:
      L, R = self._get_eigen_vec(U_m)
      dU_l = np.dot(L, dU_l)
      dU_f = np.dot(L, dU_f)
    dU = self._minmod([dU_l, dU_f])
    if self._isChara:
      dU = np.dot(R, dU)
    coef[0] = dU * 2
    cell_m.set_coef(coef)

  def _minmod(self, dus):
    mods = np.abs(dus)
    du_min = np.amin(mods, axis = 0)
    du_min *= np.sign(dus[0])
    sign = np.sign(dus)
    for j in range(3):
      for i in range(len(dus)):
        if sign[i-1,j] != sign[i,j]:
          du_min[j] = 0
    return du_min

class NewLimiter(Limiter):

  def __init__(self, n):
    self._n = n
    self._euler = Euler1d(gamma=1.4)

  def limit_periodic(self, cells):
    self._dx, self._p = cells[0].length(), cells[0].num_coef()
    self._trouble = np.zeros(len(cells))
    self._factor = np.zeros((len(cells), cells[0].num_equa()))
    for i in range(len(cells)):
      self._trouble[i-1] = self._indicator.detect(cells[i-2], cells[i-1], cells[i])
    for i in range(len(cells)):
      if (self._trouble[i-1] > self._indicator.Ck()):
        self._factor[i-1] = self._limit(cells[i-2], cells[i-1], cells[i])
    for i in range(len(cells)):
      if (self._trouble[i-1] > self._indicator.Ck()):
        C = cells[i-1].get_coef()
        cells[i-1].set_coef(C*self._factor[i-1])

  def _limit(self, cell_l, cell_m, cell_r):
    u_mean = cell_m.get_u_mean()
    s_l = self._S(cell_l.u_on_right()-u_mean, cell_m.u_on_left()-u_mean)
    s_r = self._S(cell_m.u_on_right()-u_mean, cell_r.u_on_left()-u_mean)
    return min(s_l, s_r)

  def _S(self, a, b):
    if a * b <= 0:
      return 0
    r = (a / b) ** self._n
    s = 2*r / (r**2 + 1)
    return s

class EulerNewLimiter(Limiter):

  def __init__(self, n, isChara):
    self._n = n
    self._isChara = isChara
    self._euler = Euler1d(gamma=1.4)
    self.gas = gas.Ideal(gamma=1.4)

  def limit_periodic(self, cells):
    self._dx, self._p = cells[0].length(), cells[0].num_coef()
    self._trouble = np.zeros(len(cells))
    self._factor = np.zeros((len(cells), cells[0].num_equa(), cells[0].num_equa()))
    for i in range(len(cells)):
      self._trouble[i-1] = self._indicator.detect(cells[i-2], cells[i-1], cells[i])
    for i in range(len(cells)):
      if (self._trouble[i-1] > self._indicator.Ck()):
        self._factor[i-1] = self._limit(cells[i-2], cells[i-1], cells[i])
    for i in range(len(cells)):
      if (self._trouble[i-1] > self._indicator.Ck()):
        C = cells[i-1].get_coef()
        cells[i-1].set_coef(np.transpose(np.dot(self._factor[i-1], np.transpose(C))))
    if cells[0].num_equa() > 1:
      self._positive_preserve(cells)

  def limit_free_reflect(self, cells):
    self._dx, self._p = cells[0].length(), cells[0].num_coef()
    self._trouble = np.zeros(len(cells))
    self._factor = np.zeros((len(cells), cells[0].num_equa(), cells[0].num_equa()))
    for i in range(len(cells)):
      if i == 0:
        self._trouble[i] = self._indicator.detect_end_l(cells[i], cells[i+1])
      elif i == len(cells)-1:
        self._trouble[i] = self._indicator.detect_end_r(cells[i-1], cells[i])
      else:
        self._trouble[i] = self._indicator.detect(cells[i-1], cells[i], cells[i+1])
    for i in range(len(cells)):
      if (self._trouble[i] > self._indicator.Ck()):
        if i == 0:
          self._factor[i] = self._limit_end_l(cells[i], cells[i+1])
        if i == len(cells)-1:
          self._factor[i] = self._limit_end_r(cells[i-1], cells[i])
        else:
          self._factor[i] = self._limit(cells[i-1], cells[i], cells[i+1])
    for i in range(len(cells)):
      if (self._trouble[i] > self._indicator.Ck()):
        C = cells[i].get_coef()
        cells[i].set_coef(np.transpose(np.dot(self._factor[i], np.transpose(C))))
    if cells[0].num_equa() > 1:
      self._positive_preserve(cells)

  def _limit(self, cell_l, cell_m, cell_r):
    factor = np.zeros((3, 3))
    U_m = cell_m.get_u_mean()
    dU_f, dU_b = cell_l.u_on_right()-U_m, cell_r.u_on_left() -U_m
    dU_l, dU_r = cell_m.u_on_left() -U_m, cell_m.u_on_right()-U_m
    if self._isChara:
      L, R = self._get_eigen_vec(U_m)
      dU_f, dU_b = np.dot(L, dU_f), np.dot(L, dU_b)
      dU_l, dU_r = np.dot(L, dU_l), np.dot(L, dU_r)
    s_l = self._S_euler(dU_f, dU_l)
    s_r = self._S_euler(dU_b, dU_r)
    s = np.amin([s_l, s_r], axis = 0)
    for i in range(3):
      factor[i,i] = s[i]
    if self._isChara:
      factor = np.dot(R, np.dot(factor, L))
    return factor

  def _limit_end_l(self, cell_m, cell_r):
    factor = np.zeros((3, 3))
    U_m = cell_m.get_u_mean()
    dU_b = cell_r.u_on_left() -U_m
    dU_r = cell_m.u_on_right()-U_m
    if self._isChara:
      L, R = self._get_eigen_vec(U_m)
      dU_b = np.dot(L, dU_b)
      dU_r = np.dot(L, dU_r)
    s_r = self._S_euler(dU_b, dU_r)
    for i in range(3):
      factor[i,i] = s_r[i]
    if self._isChara:
      factor = np.dot(R, np.dot(factor, L))
    return factor
  
  def _limit_end_r(self, cell_l, cell_m):
    factor = np.zeros((3, 3))
    U_m = cell_m.get_u_mean()
    dU_f = cell_l.u_on_right()-U_m
    dU_l = cell_m.u_on_left() -U_m
    if self._isChara:
      L, R = self._get_eigen_vec(U_m)
      dU_f = np.dot(L, dU_f)
      dU_l = np.dot(L, dU_l)
    s_l = self._S_euler(dU_f, dU_l)
    for i in range(3):
      factor[i,i] = s_l[i]
    if self._isChara:
      factor = np.dot(R, np.dot(factor, L))
    return factor

  def _S_euler(self, a, b):
    s = np.zeros(3)
    for i in range(3):
      if a[i] * b[i] > 0:
        r = (a[i]/b[i]) ** self._n
        s[i] = 2*r / (r**2 + 1)
    return s

class RenIndicator():

  def __init__(self, Ck):
    self._Ck = Ck

  def Ck(self):
    return self._Ck

  def detect(self, cell_l, cell_m, cell_r):
    dx, p = cell_l.length(), cell_l.num_coef()
    hp = dx** ((p+1)/2)
    mean_l = cell_l.get_u_mean()
    mean_m = cell_m.get_u_mean()
    mean_r = cell_r.get_u_mean()
    du_l = cell_l.get_u(cell_l.x_right()+dx/2) - mean_m
    du_r = cell_r.get_u(cell_r.x_left() -dx/2) - mean_m
    if cell_m.num_equa() > 1:
      u_max = max(abs(mean_l[0]), abs(mean_m[0]), abs(mean_r[0]), 1e-6)*hp*2
      indicator = (abs(du_l[0]) + abs(du_r[0])) / u_max
    else:
      u_max = max(abs(mean_l), abs(mean_m), abs(mean_r), 1e-6)*hp*2
      indicator = (abs(du_l) + abs(du_r)) / u_max
    return indicator

  def detect_end_l(self, cell_m, cell_r):
    dx, p = cell_m.length(), cell_m.num_coef()
    hp = dx** ((p+1)/2)
    mean_m = cell_m.get_u_mean()
    mean_r = cell_r.get_u_mean()
    du_r = cell_r.get_u(cell_r.x_left() -dx/2) - mean_m
    if cell_m.num_equa() > 1:
      u_max = max(abs(mean_m[0]), abs(mean_r[0]), 1e-6)*hp*2
      indicator = abs(du_r[0]) / u_max
    else:
      u_max = max(abs(mean_m), abs(mean_r), 1e-6)*hp*2
      indicator = abs(du_r) / u_max
    return indicator

  def detect_end_r(self, cell_l, cell_m):
    dx, p = cell_m.length(), cell_m.num_coef()
    hp = dx** ((p+1)/2)
    mean_m = cell_m.get_u_mean()
    mean_l = cell_l.get_u_mean()
    du_l = cell_l.get_u(cell_l.x_right()+dx/2) - mean_m
    if cell_m.num_equa() > 1:
      u_max = max(abs(mean_m[0]), abs(mean_l[0]), 1e-6)*hp*2
      indicator = abs(du_l[0]) / u_max
    else:
      u_max = max(abs(mean_m), abs(mean_l), 1e-6)*hp*2
      indicator = abs(du_l) / u_max
    return indicator

class EdgeIndicator():

  def __init__(self, Ck):
    self._Ck = Ck

  def Ck(self):
    return self._Ck

  def detect(self, cell_l, cell_m, cell_r):
    dx, p = cell_m.length(), cell_m.num_coef()
    mean_l = cell_l.get_u_mean()
    mean_m = cell_m.get_u_mean()
    mean_r = cell_r.get_u_mean()
    du_l = cell_l.u_on_right() - cell_m.u_on_left()
    du_r = cell_m.u_on_right() - cell_r.u_on_left()
    if cell_m.num_equa() > 1:
      u_max = max(abs(mean_l[0]), abs(mean_m[0]), abs(mean_r[0]), 1e-6)
      indicator = (abs(du_l[0]) + abs(du_r[0])) / u_max
    else:
      u_max = max(abs(mean_l), abs(mean_m), abs(mean_r), 1e-6)
      indicator = (abs(du_l) + abs(du_r)) / u_max
    return indicator

  def detect_end_l(self, cell_m, cell_r):
    dx, p = cell_m.length(), cell_m.num_coef()
    mean_m = cell_m.get_u_mean()
    mean_r = cell_r.get_u_mean()
    du_r = cell_m.u_on_right() - cell_r.u_on_left()
    if cell_m.num_equa() > 1:
      u_max = max(abs(mean_m[0]), abs(mean_r[0]), 1e-6)
      indicator = abs(du_r[0]) / u_max
    else:
      u_max = max(abs(mean_m), abs(mean_r), 1e-6)
      indicator = abs(du_r) / u_max
    return indicator

  def detect_end_r(self, cell_l, cell_m):
    dx, p = cell_m.length(), cell_m.num_coef()
    mean_m = cell_m.get_u_mean()
    mean_l = cell_l.get_u_mean()
    du_l = cell_m.u_on_left() - cell_l.u_on_right()
    if cell_m.num_equa() > 1:
      u_max = max(abs(mean_m[0]), abs(mean_l[0]), 1e-6)
      indicator = abs(du_l[0]) / u_max
    else:
      u_max = max(abs(mean_m), abs(mean_l), 1e-6)
      indicator = abs(du_l) / u_max
    return indicator


