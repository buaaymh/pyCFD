import abc
import numpy as np
import math
from matplotlib import pyplot as plt

from fv_cell import LinearCell
from fv_cell import VrP1Cell
from fv_cell import VrP2Cell
from fv_cell import VrP3Cell

from vr_approach import VrP1Approach
from vr_approach import VrP2Approach
from vr_approach import VrP3Approach

class Piecewise(abc.ABC):

  def set_mesh(self, x_min, x_max, x_num):
    self.x_min = x_min
    self.x_max = x_max
    self.x_num = x_num
    self.dx = (x_max - x_min) / x_num
    self.x_vec = np.linspace(start=x_min + self.dx/2,
                             stop =x_max - self.dx/2, num=x_num)

  def set_initial_condition(self, func):
    self.initial_func = func

  def set_boundary_condition(self, boundary="periodic"):
    self.boundary = boundary

  def set_limiter(self, limiter):
    self._limiter = limiter

  def set_indicator(self, indicator):
    self._limiter.set_indicator(indicator)

  def reconstruct(self):
    if (self.boundary == "periodic"):
      self._reconstruct_periodic()
    elif (self.boundary == "free"):
      self._reconstruct_free()
    elif (self.boundary == "reflect"):
      self._reconstruct_reflect()

  def limit_result(self):
    if (self.boundary == "periodic"):
      self._limiter.limit_periodic(self._cells)
    elif (self.boundary == "free"):
      self._limit_free()
    elif (self.boundary == "reflect"):
      self._limit_reflect()

  @abc.abstractmethod
  def initialize_cells(self):
    pass

  def display(self, x_num_per_cell, overlapped):
    self.fig, self.axis = plt.subplots()
    self.axis.set_xlabel("x")
    self.axis.set_ylabel("U")
    self.axis.grid(True)
    for i in range(self.x_num):
      if overlapped:
        start = self._cells[i].x_left() - self.dx
        stop  = self._cells[i].x_right() + self.dx
        x_vec = np.linspace(start=start, stop=stop, endpoint=True, num=x_num_per_cell*3)
      else:
        start = self._cells[i].x_left()
        stop  = self._cells[i].x_right()
        x_vec = np.linspace(start=start, stop=stop, endpoint=True, num=x_num_per_cell)
      u_vec = self._cells[i].get_u(x_vec)
      self.axis.plot(x_vec, np.transpose(u_vec))
    plt.show()

  def trouble_histograms(self):
    self.fig, self.axis = plt.subplots()
    self.axis.set_xlabel("x")
    self.axis.set_ylabel("I")
    trouble_vec = self._limiter.get_trouble_vec()
    plt.stem(self.x_vec, trouble_vec, linefmt='Black', markerfmt='k.', basefmt='Black')
    plt.show()


class LinearPiecewise(Piecewise):

  def __init__(self):
    pass

  def initialize_cells(self):
    self._cells = list()
    for i in range(self.x_num):
      head = self.x_min + i * self.dx
      tail = head + self.dx
      self._cells.append(LinearCell(head, tail, 1))
      u = self.initial_func(self.x_vec[i])
      self._cells[i].set_u_mean(u)

  def _reconstruct_periodic(self):
    for i in range(self.x_num):
      coef = (self._cells[i].get_u_mean() - self._cells[i-2].get_u_mean()) / 2
      self._cells[i-1].set_coef(coef)

class VrPiecewise(Piecewise):

  def __init__(self, degree):
    self._degree = degree
    self._switch_cell = {1 : VrP1Cell, 2 : VrP2Cell, 3 : VrP3Cell}
    self._switch_appr = {1 : VrP1Approach, 2 : VrP2Approach, 3 : VrP3Approach}
    Approach = self._switch_appr.get(self._degree)
    self._vr = Approach()

  def set_w_array(self, w_array):
    self._vr.set_w_array(w_array)

  def initialize_cells(self):
    self._cells = list()
    CellType = self._switch_cell.get(self._degree)
    for i in range(self.x_num):
      head = self.x_min + i * self.dx
      tail = head + self.dx
      self._cells.append(CellType(head, tail, 1))
      u = self.initial_func(self.x_vec[i])
      self._cells[i].set_u_mean(u)
    self._vr.set_boundary_condition(self.boundary)
    self._vr.initialize_mats(self._cells)

  def reconstruct(self):
    self._vr.reconstruct(self._cells)

  def _reconstruct_periodic(self):
    for i in range(self.x_num):
      coef = (self._cells[i].get_u_mean() - self._cells[i-2].get_u_mean()) / 2
      self._cells[i-1].set_coef(coef)

if __name__ == '__main__':
  # Set Solver
  # reconstruction = LinearPiecewise()
  reconstruction = VrPiecewise(3)
  # Set Mesh:
  x_min = 0.0
  x_max = 1.0
  x_num = 400
  reconstruction.set_mesh(x_min = x_min, x_max = x_max, x_num = x_num)

  # Set Initial Condition:
  def initial(x):
    return np.sin(8 * x * np.pi) * 1.5 + 2

  def initial(x):
    return np.sign(x-0.5) + 1.5

  def initial(x):
    if x > 0.2 and x < 0.3:
      return 4 - np.abs(x-0.25) * 60
    elif x > 0.6 and x < 0.9:
      return 4
    else:
      return 1

  def initial(x):
    if x < 0.5:
      return 1000
    else:
      return 0.01

  reconstruction.set_initial_condition(func=lambda x: initial(x))
  reconstruction.set_boundary_condition(boundary="periodic")
  reconstruction.initialize_cells()
  reconstruction.reconstruct()

  from limiter import BJLimiter
  from limiter import NewLimiter
  limiter = BJLimiter(10)
  # limiter = NewLimiter(1)
  reconstruction.set_limiter(limiter)

  from limiter import EdgeIndicator
  from limiter import RenIndicator
  indicator = EdgeIndicator(0.001)
  # indicator = RenIndicator(1)
  reconstruction.set_indicator(indicator)
  reconstruction.limit_result()

  reconstruction.display(x_num_per_cell=20, overlapped=False)
  reconstruction.trouble_histograms()