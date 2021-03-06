import abc
import numpy as np
import math
from time import*

from fdm import NumericalSolver
import riemann
from fv_cell import VrP1Cell, VrP2Cell, VrP3Cell
from vr_approach import VrP1Approach, VrP2Approach, VrP3Approach
from time_scheme import RK3ForVr
from equation import Euler1d

class VrSolver(NumericalSolver):

  def __init__(self, degree):
    self._time_scheme = RK3ForVr()
    self._degree = degree
    self._switch_cell = {1 : VrP1Cell, 2 : VrP2Cell, 3 : VrP3Cell}
    self._switch_appr = {1 : VrP1Approach, 2 : VrP2Approach, 3 : VrP3Approach}
    Approach = self._switch_appr.get(self._degree)
    self._vr = Approach()
    self._euler = Euler1d(gamma=1.4)

  def set_limiter(self, limiter):
    self._trouble_history = list()
    self._limiter = limiter

  def get_trouble_history(self):
    return self._trouble_history

  def set_indicator(self, indicator):
    self._limiter.set_indicator(indicator)

  def _build_cells(self):
    self._vr.set_boundary_condition(self.boundary)
    self.cells = list()
    CellType = self._switch_cell.get(self._degree)
    for i in range(self.x_num):
      head = self.x_min + i * self.dx
      tail = head + self.dx
      self.cells.append(CellType(head, tail, self.riemann.num_equations()))
    self._vr.initialize_mats(self.cells)

  def run_with_transient(self):
    self._build_cells()
    self.u_mat = np.zeros((self.x_num, self.riemann.num_equations()))
    for i in range(self.x_num):
      self.u_mat[i] = self.initial_func(self.x_vec[i])
      self.cells[i].set_u_mean(self.u_mat[i])
    # time march
    self._time_scheme.set_rhs(rhs=lambda cells: self.eval_rhs(cells))
    for i in range(1, self.t_num+1):
      if (i%50 == 0): print('run : {:.2%}'.format(i / self.t_num))
      self._time_scheme.get_cells_new(self.cells, self.dt)
      for j in range(self.x_num):
        self.u_mat[j] = self.cells[j].get_u_mean()

  def run_with_animation(self):
    self._build_cells()
    self.u_mat = np.zeros((self.t_num+1, self.x_num, self.riemann.num_equations()))
    for i in range(self.x_num):
      self.u_mat[0,i] = self.initial_func(self.x_vec[i])
      self.cells[i].set_u_mean(self.u_mat[0,i])
    # time march
    self._time_scheme.set_rhs(rhs=lambda cells: self.eval_rhs(cells))
    for i in range(1, self.t_num+1):
      if (i%50 == 0): print('run : {:.2%}'.format(i / self.t_num))
      self._time_scheme.get_cells_new(self.cells, self.dt)
      for j in range(self.x_num): self.u_mat[i,j] = self.cells[j].get_u_mean()

  def eval_rhs(self, cells):
    rhs = np.zeros((self.x_num, self.riemann.num_equations()))
    if (self.boundary == "periodic"): self._dual_periodic(cells, rhs)
    elif (self.boundary == "free"): self._dual_free(cells, rhs)
    elif (self.boundary == "reflect"): self._dual_reflect(cells, rhs)
    return rhs
  
  def _dual_periodic(self, cells, rhs):
    flux = np.zeros((self.x_num, cells[0].num_equa()))
    self._vr.reconstruct_periodic(self.cells)
    self._limiter.limit_periodic(self.cells)
    self._trouble_history.append(self._limiter.get_trouble_history_vec())
    for i in range(self.x_num):
      flux[i-1] = self.riemann.eval_flux(cells[i-2].u_on_right() ,cells[i-1].u_on_left())
    for i in range(self.x_num): rhs[i-1] = flux[i-1] - flux[i]
    rhs /= self.dx

  def _dual_free(self, cells, rhs):
    flux = np.zeros((self.x_num+1, cells[0].num_equa()))
    self._vr.reconstruct_free(self.cells)
    self._limiter.limit_free_reflect(self.cells)
    self._trouble_history.append(self._limiter.get_trouble_history_vec())
    flux[0] = self.riemann.flux(cells[0].u_on_left())
    flux[-1] = self.riemann.flux(cells[-1].u_on_right())
    for i in range(1, self.x_num):
      flux[i] = self.riemann.eval_flux(cells[i-1].u_on_right(), cells[i].u_on_left())
    for i in range(self.x_num): rhs[i] = flux[i] - flux[i+1]
    rhs /= self.dx

  def _dual_reflect(self, cells, rhs):
    flux = np.zeros((self.x_num+1, self.riemann.num_equations()))
    self._vr.reconstruct_reflect(self.cells)
    self._limiter.limit_free_reflect(self.cells)
    self._trouble_history.append(self._limiter.get_trouble_history_vec())
    flux[0] = self.riemann.eval_reflected_flux(cells[0].u_on_left(), -1)
    flux[-1] = self.riemann.eval_reflected_flux(cells[-1].u_on_right(), 1)
    for i in range(1, self.x_num):
      flux[i] = self.riemann.eval_flux(cells[i-1].u_on_right(), cells[i].u_on_left())
    for i in range(self.x_num): rhs[i] = flux[i] - flux[i+1]
    rhs /= self.dx
  
if __name__ == '__main__':
  # Set Initial Condition:
  from equation import Euler1d
  euler = Euler1d(gamma=1.4)

  def initial(x):
    if x < 1:
        return euler.u_p_rho_to_U(u=2.629369, p=10.333333, rho=3.857143)
    else:
        return euler.u_p_rho_to_U(u=0, p=1, rho=1+0.2*np.sin(5*x))

  solver = VrSolver(3)
  
  # Set Mesh:
  x_min = 0.0
  x_max = 10.0
  x_num = 200
  solver.set_mesh(x_min = x_min, x_max = x_max, x_num = x_num)
  solver.set_initial_condition(func=lambda x: initial(x))
  solver.set_boundary_condition(boundary="free")

  # Set Time Scheme:
  start = 0.0
  stop = 1.8
  t_num = 200
  solver.set_time_stepper(start = start, stop = stop, t_num = t_num)
  # Set Riemann Problem:
  # from riemann import EulerAusm
  # riemann = EulerAusm()
  from riemann import EulerVanLeer
  riemann = EulerVanLeer()
  solver.set_riemann_solver(riemann = riemann)

  # Set Limiter
  from limiter import EulerBJLimiter
  from limiter import EulerNewLimiter
  # limiter = EulerBJLimiter(True)
  limiter = EulerNewLimiter(2, True)
  solver.set_limiter(limiter)

  from limiter import EdgeIndicator
  from limiter import RenIndicator
  # indicator = EdgeIndicator(1)
  indicator = RenIndicator(1)
  solver.set_indicator(indicator)

  x_vec = solver.get_x_vec()
  t_vec = solver.get_t_vec()
  begin = time()

  from displayer import Transient
  solver.run_with_transient()
  end = time()
  print('Total time : %.2f s' %(end-begin))
  U_mat = solver.get_u_mat()
  u_vec = np.zeros(len(U_mat))
  p_vec = np.zeros(len(U_mat))
  rho_vec = np.zeros(len(U_mat))
  for i in range(len(U_mat)):
    u, p, rho = euler.U_to_u_p_rho(U_mat[i])
    u_vec[i], p_vec[i], rho_vec[i] = u, p, rho
  transient = Transient()
  transient.add_plot(x_vec = x_vec, u_vec = rho_vec, type = "k-.", label = r'$\rho(x)$')
  transient.display()
  # import data
  # filename = "result/Blast/p3_400_new2_ren.csv"
  # data.write_rows(filename, [x_vec, rho_vec])

  # from displayer import Animation
  # solver.run_with_animation()
  # U_mat = solver.get_u_mat()
  # u_vec = np.zeros((len(U_mat), len(U_mat[0])))
  # p_vec = np.zeros((len(U_mat), len(U_mat[0])))
  # rho_vec = np.zeros((len(U_mat), len(U_mat[0])))
  # for i in range(len(U_mat)):
  #     for j in range(len(U_mat[0])):
  #         u, p, rho = euler.U_to_u_p_rho(U_mat[i, j])
  #         u_vec[i,j], p_vec[i,j], rho_vec[i,j] = u, p, rho

  # animation = Animation(x_vec=x_vec, t_vec=t_vec, u_mat=rho_vec, u_label = r'$\rho(x)$')
  # animation.display(type = "k:", y_min = 0, y_max = 7.2)

  from displayer import Contour
  trouble_mat = solver.get_trouble_history()
  t_vec = np.linspace(start=start, stop=stop, endpoint=True, num=t_num*3)
  contour = Contour(x_vec=x_vec, t_vec=t_vec, u_mat=trouble_mat)
  contour.display()
