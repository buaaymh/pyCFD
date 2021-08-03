import abc
import numpy as np
import math

import fv_cell

class VrApproach(abc.ABC):

  def set_boundary_condition(self, boundary="periodic"):
    self.boundary = boundary

  def set_w_array(self, w_array):
    self._w = w_array

  def initialize_mats(self, cells):
    self._dx = cells[0].length()
    self._initialize_p_array(self._dx, self._degree)
    self._num_cells = len(cells)
    self._A_inv = np.zeros((self._num_cells, self._degree, self._degree))
    if (self.boundary == "free" or self.boundary == "reflect"):
      self._B_mat = np.zeros((self._num_cells+1, self._degree, self._degree))
      self._B_mat[+0] = self._get_B_mat_end(cells[+0], cells[+0].x_left(), cells[+0], cells[+0].x_left()) * 2
      self._B_mat[-1] = self._get_B_mat_end(cells[-1], cells[-1].x_right(), cells[-1], cells[-1].x_right()) * 2
      self._A_inv[+0] = np.linalg.inv(self._get_A_mat_end(cells[+0], -1))
      self._A_inv[-1] = np.linalg.inv(self._get_A_mat_end(cells[-1], +1))
      for i in range(1, self._num_cells-1):
        self._A_inv[i] = np.linalg.inv(self._get_A_mat_inner(cells[i]))
      for i in range(1, self._num_cells):
        self._B_mat[i] = self._get_B_mat_inner(cells[i-1], cells[i-1].x_right(), cells[i], cells[i].x_left())
    elif (self.boundary == "periodic"):
      self._B_mat = np.zeros((self._num_cells, self._degree, self._degree))
      for i in range(self._num_cells):
        self._A_inv[i] = np.linalg.inv(self._get_A_mat_inner(cells[i]))
        self._B_mat[i] = self._get_B_mat_inner(cells[i-1], cells[i-1].x_right(), cells[i], cells[i].x_left())

  def reconstruct(self, cells):
    if (self.boundary == "periodic"):
      self.reconstruct_periodic(cells)

  def reconstruct_periodic(self, cells):
    b_vec = np.zeros((self._num_cells, self._degree, cells[0].num_equa()))
    for i in range(self._num_cells):
      fron_diff = cells[i-2].get_u_mean() - cells[i-1].get_u_mean()
      back_diff = cells[i].get_u_mean() - cells[i-1].get_u_mean()
      b_vec[i-1] = np.outer(cells[i-1].f_p0_vec(cells[i-1].x_left()), fron_diff) + \
                   np.outer(cells[i-1].f_p0_vec(cells[i-1].x_right()), back_diff)
    b_vec *= self._p[0]
    for i in range(7):
      for j in range(self._num_cells):
        coef_l, coef_m, coef_r = cells[j-2].get_coef(), cells[j-1].get_coef(), cells[j].get_coef()
        coef_m = coef_m * (-0.3) + 1.3 * \
        (np.dot(np.dot(self._A_inv[j-1], np.transpose(self._B_mat[j-1])), coef_l) +
         np.dot(np.dot(self._A_inv[j-1], self._B_mat[j]), coef_r) +
         np.dot(self._A_inv[j-1], b_vec[j-1]))
        cells[j-1].set_coef(coef_m)

  def reconstruct_free(self, cells):
    b_vec = np.zeros((self._num_cells, self._degree, cells[0].num_equa()))
    for i in range(self._num_cells):
      fron_diff, back_diff = np.zeros(cells[i].num_equa()), np.zeros(cells[i].num_equa())
      if i == 0:
        back_diff = cells[i+1].get_u_mean() - cells[i].get_u_mean()
      elif i == self._num_cells-1:
        fron_diff = cells[i-1].get_u_mean() - cells[i].get_u_mean()
      else:
        fron_diff = cells[i-1].get_u_mean() - cells[i].get_u_mean()
        back_diff = cells[i+1].get_u_mean() - cells[i].get_u_mean()
      b_vec[i] = np.outer(cells[i].f_p0_vec(cells[i].x_left()), fron_diff) + \
                 np.outer(cells[i].f_p0_vec(cells[i].x_right()), back_diff)
    b_vec *= self._p[0]
    for i in range(7):
      for j in range(self._num_cells):
        if j == 0:
          coef_m, coef_r = cells[j].get_coef(), cells[j+1].get_coef()
          temp = np.dot(np.dot(self._A_inv[j], self._B_mat[j+1]), coef_r)
        elif j == self._num_cells-1:
          coef_l, coef_m = cells[j-1].get_coef(), cells[j].get_coef()
          temp = np.dot(np.dot(self._A_inv[j], np.transpose(self._B_mat[j])), coef_l)
        else:
          coef_l, coef_m, coef_r = cells[j-1].get_coef(), cells[j].get_coef(), cells[j+1].get_coef()
          temp = np.dot(np.dot(self._A_inv[j], np.transpose(self._B_mat[j])), coef_l) + \
          np.dot(np.dot(self._A_inv[j], self._B_mat[j+1]), coef_r)
        coef_m = coef_m * (-0.3) + 1.3 * (temp + np.dot(self._A_inv[j], b_vec[j]))
        cells[j].set_coef(coef_m)

  def reconstruct_reflect(self, cells):
    b_vec = np.zeros((self._num_cells, self._degree, cells[0].num_equa()))
    for i in range(self._num_cells):
      fron_diff, back_diff = np.zeros(cells[i].num_equa()), np.zeros(cells[i].num_equa())
      if i == 0:
        fron_diff[1] -= cells[i].get_u_mean()[1] * 2
        back_diff = cells[i+1].get_u_mean() - cells[i].get_u_mean()
      elif i == self._num_cells-1:
        fron_diff = cells[i-1].get_u_mean() - cells[i].get_u_mean()
        back_diff[1] -= cells[i].get_u_mean()[1] * 2
      else:
        fron_diff = cells[i-1].get_u_mean() - cells[i].get_u_mean()
        back_diff = cells[i+1].get_u_mean() - cells[i].get_u_mean()
      b_vec[i] = np.outer(cells[i].f_p0_vec(cells[i].x_left()), fron_diff) + \
                 np.outer(cells[i].f_p0_vec(cells[i].x_right()), back_diff)
    b_vec *= self._p[0]
    for i in range(7):
      for j in range(self._num_cells):
        if j == 0:
          coef_m, coef_r = cells[j].get_coef(), cells[j+1].get_coef()
          temp = np.dot(np.dot(self._A_inv[j], self._B_mat[j+1]), coef_r)
        elif j == self._num_cells-1:
          coef_l, coef_m = cells[j-1].get_coef(), cells[j].get_coef()
          temp = np.dot(np.dot(self._A_inv[j], np.transpose(self._B_mat[j])), coef_l)
        else:
          coef_l, coef_m, coef_r = cells[j-1].get_coef(), cells[j].get_coef(), cells[j+1].get_coef()
          temp = np.dot(np.dot(self._A_inv[j], np.transpose(self._B_mat[j])), coef_l) + \
          np.dot(np.dot(self._A_inv[j], self._B_mat[j+1]), coef_r)
        coef_m = coef_m * (-0.3) + 1.3 * (temp + np.dot(self._A_inv[j], b_vec[j]))
        cells[j].set_coef(coef_m)

  def _initialize_p_array(self, d, degree):
    self._p = np.zeros(degree+1)
    for i in range(degree+1):
      self._p[i] = pow(d, 2*i) * self._w[i]**2 / math.factorial(i) ** 2
  
  def _get_A_mat_inner(self, cell):
    A_mat = self._get_A_mat(cell, cell.x_left(), self._degree+1) + \
            self._get_A_mat(cell, cell.x_right(), self._degree+1)
    return A_mat

  def _get_A_mat_end(self, cell, orient):
    if orient < 0:
      return self._get_A_mat(cell, cell.x_left(), 1)*2 + self._get_A_mat(cell, cell.x_right(), self._degree+1)
    else:
      return self._get_A_mat(cell, cell.x_left(), self._degree+1) + self._get_A_mat(cell, cell.x_right(), 1)*2
  
  def _get_B_mat_end(self, cell_l, x_l, cell_r, x_r):
    B_mat = np.zeros((self._degree, self._degree))
    i0, j0 = cell_l.f_p0_vec(x_l), cell_r.f_p0_vec(x_r)
    for i in range(self._degree):
      for j in range(self._degree):
        B_mat[i,j] += i0[i] * j0[j] * self._p[0]
    return B_mat
  
class VrP1Approach(VrApproach):

  def __init__(self):
    self._degree = 1
    self._w = np.ones(self._degree+1)
    self._w[0:2] *= 1

  def _get_B_mat_inner(self, cell_l, x_l, cell_r, x_r):
    B_mat = np.zeros((self._degree, self._degree))
    fi = [cell_l.f_p0_vec(x_l), cell_l.f_p1_vec(x_l)]
    fj = [cell_r.f_p0_vec(x_r), cell_r.f_p1_vec(x_r)]
    for i in range(self._degree):
      for j in range(self._degree):
        for k in range(self._degree+1):
          B_mat[i,j] += fi[k][i] * fj[k][j] * self._p[k]
    return B_mat

  def _get_A_mat(self, cell, x, p):
    f = [cell.f_p0_vec(x), cell.f_p1_vec(x)]
    A_mat = np.zeros((self._degree, self._degree))
    for k in range(p):
      A_mat += f[k] * f[k] * self._p[k]
    return A_mat

class VrP2Approach(VrApproach):

  def __init__(self):
    self._degree = 2
    self._w = np.ones(self._degree+1)
    self._w[0:2] *= 1

  def _get_B_mat_inner(self, cell_l, x_l, cell_r, x_r):
    B_mat = np.zeros((self._degree, self._degree))
    fi = [cell_l.f_p0_vec(x_l), cell_l.f_p1_vec(x_l), cell_l.f_p2_vec(x_l)]
    fj = [cell_r.f_p0_vec(x_r), cell_r.f_p1_vec(x_r), cell_r.f_p2_vec(x_r)]
    for i in range(self._degree):
      for j in range(self._degree):
        for k in range(self._degree+1):
          B_mat[i,j] += fi[k][i] * fj[k][j] * self._p[k]
    return B_mat

  def _get_A_mat(self, cell, x, p):
    A_mat = np.zeros((self._degree, self._degree))
    f = [cell.f_p0_vec(x), cell.f_p1_vec(x), cell.f_p2_vec(x)]
    for i in range(self._degree):
      for j in range(self._degree):
        for k in range(p):
          A_mat[i,j] += f[k][i] * f[k][j] * self._p[k]
    return A_mat

class VrP3Approach(VrApproach):

  def __init__(self):
    self._degree = 3
    self._w = np.ones(self._degree+1)
    self._w[0:2] *= 1

  def _get_B_mat_inner(self, cell_l, x_l, cell_r, x_r):
    B_mat = np.zeros((self._degree, self._degree))
    fi = [cell_l.f_p0_vec(x_l), cell_l.f_p1_vec(x_l), cell_l.f_p2_vec(x_l), cell_l.f_p3_vec(x_l)]
    fj = [cell_r.f_p0_vec(x_r), cell_r.f_p1_vec(x_r), cell_r.f_p2_vec(x_r), cell_r.f_p3_vec(x_r)]
    for i in range(self._degree):
      for j in range(self._degree):
        for k in range(self._degree+1):
          B_mat[i,j] += fi[k][i] * fj[k][j] * self._p[k]
    return B_mat

  def _get_A_mat(self, cell, x, p):
    A_mat = np.zeros((self._degree, self._degree))
    f = [cell.f_p0_vec(x), cell.f_p1_vec(x), cell.f_p2_vec(x), cell.f_p3_vec(x)]
    for i in range(self._degree):
      for j in range(self._degree):
        for k in range(p):
          A_mat[i,j] += f[k][i] * f[k][j] * self._p[k]
    return A_mat

  