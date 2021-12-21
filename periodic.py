import numpy as np
import math

if __name__ == '__main__':
# Set Solver
  from fvm import VrSolver
  solver = VrSolver(3)
  # from fvm import LrSolver
  # solver = LrSolver()

# Set Initial Condition:
  def initial(x):
    return np.sin(8 * x * np.pi)*1.5 + 2
    # return np.sign(x-0.5) + 1.5


  def initial(x):
    if x > 0.1 and x < 0.4:
      return 4 - np.abs(x-0.25) * 20
    elif x > 0.6 and x < 0.9:
      return 4
    else:
      return 1

  solver.set_initial_condition(func=lambda x: initial(x))
  solver.set_boundary_condition(boundary="periodic")

# Set Mesh:
  x_min = 0.0
  x_max = 1.0
  x_num = 100
  solver.set_mesh(x_min = x_min, x_max = x_max, x_num = x_num)
# Set Time Scheme:
  start = 0.0
  stop = 1.0
  t_num = 100
  solver.set_time_stepper(start = start, stop = stop, t_num = t_num)

# Set Riemann Problem:
  from riemann import Linear
  riemann = Linear(1.0)
  # from riemann import Burgers
  # riemann = Burgers()
  solver.set_riemann_solver(riemann = riemann)

# Set Limiter
  from limiter import BJLimiter
  from limiter import NewLimiter
  # limiter = BJLimiter(10)
  limiter = NewLimiter(2)
  solver.set_limiter(limiter)

  from limiter import EdgeIndicator
  from limiter import RenIndicator
  # indicator = EdgeIndicator(1)
  indicator = RenIndicator(1)
  solver.set_indicator(indicator)

  x_vec = solver.get_x_vec()
  t_vec = solver.get_t_vec()

  from displayer import Transient
  # solver.run_with_transient()
  # u_mat = solver.get_u_mat()
  # transient = Transient()
  # transient.add_plot(x_vec = x_vec, u_vec = u_mat, type = "k", label = "U")
  # transient.display()

  from displayer import Animation
  solver.run_with_animation()
  u_mat = solver.get_u_mat()
  animation = Animation(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
  animation.display(type = "k--", y_min = -1, y_max = 5)

  from displayer import Contour
  # trouble_mat = solver.get_trouble_history()
  # t_vec = np.linspace(start=start, stop=stop, endpoint=True, num=t_num*3)
  # contour = Contour(x_vec=x_vec, t_vec=t_vec, u_mat=trouble_mat)
  # contour.display()


