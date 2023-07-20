from time import*
import numpy as np
import csv
from fdm import FirstOrderScheme
from fdm import WenoEulerScheme

if __name__ == '__main__':
    # Set Scheme:
    # solver = FirstOrderScheme()
    solver = WenoEulerScheme()

    # Set Initial Condition:
    from equation import Euler1d
    euler = Euler1d(gamma=1.4)
    U_l = euler.u_p_rho_to_U(u=0, p=1.0, rho=1.0)
    U_r = euler.u_p_rho_to_U(u=0, p=0.1, rho=0.125)

    def initial(x):
        if x < 0.5:
          return U_l
        else:
          return U_r
    
    # Set Mesh:
    x_min = 0.0
    x_max = 1.0
    x_num = 5000
    solver.set_mesh(x_min = x_min, x_max = x_max, x_num = x_num)
    solver.set_initial_condition(func=lambda x: initial(x))
    solver.set_boundary_condition(boundary="explosion")

    # Set Time Scheme:
    start = 0.0
    stop = 0.2
    t_num = 2000
    solver.set_time_stepper(start = start, stop = stop, t_num = t_num)

    from riemann import EulerVanLeer
    riemann = EulerVanLeer()
    solver.set_riemann_solver(riemann = riemann)

    x_vec = solver.get_x_vec()
    t_vec = solver.get_t_vec()

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
    transient = Transient(u_label = "U")
    transient.add_plot(x_vec = x_vec, u_vec = rho_vec, type = "k", label = r'$\rho$')
    transient.add_plot(x_vec = x_vec, u_vec = u_vec, type = "r", label = r'$u$')
    transient.add_plot(x_vec = x_vec, u_vec = p_vec, type = "b", label = r'$p$')
    transient.display()

    with open("explosion.csv", 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(["x", "rho", "u", "p"])
      for i in range(len(x_vec)):
        writer.writerow([x_vec[i], rho_vec[i], u_vec[i], p_vec[i]])