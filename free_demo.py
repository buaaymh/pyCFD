import numpy as np
import math
from time import*

from fvm import VrSolver

if __name__ == '__main__':
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

    # def initial(x):
    #     if x < 1:
    #         return euler.u_p_rho_to_U(u=2.629369, p=10.333333, rho=3.857143)
    #     else:
    #         return euler.u_p_rho_to_U(u=0, p=1, rho=1+0.2*np.sin(5*x))

    solver = VrSolver(3)
    
    # Set Mesh:
    x_min = 0.0
    x_max = 1.0
    x_num = 200
    solver.set_mesh(x_min = x_min, x_max = x_max, x_num = x_num)
    solver.set_initial_condition(func=lambda x: initial(x))
    solver.set_boundary_condition(boundary="free")

    # Set Time Scheme:
    start = 0.0
    stop = 0.2
    t_num = 100
    solver.set_time_stepper(start = start, stop = stop, t_num = t_num)
    # Set Riemann Problem:
    # from riemann import EulerAusm
    # riemann = EulerAusm()
    from riemann import EulerVanLeer
    riemann = EulerVanLeer()
    solver.set_riemann_solver(riemann = riemann)

    # Set Limiter
    from limiter import NewLimiter
    from limiter import EulerNewLimiter
    # limiter = BJLimiter(10)
    # limiter = NewLimiter(1)
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
    transient.add_plot(x_vec = x_vec, u_vec = rho_vec, type = "k.", label = r'$\rho(x)$')
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
