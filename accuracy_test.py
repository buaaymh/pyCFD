import abc
import csv
import pandas as pd
import numpy as np
import math
from displayer import Transient
from spectralAnalysis import vanAlbada
from spectralAnalysis import WENO_Z3
from spectralAnalysis import ROUNDL
from spectralAnalysis import Variational3

def function(x):
    return np.sin(np.pi*x)

def timeStepRK3func(reconstruct, u_old, dt):
    u_1 = u_old - dt * reconstruct.du_dx(u_old)
    u_2 = (3 * u_old + u_1 - dt * reconstruct.du_dx(u_1))/4
    u_new = (u_old + 2*(u_2 - dt * reconstruct.du_dx(u_2)))/3
    return u_new

if __name__ == '__main__':
    
    x_min = -1
    x_max =  1
    t_end =  2
    n_level = 5
    reconstrctors = dict()
    nCells_list = np.zeros(n_level)
    errorL1_list = dict()
    errorL1_list["vanAlbada"] = np.zeros(n_level)
    errorL1_list["WENO_Z3"] = np.zeros(n_level)
    errorL1_list["ROUNDL"] = np.zeros(n_level)
    errorL1_list["Variational3"] = np.zeros(n_level)
    errorL2_list = dict()
    errorL2_list["vanAlbada"] = np.zeros(n_level)
    errorL2_list["WENO_Z3"] = np.zeros(n_level)
    errorL2_list["ROUNDL"] = np.zeros(n_level)
    errorL2_list["Variational3"] = np.zeros(n_level)
    # Set Mesh_1:
    x_num = 10
    for i in range(n_level):
        x_num *= 2
        nCells_list[i] = x_num
        dx    = (x_max - x_min) / x_num
        t_num = x_num
        dt    = t_end / t_num
        x_vec = np.linspace(start = x_min + dx/2, stop = x_max - dx/2, num = x_num)
        reconstrctors["vanAlbada"] = vanAlbada(x_vec, dx)
        reconstrctors["WENO_Z3"] = WENO_Z3(x_vec, dx)
        reconstrctors["ROUNDL"] = ROUNDL(x_vec, dx)
        reconstrctors["Variational3"] = Variational3(x_vec, dx)
        for name, reconstructor in reconstrctors.items():
            u_vec = function(x_vec)
            for t in range(t_num):
                u_vec = timeStepRK3func(reconstructor, u_vec, dt)
            errorL1_list[name][i] = np.linalg.norm(u_vec-function(x_vec), ord=1)/x_num
            errorL2_list[name][i] = np.linalg.norm(u_vec-function(x_vec), ord=2)/np.sqrt(x_num)

    # write csv file
    data = dict()
    data["nCells"] = nCells_list
    for name, reconstructor in reconstrctors.items():
        data[name + "_L1"] = errorL1_list[name]
        data[name + "_L2"] = errorL2_list[name]
    test=pd.DataFrame(data)
    test.to_csv("result/accuracy.csv") # 如果生成excel，可以用to_excel
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("L1")
    ax2.set_xlabel("x")
    ax2.set_ylabel("L2")
    for name, reconstructor in reconstrctors.items():
        ax1.loglog(nCells_list, errorL1_list[name], base = 2, label=name)
        ax2.loglog(nCells_list, errorL2_list[name], base = 2, label=name)

    ax1.legend(loc='upper right', shadow=False, fontsize='medium')
    ax2.legend(loc='upper right', shadow=False, fontsize='medium')
    ax1.grid(True)
    ax2.grid(True)
    plt.show()
    




    