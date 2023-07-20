import abc
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as colormap
from matplotlib.animation import FuncAnimation

class Displayer(abc.ABC):

    @abc.abstractmethod
    def display(self):
        pass

class Contour(Displayer):  

    def __init__(self, x_vec, t_vec, u_mat):
        assert len(t_vec) == len(u_mat)
        assert len(x_vec) == len(u_mat[0])
        assert sorted(x_vec)
        assert sorted(t_vec)
        self.x_vec = x_vec
        self.t_vec = t_vec
        self.u_mat = u_mat

    def display(self, x_label = "x", y_label = "T"):
        x_grid, t_grid = np.meshgrid(self.x_vec, self.t_vec)
        fig, axis = plt.subplots()
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        cs = axis.pcolormesh(x_grid, t_grid, self.u_mat, shading='auto', cmap='Greys')
        fig.colorbar(cs)
        plt.show()

class Transient(Displayer):

    def __init__(self, u_label = "U"):
        self.fig, self.axis = plt.subplots()
        self.axis.set_xlabel("x")
        self.axis.set_ylabel(u_label)

    def add_plot(self, x_vec, u_vec, label = "Undefined"):
        self.axis.plot(x_vec, u_vec, label=label)

    def display(self):
        self.axis.grid(True)
        legend = self.axis.legend(loc='upper right', shadow=False, fontsize='medium')
        plt.show()
    
    def save_fig(self, filename):
        legend = self.axis.legend(loc='upper right', shadow=False, fontsize='medium')
        plt.savefig(filename)
    

class Animation(Displayer):

    def __init__(self, x_vec, t_vec, u_mat, u_label = "U"):
        assert len(t_vec) == len(u_mat)
        assert len(x_vec) == len(u_mat[0])
        assert sorted(x_vec)
        assert sorted(t_vec)
        self.x_vec = x_vec
        self.t_vec = t_vec
        self.u_mat = u_mat
        self.u_label = u_label

    def display(self, type = "k", y_min = -2, y_max = 2):
        fig, axis = plt.subplots()
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("x")
        axis.set_ylabel(self.u_label)
        line, = axis.plot(self.x_vec, self.u_mat[0], type, animated=False)
        axis.grid(True)
        # 	{'-', '--', '-.', ':', '', (offset, on-off-seq), ...}

        def init():
            line.set_ydata([np.nan] * len(self.x_vec))
            return line,

        def update(n):
            ti = "t = {0:.2f}". format(self.t_vec[n])
            axis.set_title(ti)
            axis.figure.canvas.draw()
            line.set_ydata(self.u_mat[n])
            return line,
        
        animation = FuncAnimation(fig, update, frames=np.arange(0, len(self.t_vec)),
                                  init_func=init, interval=100)
        plt.show()

if __name__ == '__main__':
    c = 1.0
    start = 0.0
    end = 5.0
    n = 100
    dx = (start - end) / n
    x_vec = np.linspace(start=dx/2, stop=end-dx/2, num=n)
    t_vec = np.linspace(start=0.0, stop=1.0, num=51)
    u_mat = np.zeros((len(t_vec), len(x_vec)))
    # u_0 = lambda x: np.sign(np.sin(x))
    # for i in range(len(t_vec)):
    #     for j in range(len(x_vec)):
    #         x_0 = x_vec[j] - c * t_vec[i]
    #         u_mat[i][j] = u_0(x_0)

    # d = Contour(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
    # d.display()

    d = Transient()
    d.add_plot(x_vec = x_vec, u_vec = u_mat[0], type = "k", label = "U0")
    d.add_plot(x_vec = x_vec, u_vec = u_mat[1], type = "k--", label = "U1")
    d.display()

    d = Animation(x_vec=x_vec, t_vec=t_vec, u_mat=u_mat)
    d.display(type = "k-")


    # x_vec = np.linspace(start=0.0, stop=2, num=n*10, endpoint = False)
    # u_1 = lambda x: 2*x**10 / (x**20 + 1)
    # u_new = np.zeros(len(x_vec))
    # for i in range(len(x_vec)):
    #   u_new[i] = u_1(x_vec[i])
    # d.add_plot(x_vec = x_vec, u_vec = u_new, type = "k", label = "U")
    # d.display()