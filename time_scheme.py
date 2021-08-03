import abc
import numpy as np

class TimeScheme(abc.ABC):

    def set_rhs(self, rhs):
        self._rhs = rhs

class RK3ForFDM(TimeScheme):

    def __init__(self):
        pass

    def get_u_new(self, u_old, dt):
        u_1 = u_old + dt * self._rhs(u_old)
        u_2 = (3 * u_old + u_1 + dt * self._rhs(u_1))/4
        u_new = (u_old + 2*(u_2 + dt * self._rhs(u_2)))/3
        return u_new

class RK3ForVr(TimeScheme):
    def __init__(self):
        pass
    
    def get_cells_new(self, cells_old, dt):
        # step 1
        u_mean_old = self._get_u_mean(cells_old)
        u_mean_1 = u_mean_old + dt * self._rhs(cells_old)
        self._update(cells_old, u_mean_1)
        # step 2
        u_mean_2 = (3 * u_mean_old + u_mean_1 + dt * self._rhs(cells_old))/4
        self._update(cells_old, u_mean_2)
        # step 3
        u_mean_new = (u_mean_old + 2*(u_mean_2 + dt * self._rhs(cells_old)))/3
        self._update(cells_old, u_mean_new)
    
    def _update(self, cells, u_means):
        for i in range(len(cells)):
            cells[i].set_u_mean(u_means[i])
    
    def _get_u_mean(self, cells):
        u_mean = np.zeros((len(cells), cells[0].num_equa()))
        for i in range(len(cells)):
            u_mean[i] = cells[i].get_u_mean()
        return u_mean
