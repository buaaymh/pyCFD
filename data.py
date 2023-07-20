import numpy as np
import csv

import displayer
# coding=gbk
def read_cols(filename):
  data = list()
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    for i in reader:
      data.append(i)
  return np.array(data, dtype=np.float64).transpose()

def read_rows(filename):
  data = list()
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    for i in reader:
      data.append(i)
  return np.array(data, dtype=np.float64)

def write_rows(filename, data):
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)

if __name__ == '__main__':
  from displayer import Transient
  transient = Transient(u_label = "Density")
  filename = "result/Shu-Osher/weno_5000.csv"
  xu1 = read_rows(filename)
  with open("Shu_weno_5000.csv", 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(["x", "rho"])
      for i in range(len(xu1[0])):
        writer.writerow([xu1[0][i], xu1[1][i]])

  transient.add_plot(x_vec = xu1[0], u_vec = xu1[1], type = "k", label = "Exact")
  # filename = "result/Blast/p3_400_new1.csv"
  # xu2 = read_rows(filename)
  # transient.add_plot(x_vec = xu2[0], u_vec = xu2[1],type = "b", label = "Numerical")
  transient.display()
