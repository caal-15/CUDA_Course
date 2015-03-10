import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
  return fromfile(fname, sep='\n')

x = m_load('build/x.mio')
y_con = m_load('build/y_con.mio')
y_con_tiled = m_load('build/y_con_tiled.mio')
plt.title ('Matrix Multiplication Exec. Time comparison (GTX 760)')
plt.xlabel ('Matrix size')
plt.ylabel ('Execution time (seconds)')
plt.plot (x, y_con, 'r', label = 'Non-Tiled Time')
plt.plot (x, y_con_tiled, 'b', label = 'Tiled Time')
plt.legend (loc='upper left')


plt.hold()
plt.show()
