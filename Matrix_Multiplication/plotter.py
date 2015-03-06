import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
  return fromfile(fname, sep='\n')

x = m_load('build/x.mio')
y_seq = m_load('build/y_seq.mio')
y_con = m_load('build/y_con.mio')
plt.title ('Matrix Multiplication Exec. Time comparison')
plt.xlabel ('Matrix size')
plt.ylabel ('Execution time (seconds)')
plt.plot (x, y_seq, 'r', label = 'Seq Time (Optimized, Core i7 3770)')
plt.plot (x, y_con, 'b', label = 'Parallel Time (GTX 760)')
plt.legend (loc='upper left')


plt.hold()
plt.show()
