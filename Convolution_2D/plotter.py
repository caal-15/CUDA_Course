import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
  return fromfile(fname, sep='\n')

x = m_load('build/x.mio')
#y_seq = m_load('build/y_seq.mio')
y_con_g = m_load('build/y_con_g.mio')
y_con_c = m_load('build/y_con_c.mio')
y_con_t = m_load('build/y_con_t.mio')
plt.title ('Convolution 2D Time comparison')
plt.xlabel ('Pixel Count')
plt.ylabel ('Execution time (seconds)')
#plt.plot (x, y_seq, 'r', label = 'Seq Time (Optimized, Core i7 3770)')
plt.plot (x, y_con_g, 'b', label = 'Parallel Time (Global memory, GTX 760)')
plt.plot (x, y_con_c, 'g', label = 'Parallel Time (Constant memory, GTX 760)')
plt.plot (x, y_con_t, 'y', label = 'Parallel Time (Tiled, Constant memory, GTX 760)')
plt.legend (loc='upper left')


plt.hold()
plt.show()
