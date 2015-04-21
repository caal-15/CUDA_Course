from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def m_load(fname) :
  return fromfile(fname, sep='\n')

N = 6

ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars

fig, ax = plt.subplots()

y_seq = m_load('build/y_seq.mio')
rects1 = ax.bar(ind, y_seq, width, color='r')

y_con_g = m_load('build/y_con_g.mio')
rects2 = ax.bar(ind + width, y_con_g, width, color='b')

y_con_c = m_load('build/y_con_c.mio')
rects3 = ax.bar(ind + width * 2, y_con_c, width, color='g')

y_con_t = m_load('build/y_con_t.mio')
rects4 = ax.bar(ind + width * 3, y_con_t, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('Execution Time (Seconds)')
ax.set_title('Convolution Execution Time Comaprison')
ax.set_xticks(ind + width)
ax.set_xticklabels( ('IMG1', 'IMG2', 'IMG3', 'IMG4', 'IMG5', 'IMG6') )

ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('Seq (Core i7 3770 Optimized)', 'Con (GTX760 Global)', 'Con (GTX760 Constrant)', 'Con (GTX760 Constrant + Tiled)'), loc = 'upper left' )

plt.show()
