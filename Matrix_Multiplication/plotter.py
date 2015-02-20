import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
  return fromfile(fname, sep='\n')

x = m_load('build/x.mio')
y_seq = m_load('build/y_seq.mio')
y_con = m_load('build/y_con.mio')
plt.plot(x, y_seq, 'r')
plt.plot(x, y_con, 'b')
plt.hold()
plt.show()
