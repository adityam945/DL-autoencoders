#Three lines to make our compiler able to draw:
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([
2476.197021,
52086.323891,
102003.139484,
151966.931619,
201944.777285,
251928.312131,
301916.117230,
351906.795615
])

plt.plot(xpoints)
plt.show()

#Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()