import os

d = './'

array2D = []

for filename in os.listdir(d):
    if not filename.endswith('.txt'):
        continue

    with open(filename, 'r') as f:
        for line in f.readlines():
            array2D.append(line.split(' '))

print(array2D)

y = []
x = []

for i in range(len(array2D)):
    x.append(array2D[i][0])
    y.append(array2D[i][1][:-3])

import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


# y = [331.065585, 294.247678, 286.768601,274.506764, 264.810079, 258.094461, 252.851443, 250.495644, 248.457700, 246.949334, 245.793287]
# cifar
y = [1847.422091, 1837.368806, 1833.603732, 1831.013325, 1829.006313, 1827.250325,1826.834943, 1825.614124, 1825.163747, 1825.177202]
# 
# y = [16026.227047, 15955.823141, 15880.535062, 15905.362969, 15883.614922]
print(len(y))
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# x = [0,1, 2, 3, 4]
print(len(y), len(x))


xpoints = np.array(x)
ypoints = np.array(y)


plt.plot(ypoints, xpoints)
plt.xlabel('Loss')
plt.ylabel('epochs')
plt.savefig('cifar')
# plotting a line plot after changing it's width and height
f = plt.figure()
f.set_figwidth(40)
f.set_figheight(40)
f.savefig('test2png.png', dpi=100)


sys.stdout.flush()



