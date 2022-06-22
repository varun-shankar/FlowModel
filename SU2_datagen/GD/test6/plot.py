#
import numpy as np
import matplotlib.pyplot as plt

bc0 = 3.0
bc1 = 4.0
bc2 = 5.0

l0 = np.load("log_bump_IC_0.npy") # [dvs, drags, times]
l1 = np.load("log_bump_IC_1.npy")
l2 = np.load("log_bump_IC_2.npy")

fig = plt.figure()
ax = plt.axes()
plt.xlabel("Wall Time (s)")
plt.ylabel("Bump Location")
plt.title("Adjoint Method with Different Starting Points")

for i in range(3):
    name = "log_bump_IC_" + str(i) + ".npy"
    [dv, drag, time] = np.load(name)
    ax.plot(time, dv, '-x')

plt.savefig("dv_vs_time.png")
plt.show()

fig = plt.figure()
ax = plt.axes()
plt.xlabel("Wall Time (s)")
plt.ylabel("Drag")
plt.title("Adjoint Method with Different Starting Points")

for i in range(3):
    name = "log_bump_IC_" + str(i) + ".npy"
    [dv, drag, time] = np.load(name)
    ax.plot(time, drag, '-x')

plt.savefig("drag_vs_time.png")
plt.show()
