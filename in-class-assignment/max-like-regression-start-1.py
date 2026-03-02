import numpy as np
import matplotlib.pyplot as plt

plt.ion()

num_points = 100
x_data = np.linspace(-1, 1, num_points)
e = np.random.normal(0, 0.2, num_points)
m = np.random.rand()
b = np.random.rand()
y_data = m*x_data + b + e

fig, ax = plt.subplots()
fig = plt.gcf()

ax.scatter(x_data, y_data)
ax.plot(x_data, m*x_data + b, color='red')

def model(m, b=b):
    return m*x_data + b

def loss(m,b=b):
    raise NotImplementedError

def grad_loss(m, b=b):
    raise NotImplementedError


mhat = 1*np.random.rand() + 1
bhat = 1*np.random.rand() + 1
ax.plot(x_data, model(mhat, bhat), color='green')
plt.legend(['Data', 'Exact', 'Random'])
plt.show()

learning_rate = 0.005
for i in range(10):
    break
    raise NotImplementedError
    print(loss(mhat))
    if len(ax.lines) > 1:
        for art in reversed(list(ax.lines)):
            art.remove()
            break
    ax.plot(x_data, mhat*x_data + b, '--', lw=3, color ='black')
    plt.pause(0.5)


plt.ioff()
plt.show()


