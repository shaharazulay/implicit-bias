# # Fully Connected Linear Networks (Section 6)

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'figure.dpi': 300})

import numpy as np
from scipy.optimize import minimize
from numpy.random import multivariate_normal
import torch
import torch.nn as nn
import torch.nn.functional as F


# ### Define General Setting

d = 1000
N = 100
np.random.seed(1993)  # set seed for repetative results.


# ### Define Dataset (Woodworth et al.)

s = 5
noise_var = 0.01

X_train = multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=N)
X_test = multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=2000)
w_star = np.zeros(d)
support = np.random.choice(np.arange(d), s, replace=False)
w_star[support] = 1/np.sqrt(s)
y_train = np.dot(X_train, w_star) + np.sqrt(noise_var) * np.random.randn(N)
y_test = np.dot(X_test, w_star)

X_train = torch.DoubleTensor(X_train)
X_test = torch.DoubleTensor(X_test)
y_train = torch.DoubleTensor(y_train)
y_test = torch.DoubleTensor(y_test)


# ## Define Depth $2$ Fully Connected Network (Section 6)

def symmetric_init(alpha, s, m, d, symmetric=True):
    """
    alpha = |a_0| + ||w_0||
    s = (|a_0| - ||w_0||) / (|a_0| + ||w_0||)
    """
    norms_w = []
    norms_a = []
    for alpha_i, s_i in zip(alpha, s):
        norm_w = np.sqrt(alpha_i * (1 - s_i) / (1 + s_i))
        norm_a = np.sqrt(alpha_i * (1 + s_i) / (1 - s_i))
        norms_w.append(norm_w)
        norms_a.append(norm_a)

    w_0 = np.random.normal(size=(m, d), loc=0, scale=1)
    w_0_norms = np.linalg.norm(w_0, axis=1, ord=2)
    w_0 = w_0 / w_0_norms[:, np.newaxis] * np.array(norms_w)[:, np.newaxis]

    a_0 = np.random.normal(size=(1, m), loc=0, scale=1)
    a_0 = np.multiply(np.ones_like(a_0) * np.array(norms_a), (1 * (a_0 > 0) - 0.5) * 2)

    if symmetric:
        w_0[m // 2:, :] = w_0[:m // 2, :]
        a_0[:, m // 2:] = -a_0[:, :m // 2]

    return w_0.transpose(), a_0.reshape(-1,)


class TwoLayerModel(nn.Module):
    def __init__(self, alpha, s, m, d, use_relu=False, symmetric=True):
        super(TwoLayerModel, self).__init__()

        self.w = nn.Parameter(torch.DoubleTensor(d, m))
        self.a = nn.Parameter(torch.DoubleTensor(m))

        if use_relu:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        with torch.no_grad():
            # unbiased init
            w_0, a_0 = symmetric_init(alpha, s, m, d, symmetric=symmetric)
            self.w.data, self.a.data = torch.DoubleTensor(w_0), torch.DoubleTensor(a_0)
            self._w_0 = self.w.transpose(1, 0).clone().detach().numpy()
            self._a_0 = self.a.view(-1, 1).clone().detach().numpy()

    def forward(self, x):
        x = torch.matmul(x, self.w)
        x = self.activation(x)
        x = torch.mv(x, self.a)
        return x
    
    @property
    def current_w(self):
        return self.w.transpose(1, 0).detach().numpy()
    
    @property
    def current_a(self):
        return self.a.view(-1, 1).detach().numpy()

    @property
    def w_0(self):
        return self._w_0

    @property
    def a_0(self):
        return self._a_0
    
    
def train(alpha, s, m, d, step_size, use_relu=False, symmetric=True, early_stop_tol=1e-5, num_epochs=int(1e7)):
    model = TwoLayerModel(alpha=alpha, s=s, m=m, d=d, use_relu=use_relu, symmetric=symmetric)
    optimizer = torch.optim.SGD(model.parameters(), lr=step_size)

    w_tilde_array = []
    a_array = []
    w_tilde_array.append(model.current_w * model.current_a)  # get the initial state
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()

        w_tilde_array.append(model.current_w * model.current_a)
        a_array.append(model.current_a.copy())
        if loss.item() < early_stop_tol:
            print(f'early stopping after {epoch + 1} epochs')
            break

    with torch.no_grad():
        test_loss = F.mse_loss(model(X_test), y_test)
    return model, loss, test_loss, w_tilde_array, a_array


# ## Define Q Function (Theorem 6.1)

def q(w_i, w_i_0, delta_i):
    d = delta_i/2
    norm_w_i = np.linalg.norm(w_i, ord=2)
    norm_w_i_0 = np.linalg.norm(w_i_0, ord=2)
    
    return (1/norm_w_i) * (norm_w_i**2 - d * (d + np.sqrt(norm_w_i**2 + d**2))) * np.sqrt(np.sqrt(norm_w_i**2 + d**2) - d)        - (3/2) * np.sqrt(np.sqrt(norm_w_i_0**2 + d**2) - d) * np.matmul(w_i_0.transpose(), w_i) / norm_w_i_0


def Q(w, w_0, delta):
    f = 0
    for w_i, w_i_0, delta_i in zip(w, w_0, delta):
        f += q(w_i, w_i_0, delta_i)
    return f


def flatten(w, a):
    return np.hstack([w.copy().flatten(), a.copy().flatten()])


def restore(v, m, d):
    w = v.copy()[: m * d].reshape(m, d)
    a = v.copy()[m * d:].reshape(m, 1)
    return w, a


def constraint(v, x, y, m, d):
    w, a = restore(v, m, d)
    w_tilde = w * a
    return (np.matmul(w_tilde, x.transpose()).sum(axis=0) - y).squeeze()


def constraint_relu(v, x, y, m, d):
    w, a = restore(v, m, d)
    y_pred = (np.maximum(np.matmul(w, x.transpose()), 0) * a).sum(axis=0)
    return (y_pred - y).squeeze()


def solver(x, y, w_0, a_0, alpha, s, obj='L1', optim_tol=1e-5, use_relu=False):
    x0 = flatten(w_0, a_0)

    if use_relu:
        cons = {'type': 'eq', 'fun': lambda v: constraint_relu(v, x, y, m, d)}
    else:
        cons = {'type': 'eq', 'fun': lambda v: constraint(v, x, y, m, d)}

    if obj == 'L1':
        objective = lambda v: np.linalg.norm(v, ord=2)
    elif obj == 'Q':
        delta = [4 * alpha_i * s_i / (1 - s_i**2) for alpha_i, s_i in zip(alpha, s)]
        w_tilde_0 = w_0 * a_0
        objective = lambda v: Q(restore(v, m, d)[0] * restore(v, m, d)[1] , w_tilde_0, delta)
    else:
        raise ValueError('objective not supported.')

    sol = minimize(
        fun=objective,
        x0=x0,
        constraints=cons,
        tol=optim_tol,
        method='SLSQP',
        options={
            'maxiter': 100000,
            'disp': False
        }
    )
    is_failed = (not sol.success)
    if is_failed:
        raise RuntimeError('Minimization Failed.')

    return restore(sol.x, m, d)


# ## Linear Network with a Single Hidden Neuron ($m= 1$) (Theorem 6.1)

m = 1
alpha_range = [0.01, 0.1]
s_range = [0, 0.5, 0.95]

results = {}
for s in s_range:
    for alpha in alpha_range:
        model, train_loss, test_loss, w_tilde_array, _ = train(
            alpha=[alpha],
            s=[s],
            m=m,
            d=d,
            step_size=1e-3,
            symmetric=False,
            early_stop_tol=1e-10)
        print(train_loss.item(), test_loss.item())
        results[(s, alpha)] = {'w_tilde_array': w_tilde_array, 'model': model}


linstyles = ['', ':', '-.', '--']

legend_ = []
idx = 0
plt.figure()
for alpha in alpha_range:
    for s in s_range:
        legend_.append(f'α={alpha}, s={s}')
        print(s, alpha)
        model = results[(s, alpha)]['model']
        w_tilde_array = results[(s, alpha)]['w_tilde_array']

        w_Q, a_Q = solver(X_train.numpy(), y_train.numpy(), model.w_0, model.a_0, alpha=[alpha], s=[s], obj='Q', optim_tol=1e-7)
        w_tilde_Q = w_Q * a_Q

        plt.plot([np.linalg.norm(w_tilde.flatten() - w_tilde_Q.flatten(), ord=2)
             for w_tilde in w_tilde_array], linstyles[idx % len(linstyles)])

        idx += 1

plt.title('Distance between Q minimization and GD solution')
plt.ylabel('$\ell_2$ distance')
plt.xlabel('epoch')
plt.xscale('log')
plt.yscale('log')
plt.legend(legend_)


# ## Effect of Shape and Scale (Section 7)

# Visualize q_hat (d = 2)

def q_hat(x, delta_i):
    d = delta_i/2
    
    return (1/x) * (x**2 - d * (d + np.sqrt(x**2 + d**2))) * np.sqrt(np.sqrt(x**2 + d**2) - d)


linstyles = ['', ':', '-.', '--']

x_range = np.arange(0, 10, 0.5)
delta_range = [0, 1, 2, 5]

plt.figure()
for delta_i in delta_range:
    plt.plot(list(map(lambda x: q_hat(x, delta_i), x_range)))

plt.title('The function q_hat (first element of the implicit bias)')
plt.ylabel('')
plt.xlabel('x')
plt.legend([f'$\delta = {delta_i}$' for delta_i in delta_range])
plt.yticks([])


# Visualize the full q function (d = 2)

w1 = np.arange(-10.0, 10.0, 0.1)
w2 = np.arange(-10.0, 10.0, 0.1)

Z = np.zeros((len(w1), len(w2)))

fig, axes = plt.subplots(2, 3)
axes = axes.ravel()

# setting initializaiton
w_0 = np.array([0.6, 0.8])

# changing delta
alpha = 2
for ax_idx, s in enumerate([0, 0.2, 0.8]):
    for i, w1_i in enumerate(w1):
        for j, w2_j in enumerate(w2):
            w = np.array([w1_i, w2_j])
            delta = 4 * alpha * s / (1 - s**2)
            z_value = q(w, alpha * w_0, delta)
            Z[i, j] = z_value

    ax = axes[ax_idx]
    CS = ax.contour(Z)
    ax.axhline(y=100, xmin=0, xmax=200, c='grey', linewidth=0.5, zorder=0)
    ax.axvline(x=100, ymin=0, ymax=200, c='grey', linewidth=0.5, zorder=0)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.set_aspect(1)
    ax.plot([alpha * w_0[1]/0.1 + 100], [alpha * w_0[0]/0.1 + 100], 'ro', markersize=3)
    ax.clabel(CS, inline=1, fontsize=5)

# changing alpha
s = 0.1
for ax_idx, alpha in enumerate([0.01, 1, 2.5]):
    for i, w1_i in enumerate(w1):
        for j, w2_j in enumerate(w2):
            w = np.array([w1_i, w2_j])
            delta = 4 * alpha * s / (1 - s**2)
            z_value = q(w, alpha * w_0, delta)
            Z[i, j] = z_value

    ax = axes[ax_idx + 3]
    CS = ax.contour(Z)
    ax.axhline(y=100, xmin=0, xmax=200, c='grey', linewidth=0.5, zorder=0)
    ax.axvline(x=100, ymin=0, ymax=200, c='grey', linewidth=0.5, zorder=0)
    ax.plot([alpha * w_0[1]/0.1 + 100], [alpha * w_0[0]/0.1 + 100], 'ro', markersize=3)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.set_aspect(1)
    ax.clabel(CS, inline=1, fontsize=5)

plt.show()
