import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12


# траектория GD
def plot_gd(X, y, w_history, title='Траектория градиентного спуска'):
    n_features = X.shape[1]
    if n_features != 2:
        print(f"skipping {title}: available for 2D only.")
        return

    A, B = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    levels = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            w_tmp = np.array([A[i, j], B[i, j]])
            levels[i, j] = np.mean(np.power(np.dot(X, w_tmp) - y, 2))

    plt.figure(figsize=(13, 9))
    plt.title(title)
    plt.xlabel(r'$w_1$')
    plt.ylabel(r'$w_2$')
    plt.xlim((-2.1, 2.1))
    plt.ylim((-2.1, 2.1))

    CS = plt.contour(A, B, levels, levels=np.logspace(0, 2, num=10), cmap=plt.cm.rainbow_r)
    plt.colorbar(CS, shrink=0.8, extend='both')

    w_list = np.array(w_history)
    w_true = np.linalg.inv(X.T @ X) @ X.T @ y
    plt.scatter(w_true[0], w_true[1], c='r', marker='*')
    plt.scatter(w_list[:, 0], w_list[:, 1])
    plt.plot(w_list[:, 0], w_list[:, 1])
    plt.show()


class LinearRegressionVectorized(BaseEstimator):
    def __init__(self, epsilon=1e-4, max_steps=2000, w0=None, alpha=1e-2, batch_size=None):
        # разница для нормы изменения весов
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.w0 = w0
        self.alpha = alpha
        self.batch_size = batch_size
        self.w = None
        self.w_history = []
        self.loss_history = []

    def fit(self, X, y):
        l, d = X.shape
        if self.w0 is None:
            self.w0 = np.zeros(d)
        self.w = self.w0.copy()

        indices = np.arange(l)
        for step in range(self.max_steps):
            self.w_history.append(self.w.copy())
            self.loss_history.append(mean_squared_error(y, self.predict(X)))

            # проверка на полном батче
            if self.batch_size is None or self.batch_size >= l:
                grad = self.calc_gradient(X, y)
            else:
                np.random.shuffle(indices)
                batch_indices = indices[:self.batch_size]
                grad = self.calc_gradient(X[batch_indices], y[batch_indices])

            self.update_weights(grad)

            if np.linalg.norm(self.w_history[-1] - self.w) < self.epsilon:
                break

        return self

    def predict(self, X):
        if self.w is None:
            raise Exception('model didn\'t train')
        return np.dot(X, self.w)

    def calc_gradient(self, X, y):
        l = X.shape[0]
        return (2 / l) * np.dot(X.T, (np.dot(X, self.w) - y))

    def update_weights(self, grad):
        self.w -= self.alpha * grad


class MomentumGD(LinearRegressionVectorized):
    def __init__(self, epsilon=1e-4, max_steps=2000, w0=None, alpha=1e-2, batch_size=None, gamma=0.9):
        super().__init__(epsilon, max_steps, w0, alpha, batch_size)
        # coef
        self.gamma = gamma
        self.v = np.zeros_like(self.w0)

    def update_weights(self, grad):
        self.v = self.gamma * self.v + self.alpha * grad
        self.w -= self.v


class AdagradGD(LinearRegressionVectorized):
    def __init__(self, epsilon=1e-4, max_steps=2000, w0=None, alpha=1e-2, batch_size=None, eps=1e-8):
        super().__init__(epsilon, max_steps, w0, alpha, batch_size)
        self.eps = eps
        # squares of GD
        self.g = np.zeros_like(self.w0)

    def update_weights(self, grad):
        self.g += grad ** 2
        self.w -= self.alpha * grad / (np.sqrt(self.g) + self.eps)


class AdamGD(LinearRegressionVectorized):
    def __init__(self, epsilon=1e-4, max_steps=2000, w0=None, alpha=1e-2, batch_size=None, beta1=0.9, beta2=0.999,
                 eps=1e-8):
        super().__init__(epsilon, max_steps, w0, alpha, batch_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # 1st m
        self.m = np.zeros_like(self.w0)
        # 2nd m
        self.v = np.zeros_like(self.w0)
        self.t = 0

    def update_weights(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.w -= self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSPropGD(LinearRegressionVectorized):
    def __init__(self, epsilon=1e-4, max_steps=2000, w0=None, alpha=1e-2, batch_size=None, gamma=0.9, eps=1e-8):
        super().__init__(epsilon, max_steps, w0, alpha, batch_size)
        # coef
        self.gamma = gamma
        self.eps = eps
        # mov avg
        self.g = np.zeros_like(self.w0)

    def update_weights(self, grad):
        self.g = self.gamma * self.g + (1 - self.gamma) * (grad ** 2)
        self.w -= self.alpha * grad / (np.sqrt(self.g) + self.eps)


# gen syn data
np.random.seed(1)
n_features = 2
n_objects = 2500
w_true = np.random.normal(0, 0.1, size=(n_features,))
X = np.random.uniform(-5, 5, (n_objects, n_features))
y = np.dot(X, w_true) + np.random.normal(0, 1, (n_objects))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. fix params
epsilon = 1e-6
max_steps = 5000
alpha = 1e-3
w0 = np.random.uniform(-2, 2, (n_features,))

optimizers = {
    'Momentum': MomentumGD(epsilon=epsilon, max_steps=max_steps, w0=w0, alpha=alpha),
    'Adagrad': AdagradGD(epsilon=epsilon, max_steps=max_steps, w0=w0, alpha=alpha),
    'Adam': AdamGD(epsilon=epsilon, max_steps=max_steps, w0=w0, alpha=alpha),
    'RMSProp': RMSPropGD(epsilon=epsilon, max_steps=max_steps, w0=w0, alpha=alpha)
}

# 2. train time
training_times = {}
for name, opt in optimizers.items():
    start_time = time.time()
    opt.fit(X_train, y_train)
    end_time = time.time()
    training_times[name] = end_time - start_time
    print(f"Time {name}: {training_times[name]} sec")
    plot_gd(X_train, y_train, opt.w_history, title=f'trajectory GD for {name}')

# 3. MSE on train epochs
plt.figure(figsize=(10, 6))
for name, opt in optimizers.items():
    plt.plot(opt.loss_history, label=name)
plt.title('MSE на train от эпох')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# 4. error graph
ratios = {}
for name, opt in optimizers.items():
    train_mse = mean_squared_error(y_train, opt.predict(X_train))
    test_mse = mean_squared_error(y_test, opt.predict(X_test))
    ratios[name] = train_mse / test_mse if test_mse != 0 else 0

plt.figure(figsize=(8, 5))
plt.bar(ratios.keys(), ratios.values())
plt.title('MSE train / MSE test')
plt.ylabel('Value')
plt.grid(True)
plt.show()

strategies = {
    'StochBatch': 1,
    'MiniBatch': 32
}

plt.figure(figsize=(10, 6))
for strat_name, batch_size in strategies.items():
    opt = AdamGD(epsilon=epsilon, max_steps=max_steps, w0=w0, alpha=alpha, batch_size=batch_size)
    opt.fit(X_train, y_train)
    plt.plot(opt.loss_history, label=strat_name)
plt.title('MSE train samples on epochs for different batches strategies')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# 6.
batch_sizes = [1, 4, 16, 64, 256, 512]
batch_labels = ['1', '4', '16', '64', '256', 'Full\n(512)']

optimizers_classes = {
    'Momentum': MomentumGD,
    'Adagrad': AdagradGD,
    'Adam': AdamGD,
    'RMSProp': RMSPropGD
}

plt.figure(figsize=(14, 8))

for opt_name, opt_class in optimizers_classes.items():
    final_test_errors = []

    for batch_size in batch_sizes:
        actual_batch_size = None if batch_size == 512 else batch_size

        current_alpha = alpha
        if opt_name == 'Adagrad':
            current_alpha = 1e-3

        opt = opt_class(epsilon=epsilon, max_steps=max_steps, w0=w0.copy(),
                        alpha=current_alpha, batch_size=actual_batch_size)
        opt.fit(X_train, y_train)

        test_error = mean_squared_error(y_test, opt.predict(X_test))
        final_test_errors.append(test_error)

        print(f"{opt_name}, batch_size={batch_size}, test_error={test_error:.6f}")

    plt.plot(batch_labels, final_test_errors,
             marker='o', linewidth=2, markersize=8, label=opt_name)

plt.title('MSE on optimizers', fontsize=14)
plt.xlabel('Batch size', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.ylim(bottom=0, top=2)

plt.tight_layout()
plt.show()
