
import pandas as pd
import numpy as np

def normalize(x, method='normal'):
    """
    Desc: normalize features of any model
    :param x: A dataFrame or 2-D array
    :param method: str indicating the method to normalize x
    :return: dataframe or 2-D array
    """
    x = x.copy()
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if method == 'max_min':
        result = x.apply(lambda w: (w - w.min())/(w.max() - w.min()))
    elif method == 'normal':
        result = x.apply(lambda w: (w - w.mean())/w.std())
    return result

class optimize:

    def grad(self, f, x, delta):
        """
        Desc: Get gradient vector of function f in x0 \n
        :param f: A function
        :param x: A columns vector representing vector of x
        :param delta: A float indicating delta of x
        formula: df(x)/dx = (f(x+h) - f(x))/h 当h很小时，约等于导数
        Return: A vector containing partial derivatives of each elements
        """
        fun = f(x)
        nrow = x.shape[0]
        fun_h = np.ones([nrow, 1])
        for j in range(nrow):
            x1 = x.copy()
            x1[j, 0] = x1[j, 0] + delta
            fun_h[j, 0] = f(x1)
        grad_ = (fun_h - fun)/delta
        return grad_

    def hess(self, f, x, delta):
        """
        Desc: compute Hessian matrix
        formula: df(x, y)/dxdy = (f(x+h, y+h) - f(x+h, y) - f(x, y+h) + f(x, y))/h^2  当h很小时，约等于二阶导数
        """
        fun = f(x)
        nrow = x.shape[0]
        hess_ = np.ones([nrow, nrow])
        for i in range(nrow):
            for j in range(nrow):
                x1 = x.copy()
                x2 = x.copy()
                x3 = x.copy()
                x1[i, 0] = x1[i, 0] + delta
                x2[j, 0] = x2[j, 0] + delta
                if i == j:
                    x3[i, 0] = x3[i, 0] + 2*delta
                else:
                    x3[[i, j], 0] = x3[[i, j], 0] + delta
                f1 = f(x1)
                f2 = f(x2)
                f3 = f(x3)
                hess_[i, j] = (f3 - f2 - f1 + fun)/delta**2
        return hess_

    def grad_desc(self, func, x0, tol, max_iter, alpha, delta):
        """
        Desc: Execute Gradient descent algorithm to find local optimal solution of a given function \n
        :param func: A function whose return value is a scalar
        :param x0: initial guess(columns vector)
        :param tol: 根号下梯度向量平方和的最大可接受值
        :param max_iter: max iteration times
        :param alpha: A positive number (learning rate)
        Return: optimal solution
        """
        count = 0
        diff = 10
        x_now = x0.copy()
        # grad_list = []
        fun = []
        while count < max_iter and diff > tol:
            gradient = self.grad(f=func, x=x_now, delta=delta)          # Compute gradient of given function in point x_now
            diff = np.sqrt(np.sum(gradient**2))
            x_now = x_now - alpha * gradient                          # update point in minus gradient direction
            fun_now = func(x_now)
            fun.append(fun_now)
            # grad_list.append(gradient.ravel().tolist())
            count += 1
        # grad_list = pd.DataFrame(grad_list)
        success = True if count < max_iter else False
        result = {'x': x_now, 'iter_num': count, 'diff': diff, 'success': success}#, 'grad_list': grad_list}
        return result

    def newton_iter(self, func, x0, tol, max_iter, delta):
        """
        Desc: get optimize solution by newton iterative method
        """
        count = 0
        x_now = x0.copy()
        while count <= max_iter:
            gradient = self.grad(f=func, x=x_now, delta=delta)
            diff = np.sqrt(np.sum(gradient**2))
            if diff <= tol:
                break
            else:
                count += 1
                hessian_matrix = self.hess(f=func, x=x_now, delta=delta)
                x_now = x_now - np.linalg.inv(hessian_matrix) @ gradient
        success = True if count < max_iter else False
        result = {"x": x_now, 'iter_num': count, 'diff': diff, 'success': success}
        return result


if __name__ == '__main__':
    # Solve linear regression by gradient descent
    from sklearn.datasets import load_boston
    boston = load_boston()
    y = boston['target'][:100].reshape(100, 1)
    X = boston['data'][:100, :1]
    # X = normalize(X).values
    n = X.shape[1] + 1
    f = lambda x: np.sum((y - x[0] * np.exp(X/x[1]))**2)
    optim = optimize()
    # hess = optim.hess(f, np.array([[0.2, 0.3, 0.4]]).T, 10**(-5))
    # Gradient_solution = optim.grad_desc(f, np.array([[0.2, 0.3, 0.4]]).T, tol=10**(-6), max_iter=10**4, alpha=0.1, delta=10**(-5))   # 学习率太大导致梯度发散了
    Gradient_solution = optim.grad_desc(f, np.array([[0.3, 4]]).T, tol=10 ** (-3), max_iter=10 ** 5,
                                        alpha=10**(-3), delta=10**(-4))
    # Newton_solution = optim.newton_iter(f, np.array([[0.3, 4]]).T, tol=10**(-6), max_iter=10**4, delta=10**(-5))
    # import statsmodels.api as sm
    # sm.OLS(y, X).fit().params

