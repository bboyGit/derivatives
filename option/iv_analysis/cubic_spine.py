
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from option.iv_analysis.optimize import optimize
import matplotlib.pyplot as plt

class calibrate_cs:
    """
    Desc: 用三次样条拟合波动率曲线
    (1) 无套利检查：检查输入的隐含波动率是否满足无碟式套利并剔除可套利的点
    (2) 补点：填补被剔除的点
    (3) 构造 cubic regression spline 所需的特征矩阵 X
    (4) 拟合三次样条函数
    """
    def __init__(self, x, y, knots, natural, smooth_index):
        """
        :param x: A 1-D vector
        :param y: A columns vector
        :param knots: Optional. If it's an int, it represents the number of knots. If it's an 1-D array, then it's cutting points.
        :param natural: True-拟合自然三次样条 or False-拟合基础版本的三次样条
        :param smooth_index: 三次样条拟合的平滑参数
        """
        self.x = x
        self.y = y
        self.knots = knots
        self.natural = natural
        self.smooth_index = smooth_index

    def check_butterfly_arbitrage(self):
        """
        Desc: 检查输入的iv是否有碟式套利机会
        """
        
        return

    def check_input_data(self):
        """
        Desc: 检查输入样本x和y的数据类型数据量是否一致
        """
        if not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray):
            raise TypeError("Both x and y must be array.")
        if self.x.ndim != 1:
            self.x = self.x.ravel()
        if self.x.shape[0] != self.y.shape[0]:
            raise Exception("The number of observations of x and y must be the same.")

    def form_spline(self, power=3, const=True):
        """
        Desc: 形成三次样条回归的节点序列和特征矩阵X
        function form:
            基础三次样条: f(x) = a + a1*x + a2*x^2 + a3*x^3 + sum of g(x, K_i)
                        g(x, K_i) = (x - K_i)^power if x > K_i else 0
            自然三次样条: f(x) = a + a1*x + sum of g(x, K_i)
                       x > 最大 cutting point 时 g(x, K_i) = x - K_i, 否则 g(x, K_i) = (x - K_i)^power if x > K_i else 0
        properties: Function is continuous in each cutting point in its 0, 1, 2...... knot - 1 derivatives.
        Return: X, knot
        """
        # (1) Deal exception
        self.check_input_data()
        # (2) Construct X
        X = {}
        if isinstance(self.knots, np.ndarray) or isinstance(self.knots, int):
            # （2.1）构建X中 x 或 x, x^2, x^3 那部分的数据
            if self.natural:
                X[0] = self.x
            else:
                for i in range(power):
                    X[i] = self.x ** (i + 1)
            # (2.2) 计算出cutting points序列
            if isinstance(self.knots, np.ndarray):
                knot = self.knots.copy()                                     # knot represents cutting points for spline
            else:
                # knot represents the number of cutting points
                if self.knots > len(self.x):
                    raise Exception("The number of knot can not exceed the number of observations")
                x_max, x_min = self.x.max(), self.x.min()
                knot = x_min + np.linspace(0, x_max - x_min, self.knots + 2)[1: -1]             # 等距离设置cutting points
            # (2.3) 构建X中 g(x, K_i)那部分的数据
            if self.natural:
                # 自然三次样条
                for idx, value in enumerate(knot):
                    if idx < len(knot) - 1:
                        X[idx + power] = [(i - value) ** power if i > value else 0 for i in self.x]
                    else:
                        X[idx + power] = [(i - value) if i > value else 0 for i in self.x]
            else:
                # 基础三次样条
                for idx, value in enumerate(knot):
                    X[idx + power] = [(i - value)**power if i > value else 0 for i in self.x]
            X = pd.DataFrame(X).values
            if const:
                c = np.array([[1] * self.x.shape[0]]).T
                X = np.concatenate([c, X], axis=1)
            return X, knot
        else:
            raise TypeError('knots must be an int or 1d array')

    def fit(self, X):
        """
        Desc: 拟合三次样条曲线
        :param X: output of form_spline
        loss function: sum((y - f(x))^2) + lambda * sum(f''(x)^2); lambda就是
        Return: 系数和拟合值
        """
        if self.smooth_index == 0:
            # 平滑系数等于0时直接用最小二乘的解析解求解三次样条函数的系数
            mat1 = np.linalg.inv(X.T @ X) @ X.T
            beta = mat1 @ self.y
            y_hat = X @ beta
        elif self.smooth_index < 0:
            raise ValueError("the smoothing parameter must be non-negative")
        else:
            # 平滑系数不等于0时需要用数值方法求解三次样条函数的系数
            n = X.shape[1]
            def loss_func(beta):
                # Desc: 用于求解三次样条函数系数的损失函数
                beta = beta.reshape(len(beta), 1)
                optim = optimize()
                error_term = np.dot((self.y - X @ beta).T, self.y - X @ beta)[0, 0]        # 误差项
                second_derivative = optim.hess(lambda x: np.sum(X @ x), beta, delta=10**(-4)).diagonal()
                second_derivative = second_derivative.reshape(len(second_derivative), 1)   # 二阶导的列向量
                penalty_term = (second_derivative.T @ second_derivative)[0, 0]             # 惩罚项
                result = error_term + self.smooth_index * penalty_term
                return result

            fits = minimize(fun=loss_func, x0=np.zeros(n), method='SLSQP',
                            tol=10 ** (-10), options={'maxiter': 100, 'disp': True})
            beta = fits.x
            beta = beta.reshape(len(beta), 1)
            y_hat = X @ beta
        return beta, y_hat

    def extrapolate(self, beta, knot, new_x):
        """
        Desc: 利用fit方法中已经得出系数beta对新的点new_x进行插值
        :param beta: fit中拟合出来的三次样条的系数
        :param knot: form_spline中给出来cutting_point序列
        :param new_x: 我们要extrapolate的点
        :return: 新的点的y值
        """
        new_y = []
        for x in new_x:
            if self.natural:
                # 自然三次样条
                y = beta[0, 0] + beta[1, 0] * x
                for idx, value in enumerate(knot):
                    if idx < len(knot) - 1:
                        item = (x - value) ** 3 if x > value else 0
                    else:
                        item = x - value if x > value else 0
                    y = y + beta[idx + 2, 0] * item
            else:
                # 基础三次样条
                y = beta[0, 0] + beta[1, 0] * x + beta[2, 0] * x**2 + beta[3, 0] * x**3
                for idx, value in enumerate(knot):
                    item = (x - value) ** 3 if x > value else 0
                    y = y + beta[idx + 4, 0] * item
            new_y.append(y)
        df = pd.DataFrame([new_x, new_y])
        return df

if __name__ == '__main__':
    strike = np.arange(5300, 6500, 100)
    strike = strike.reshape(len(strike), 1)
    md_iv = np.array([[0.123, 0.12, 0.1108, 0.1099, 0.1137, 0.1178, 0.1273,
                      0.1396, 0.1435, 0.1574, 0.1652, 0.178]]).T
    underlying = 5700
    moneyness = strike/underlying
    iv = [md_iv.ravel()]
    cs = calibrate_cs(moneyness, md_iv, knots=4, natural=True, smooth_index=0)
    X, knot = cs.form_spline()
    beta, y_hat = cs.fit(X)
    iv.append(y_hat.ravel())
    plt.scatter(moneyness, md_iv, s=5)
    plt.scatter(moneyness, y_hat, s=5)
    plt.legend(['md_iv', 'cs_natural_iv'])
    iv = pd.DataFrame(iv)

    df = cs.extrapolate(beta, knot, new_x=[0.9, 0.95, 1, 1.05, 1.1])
    # from scipy.interpolate import CubicSpline
    # strike = np.arange(5300, 6500, 100)
    # md_iv = pd.Series([0.123, 0.1136, 0.1108, 0.1099, 0.1137, 0.1178, 0.1273,
    #                    0.1396, 0.1435, 0.1574, 0.1652, 0.178],
    #                   index=strike)
    #     cs = CubicSpline(moneyness, md_iv)
    #     cali_iv = cs(moneyness)

    #    extra_moneyness = np.arange(0.9, 1.13, 0.005)
    #    extra_moneyness = np.concatenate([moneyness , extra_moneyness])
    #    extra_moneyness.sort()


    #
    #    calibrate_iv = pd.Series(cs(extra_moneyness), index=extra_moneyness)
    #    plt.plot(calibrate_iv, 'o')

