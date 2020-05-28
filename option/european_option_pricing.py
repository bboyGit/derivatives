import numpy as np
from numpy.random import normal
from scipy.stats import norm

class european_option_pricing:
    """
    Desc: 此类用于实现各种方式的欧式期权定价
    """
    def __init__(self, r, sigma, T, K, s0, call):
        """
        :param r: 无风险利率
        :param sigma: 标的资产波动率
        :param T: 期权期限
        :param K: 行权价
        :param s0: 标的资产当前价格
        :param call: 是否是看涨期权
        """
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.s0 = s0
        self.call = call

    def bs_formula(self):
        """
        Desc: 由BS公式计算欧式期权价格
        """
        d1 = (np.log(self.s0/self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.s0/self.K) + (self.r - self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        if self.call:
            f0 = self.s0 * norm.cdf(d1) - \
                 self.K * np.exp(-self.r * self.T) * norm.cdf(d2)                      # c = s0*N(d1) - Ke^{-rT} * N(d2)
        else:
            f0 = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - \
                 self.s0 * norm.cdf(-d1)                                           # p = Ke^{-rT} * N(-d2) - s0 * N(-d1)
        return f0

    def monte_carlo(self, log, path_num, step_num):
        """
        Desc: 风险中性假设下利用蒙特卡洛方法计算欧式期权价格
        :param log: 是否根据标的资产对数来模拟路径。
            True: 假设标的资产价格服从对数正态分布[logSt ~ N(logS0 + (r-sigma^2/2)t, sigma^2 *t)]并以此计算sT。
            False: 假设标的资产价格服从几何布朗运动[ds = r*s*dt + sigma*s*dz]并以此计算sT。
        :param path_num: 蒙特卡洛生成的路径数量
        :param step_num: 每条价格路径的步数
        """
        # (1) 风险中性下模拟标的资产价格路径并计算到期时各路径下的价格
        if log:
            # 生成1000条价格路径
            log_sT = np.log(self.s0) + (self.r - self.sigma**2/2) * self.T + \
                     self.sigma * normal(size=path_num) * np.sqrt(self.T)
            sT = np.exp(log_sT)
        else:
            # 生成path_num条价格路径，每条路径走1000步
            sT = []
            for i in range(path_num):
                simu_ret = self.r * self.T/step_num + self.sigma * normal(size=step_num) * np.sqrt(self.T/step_num)     # ds/s = r * dt + sigma * dz
                cum_ret = (1 + simu_ret).prod()
                sT.append(self.s0 * cum_ret)
        # (2) 基于模拟的到期日标的资产价格计算期权价格
        if self.call:
            fT = np.array([max(0, s - self.K) for s in sT])                   # 看涨期权到期时各路径下的价格
        else:
            fT = np.array([max(0, self.K - s) for s in sT])                   # 看跌期权到期时各路径下的价格
        f0 = np.mean(fT * np.exp(-self.r * self.T))
        return f0

    def binary_tree(self, step_num):
        """
        Desc: 利用二叉树，从期权到期时的叶节点倒推地计算前面各个节点的期权价值，直到0时刻即得出期权价格。
        Note: 欧式期权只需要倒推地计算每一层各个节点期权的隐含价值但美式期权还要比较每个节点期权隐含价值与直接行权价值
              的相对大小，并最终取价值大的那一个作为该节点的期权价值。
        :param step_num:  二叉树步数
        """
        # (1) 计算 u, d, p, q and dt
        dt = self.T/step_num
        u = np.exp(self.sigma * np.sqrt(dt))
        d = np.exp(-self.sigma * np.sqrt(dt))
        p = (np.exp(self.r * dt) - d)/(u - d)
        q = 1 - p
        # (2) 构造二叉树价格路径的矩阵(n步二叉树需要一个n+1*n+1矩阵来装价格路径)
        s = np.array([[0] * (1 + step_num)] * (1 + step_num), dtype=float)
        for i in range(0, 1 + step_num):                           # 遍历每一步
            for j in range(i+1):                                   # 遍历每一步中标的资产价格的每种可能
                s[j, i] = self.s0 * u**(i-j) * d**j
        # (3) 构造二叉树下期权内涵价值路径的矩阵
        f = np.array([[0] * (1 + step_num)] * (1 + step_num), dtype=float)
        for i in range(step_num, -1, -1):
            # 倒推地计算期权价值
            if i == step_num:
                # 最后一步的期权价格
                if self.call:
                    f[:, i] = [max(sT - self.K, 0) for sT in s[:, i]]
                else:
                    f[:, i] = [max(self.K - sT, 0) for sT in s[:, i]]
            else:
                # 其他时候的期权价格(因为基于风险中性，所以第t时刻的期权价值等于t+1时刻期权价值的贴现)
                for j in range(i+1):
                    f[j, i] = np.exp(-self.r * dt) * (p * f[j, i+1] + q * f[j+1, i+1])
        f0 = f[0, 0]
        return f0

if __name__=='__main__':
    europe_opt_price = european_option_pricing(r=0.05, sigma=0.2, T=1, K=10, s0=10, call=True)
    bs_price = europe_opt_price.bs_formula()
    monte_carlo_price = europe_opt_price.monte_carlo(log=True, path_num=10**6, step_num=1000)
    binary_tree_price = europe_opt_price.binary_tree(step_num=100)
    print('BS公式下的期权价格：', bs_price)
    print('蒙特卡洛下的期权价格：', monte_carlo_price)
    print('二叉树下的期权价格：', binary_tree_price)
    
