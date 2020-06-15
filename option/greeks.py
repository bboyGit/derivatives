import numpy as np
import pandas as pd
from implied_volatility import implied_vol
from america_option_price import american_option_pricing
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class Greeks:
    """
    Desc: 计算欧式和美式期权多头的希腊值。欧式期权通过解析解计算，美式期权通过二叉树计算。
    """
    def __init__(self, f, s0, K, T, r, call, option_type, sigma):
        """
        :param f: 期权价格
        :param s0: 标的资产当前价格
        :param K: 行权价
        :param T: 到期期限
        :param r: 无风险利率
        :param call: 是否为看涨
        :param option_type: 1-欧式，2-美式
        :param sigma: 标的资产波动率
        """
        self.f = f
        self.s0 = s0
        self.K = K
        self.T = T
        self.r = r
        self.call = call
        self.option_type = option_type
        if pd.isnull(sigma):
            self.sigma = self.calcu_iv()
        else:
            self.sigma = sigma
        self.d1 = (np.log(self.s0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = (np.log(self.s0 / self.K) + (self.r - self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def calcu_iv(self):
        """
        Desc: 在初始化没有输入标的资产波动率时自行计算相应期权的隐含波动率
        """
        iv = implied_vol(f=self.f, s0=self.s0, K=self.K, r=self.r, T=self.T, option_type=self.option_type,
                         call=self.call, max_iter=10**3)
        sigma = iv.bisection(low=0, up=1, threshold=0.001)['implied_sigma']
        return sigma

    def delta(self):
        """
        Desc: 计算给定期权的Δ
        """
        if self.option_type == 1:
            if self.call:
                result = norm.cdf(self.d1)
            else:
                result = norm.cdf(self.d1) - 1
        elif self.option_type == 2:
            usa_opt_price = american_option_pricing(r=self.r, sigma=self.sigma, T=self.T, K=self.K,
                                                    s0=self.s0, call=self.call)
            f, s = usa_opt_price.binary_tree(step_num=100)
            result = (f[1, 0] - f[1, 1])/(s[1, 0] - s[1, 1])
        else:
            raise Exception('目前只支持欧式和美式')
        return result

    def theta(self):
        """
        Desc: 计算给定期权的Θ
        """
        if self.option_type == 1:
            if self.call:
                result = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2) - \
                         self.s0 * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
            else:
                result = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - \
                         self.s0 * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
        elif self.option_type == 2:
            usa_opt_price = american_option_pricing(r=self.r, sigma=self.sigma, T=self.T, K=self.K,
                                                    s0=self.s0, call=self.call)
            f, s = usa_opt_price.binary_tree(step_num=100)
            result = (f[1, 2] - f[0, 0])/(2 * self.T/100)
        else:
            raise Exception('目前只支持欧式和美式')
        return result

    def gamma(self):
        """
        Desc: 计算给定期权的Γ
        """
        Delta = self.delta()
        Theta = self.theta()
        result = 2 * (self.r * self.f - Delta * self.r * self.s0 - Theta)/(self.sigma**2 * self.s0**2)
        return result

    def vega(self):
        """
        Desc: 计算给定期权的Vega
        """
        if self.option_type == 1:
            result = self.s0 * np.sqrt(self.T) * norm.pdf(self.d1)
        elif self.option_type == 2:
            usa_opt_price = american_option_pricing(r=self.r, sigma=self.sigma * 1.01, T=self.T,
                                                    K=self.K, s0=self.s0, call=self.call)
            f, s = usa_opt_price.binary_tree(step_num=100)
            f1 = f[0, 0]
            result = (self.f - f1)/(self.sigma * 0.01)
        else:
            raise Exception('目前只支持欧式和美式期权')
        return result

    def rho(self):
        """
        Desc: 计算给定期权的rho(期权对无风险利率求导)
        """
        if self.option_type == 1:
            if self.call:
                result = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            else:
                result = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        elif self.option_type == 2:
            usa_opt_price = american_option_pricing(r=self.r * 1.01, sigma=self.sigma, T=self.T,
                                                    K=self.K, s0=self.s0, call=self.call)
            f, s = usa_opt_price.binary_tree(step_num=100)
            f1 = f[0, 0]
            result = (f1 - self.f)/(self.r * 0.01)
        else:
            raise Exception('只支持欧式和美式期权')
        return result


if __name__=='__main__':
    greek = Greeks(f=0.1376, s0=4.04, K=3.9, T=14/360, r=2.0291/100, call=True, option_type=1, sigma=None)
    print('Delta:', greek.delta())
    print('Gamma:', greek.gamma())
    print('Theta:', greek.theta())
    print('Vega:', greek.vega())
    print('Rho:', greek.rho())
