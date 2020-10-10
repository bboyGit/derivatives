import numpy as np
import pandas as pd
from collections import Iterable
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
from option.european_option_pricing import european_option_pricing
from option.american_option_pricing import american_option_pricing
from option.fundamental_tools.greeks import Greeks

class spread_option:
    """
    Desc: 本类用于构造各种经典的价差期权。将在既定标的价格、波动率、到期期限、无风险利率的情况下价算各价差期权组合的价值和希腊值。
    """
    def __init__(self, spread_type, s0, r, option_type, sigma):
        """
        :param spread_type: 价差组合类型
        :param s0: 标的资产价格(must be iterable)
        :param r: 无风险利率
        :param option_type: 期权类型. 1-欧式，2-美式
        :param sigma: 波动率(must be iterable)
        """
        # (1) 异常处理
        if not isinstance(s0, Iterable):
            TypeError('argument s0 must be Iterable')
        if not isinstance(sigma, Iterable):
            TypeError('argument sigma must be Iterable')
        # (2) 获取该类中的全局变量
        self.spread_type = spread_type
        self.s0 = s0
        self.r = r
        self.option_type = option_type
        self.sigma = sigma

    def stradle(self, K, T, long):
        """
        Desc: 计算跨式期权的价值和希腊值
        :param K: 执行价（默认平值）
        :param T: 到期期限(单位：年)
        :param long: long vega/gamma or short vega/gamma ?
        """
        # (1) 初始化数据容器
        option_items = ['call_value', 'put_value', 'call_delta', 'put_delta', 'call_gamma', 'put_gamma',
                        'call_vega', 'put_vega', 'call_theta', 'put_theta', 'call_rho', 'put_rho']
        portfolio_items = ['value', 'delta', 'gamma', 'vega', 'theta', 'rho']
        option_result = {}
        portfolio_result = {}
        for i in option_items:
            option_result[i] = pd.DataFrame(index=self.s0, columns=self.sigma)
            option_result[i].rename_axis(index='underlying_price', columns='iv')
        for i in portfolio_items:
            portfolio_result[i] = pd.DataFrame(index=self.s0, columns=self.sigma)
            portfolio_result[i].rename_axis(index='underlying_price', columns='iv')
        # (2) 进行具体计算
        sign = 1 if long else -1
        for underlying_price in self.s0:
            for iv in self.sigma:
                # (2.1) 计算 call & put 的期权价格
                call = Greeks(f=None, s0=underlying_price, K=K, T=T, r=self.r, call=1, option_type=1, sigma=iv)
                put = Greeks(f=None, s0=underlying_price, K=K, T=T, r=self.r, call=0, option_type=1, sigma=iv)
                option_result['call_value'].loc[underlying_price, iv] = call.f * sign
                option_result['put_value'].loc[underlying_price, iv] = put.f * sign
                # (2.2) 计算 call & put 的希腊值
                option_greeks = [call.delta(), put.delta(), call.gamma(), put.gamma(), call.vega(), put.vega(),
                                 call.theta(), put.theta(), call.rho(), put.rho()]
                for key, value in zip(option_items[2:], option_greeks):
                    option_result[key].loc[underlying_price, iv] = value * sign
                # (2.3) 计算组合的价值和希腊值
                for key in portfolio_items:
                    portfolio_result[key].loc[underlying_price, iv] = option_result['call_' + key].loc[underlying_price, iv] + \
                                                                      option_result['put_' + key].loc[underlying_price, iv]
        return option_result, portfolio_result

    def strangle(self, K1, K2, T, long):
        pass

    def bull_spread(self, K1, K2, T, call):
        pass

    def bear_sread(self, K1, K2, T, call):
        pass


def plot_price_iv(df, name, spread_type):
    """
    Desc: 绘制三维图，x-标的价格、y-隐含波动率
    :param df: 待绘制的数据
    :param name: 待绘制的名称
    :param spread_type: 价差类型
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z = df.values.astype(float)
    y = df.index.to_numpy().reshape(len(df), 1)
    y = np.concatenate([y for i in range(df.shape[1])], axis=1)
    x = df.columns.to_numpy().reshape(1, df.shape[1])
    x = np.concatenate([x for i in range(len(df))])
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2)
    plt.xlabel('iv', {'size': 15})
    plt.ylabel('underlying_price', {'size': 15})
    plt.title(spread_type + ' ' + name, {'size': 15})
    plt.show()

if __name__=="__main__":
    stradle = spread_option(spread_type='stradle', s0=np.arange(9, 11, 0.1), r=0.03,
                            option_type=1, sigma=np.arange(0.1, 0.3, 0.01))
    option, portfolio = stradle.stradle(K=10, T=0.05, long=1)
    for key in ['value', 'delta', 'gamma', 'vega', 'theta', 'rho']:
        plot_price_iv(df=portfolio[key], name=key, spread_type='stradle')

