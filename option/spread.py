import numpy as np
import pandas as pd
from collections import Iterable
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
from option.fundamental_tools.european_option_pricing import european_option_pricing
from option.fundamental_tools.american_option_pricing import american_option_pricing
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

    def init_data_container(self, option_num):
        """
        Desc: 初始化数据容器
        :param option_num: 价差组合的期权数量
        """
        portfolio_items = ['value', 'delta', 'gamma', 'vega', 'theta', 'rho']
        option_items = []
        for key in portfolio_items:
            for i in range(1, option_num + 1):
                option_items.append(key + str(i))
        option_result = {}
        portfolio_result = {}
        for i in option_items:
            option_result[i] = pd.DataFrame(index=self.s0, columns=self.sigma)
            option_result[i].rename_axis(index='underlying_price', columns='iv')
        for i in portfolio_items:
            portfolio_result[i] = pd.DataFrame(index=self.s0, columns=self.sigma)
            portfolio_result[i].rename_axis(index='underlying_price', columns='iv')

        return option_result, portfolio_result, option_items, portfolio_items

    def stradle(self, K1, K2, T, long):
        """
        Desc: 计算宽跨式(当K1=K2时是跨式)期权组合的价值及性质
        :param K1: 看涨期权的行权价
        :param K2: 看跌期权的行权价
        :param T: 到期期限（年）
        :param long: 多头还是空头
        """
        # (1) 初始化数据容器
        option_result, portfolio_result, option_items, portfolio_items = self.init_data_container(option_num=2)
        # (2) 进行具体计算
        sign = 1 if long else -1
        for underlying_price in self.s0:
            for iv in self.sigma:
                # (2.1) 计算 call & put 的期权价格
                call = Greeks(f=None, s0=underlying_price, K=K1, T=T, r=self.r, call=1, option_type=1, sigma=iv)
                put = Greeks(f=None, s0=underlying_price, K=K2, T=T, r=self.r, call=0, option_type=1, sigma=iv)
                option_result['value1'].loc[underlying_price, iv] = call.f * sign
                option_result['value2'].loc[underlying_price, iv] = put.f * sign
                # (2.2) 计算 call & put 的希腊值
                option_greeks = [call.delta(), put.delta(), call.gamma(), put.gamma(), call.vega(), put.vega(),
                                 call.theta(), put.theta(), call.rho(), put.rho()]
                for key, value in zip(option_items[2:], option_greeks):
                    option_result[key].loc[underlying_price, iv] = value * sign
                # (2.3) 计算组合的价值和希腊值
                for key in portfolio_items:
                    portfolio_result[key].loc[underlying_price, iv] = option_result[key + '1'].loc[underlying_price, iv] + \
                                                                      option_result[key + '2'].loc[underlying_price, iv]
        return option_result, portfolio_result

    def bull_spread(self, K1, K2, T, call):
        """
        Desc: 计算牛市价差期权组合的价值和希腊值
        :param K1: 低执行价
        :param K2: 高执行价
        :param T: 到期期限
        :param call: 利用看涨期权合成还是看跌期权合成（1：long K1 short K2, 0：long K1 short K2）
        """
        # (1) 初始化数据容器
        option_result, portfolio_result, option_items, portfolio_items = self.init_data_container(option_num=2)
        # (2) 计算价差组合的性质
        for underlying_price in self.s0:
            for iv in self.sigma:
                # (2.1) 初始化期权对象
                opt1 = Greeks(f=None, s0=underlying_price, K=K1, T=T, r=self.r, call=call, option_type=1, sigma=iv)
                opt2 = Greeks(f=None, s0=underlying_price, K=K2, T=T, r=self.r, call=call, option_type=1, sigma=iv)
                # (2.2) 计算两期权的价值和希腊值
                the_option_result = [opt1.f, opt2.f, opt1.delta(), opt2.delta(), opt1.gamma(), opt2.gamma(),
                                     opt1.vega(), opt2.vega(), opt1.theta(), opt2.theta(), opt1.rho(), opt2.rho()]
                for key, value in zip(option_items, the_option_result):
                    option_result[key].loc[underlying_price, iv] = value
                # (2.3) 计算组合的价值和希腊值
                for key in portfolio_items:
                    portfolio_result[key].loc[underlying_price, iv] = option_result[key + '1'].loc[underlying_price, iv] - \
                                                                      option_result[key + '2'].loc[underlying_price, iv]
        return option_result, portfolio_result

    def bear_sread(self, K1, K2, T, call):
        """
        Desc: 计算熊市价差组合的价值和希腊值
        :param K1: 低执行价
        :param K2: 高执行价
        :param T: 到期期限
        :param call: 利用看涨期权合成还是看跌期权合成（1：long K2 short K1, 0：long K2 short K1）
        """
        # (1) 初始化数据容器
        option_result, portfolio_result, option_items, portfolio_items = self.init_data_container(option_num=2)
        # (2) 计算价差组合的性质
        for underlying_price in self.s0:
            for iv in self.sigma:
                # (2.1) 初始化期权对象
                opt1 = Greeks(f=None, s0=underlying_price, K=K1, T=T, r=self.r, call=call, option_type=1, sigma=iv)
                opt2 = Greeks(f=None, s0=underlying_price, K=K2, T=T, r=self.r, call=call, option_type=1, sigma=iv)
                # (2.2) 计算两期权的价值和希腊值
                the_option_result = [opt1.f, opt2.f, opt1.delta(), opt2.delta(), opt1.gamma(), opt2.gamma(),
                                     opt1.vega(), opt2.vega(), opt1.theta(), opt2.theta(), opt1.rho(), opt2.rho()]
                for key, value in zip(option_items, the_option_result):
                    option_result[key].loc[underlying_price, iv] = value
                # (2.3) 计算组合的价值和希腊值
                for key in portfolio_items:
                    portfolio_result[key].loc[underlying_price, iv] = - option_result[key + '1'].loc[underlying_price, iv] + \
                                                                      option_result[key + '2'].loc[underlying_price, iv]
        return option_result, portfolio_result

    def butterfly(self, K1, K2, K3, T, long_atm, call):
        """
        Desc: 碟式期权组合的价值和希腊值
        :param K1: 最小执行价
        :param K2: 中间执行价（一般是平直）
        :param K3: 最大执行价
        :param T: 到期期限
        :param long_atm: 是否多平值
        :param call: 操作call还是put
        """
        # (1) 初始化数据容器
        option_result, portfolio_result, option_items, portfolio_items = self.init_data_container(option_num=3)
        # (2) 计算各期权以及组合的价值和希腊值
        for underlying_price in self.s0:
            for iv in self.sigma:
                # (2.1) 初始化期权对象
                opt1 = Greeks(f=None, s0=underlying_price, K=K1, T=T, r=self.r, call=call, option_type=1, sigma=iv)
                opt2 = Greeks(f=None, s0=underlying_price, K=K2, T=T, r=self.r, call=call, option_type=1, sigma=iv)
                opt3 = Greeks(f=None, s0=underlying_price, K=K3, T=T, r=self.r, call=call, option_type=1, sigma=iv)
                # (2.2) 计算单个期权的价值和希腊值
                the_option_result = [opt1.f, opt2.f, opt3.f, opt1.delta(), opt2.delta(), opt3.delta(),
                                     opt1.gamma(), opt2.gamma(), opt3.gamma(), opt1.vega(), opt2.vega(), opt3.vega(),
                                     opt1.theta(), opt2.theta(), opt3.theta(), opt1.rho(), opt2.rho(), opt3.rho()]
                for key, value in zip(option_items, the_option_result):
                    option_result[key].loc[underlying_price, iv] = value
                # (2.3) 计算碟式组合的价值和希腊值
                for key in portfolio_items:
                    if long_atm:
                        portfolio_result[key].loc[underlying_price, iv] = - option_result[key + '1'].loc[underlying_price, iv] + \
                                                                          2 * option_result[key + '2'].loc[underlying_price, iv] - \
                                                                          option_result[key + '3'].loc[underlying_price, iv]
                    else:
                        portfolio_result[key].loc[underlying_price, iv] = option_result[key + '1'].loc[underlying_price, iv] - \
                                                                          2 * option_result[key + '2'].loc[underlying_price, iv] + \
                                                                          option_result[key + '3'].loc[underlying_price, iv]
        return option_result, portfolio_result

    def canlender(self, K, T1, T2, long_gamma, call):
        """
        Desc: 计算日历价差组合的价值和希腊值
        :param K: 执行价
        :param T1: 近月到期期限
        :param T2: 远月到期期限
        :param long_gamma: 是否long gamma(short vega)，即是否买近卖远
        :param call: 是否用看涨期权
        """
        # (1) 初始化数据容器
        option_result, portfolio_result, option_items, portfolio_items = self.init_data_container(option_num=2)
        # (2) 计算各期权以及组合的价值和希腊值
        for underlying_price in self.s0:
            for iv in self.sigma:
                # (2.1) 初始化期权对象
                opt1 = Greeks(f=None, s0=underlying_price, K=K, T=T1, r=self.r, call=call, option_type=1, sigma=iv)
                opt2 = Greeks(f=None, s0=underlying_price, K=K, T=T2, r=self.r, call=call, option_type=1, sigma=iv)
                # (2.2) 计算各期权的价值和希腊值
                the_option_result = [opt1.f, opt2.f, opt1.delta(), opt2.delta(), opt1.gamma(), opt2.gamma(),
                                     opt1.vega(), opt2.vega(), opt1.theta(), opt2.theta(), opt1.rho(), opt2.rho()]
                for key, value in zip(option_items, the_option_result):
                    option_result[key].loc[underlying_price, iv] = value
                # (2.3) 计算日历价差组合的价值和希腊值
                    for key in portfolio_items:
                        if long_gamma:
                            portfolio_result[key].loc[underlying_price, iv] = option_result[key + '1'].loc[underlying_price, iv] - \
                                                                              option_result[key + '2'].loc[underlying_price, iv]
                        else:
                            portfolio_result[key].loc[underlying_price, iv] = - option_result[key + '1'].loc[underlying_price, iv] + \
                                                                              option_result[key + '2'].loc[underlying_price, iv]
        return option_result, portfolio_result

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
    spread = spread_option(spread_type='stradle', s0=np.arange(7, 13.1, 0.1), r=0.03,
                            option_type=1, sigma=np.arange(0.1, 0.3, 0.01))
    # option, portfolio = spread.stradle(K1=9, K2=11, T=1/365, long=0)
    # option, portfolio = spread.bull_spread(K1=9, K2=11, T=30/365, call=1)
    # option, portfolio = spread.bear_sread(K1=9, K2=11, T=30/365, call=0)
    # option, portfolio = spread.butterfly(K1=9, K2=10, K3=11, T=30/365, long_atm=1, call=1)
    option, portfolio = spread.canlender(K=10, T1=1/365, T2=30/365, long_gamma=1, call=1)
    for key in ['value', 'delta', 'gamma', 'vega', 'theta', 'rho']:
        plot_price_iv(df=portfolio[key], name=key, spread_type='canlendar spread')

