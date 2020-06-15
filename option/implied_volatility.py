from europe_option_price import european_option_pricing
from america_option_price import american_option_pricing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class implied_vol:

    def __init__(self, f, s0, K, r, T, option_type, call, max_iter):
        """
        :param f: 期权价格
        :param s0: 标的资产价格
        :param K: 行权价
        :param r: 无风险利率
        :param T: 到期期限，单位年
        :param option_type: 期权类型：1-欧式，2-美式
        :param call: 是否是看涨期权
        :param max_iter: 最大迭代次数
        """
        self.f = f
        self.s0 = s0
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type
        self.call = call
        self.max_iter = max_iter

    def bisection(self, low, up, threshold):
        """
        Desc: 二分法计算隐含波动率(由于波动率和期权价格单调正相关，因此可以用二分法计算隐含波动率)
        :param low: 最低波动率，默认为0
        :param up: 最高波动率，默认为1
        :param threshold: 真实期权价格和按猜测波动率计算的期权价格之差的绝对值占真实期权价格比例上限
        """
        # (1) 确定volatility = up时的期权价格大于真实价格
        if self.option_type == 1:
            europe_opt = european_option_pricing(r=self.r, sigma=up, T=self.T, K=self.K, s0=self.s0,
                                                 call=self.call)
            price_up = europe_opt.bs_formula()
        elif self.option_type == 2:
            usa_opt = american_option_pricing(r=self.r, sigma=up, T=self.T, K=self.K, s0=self.s0,
                                              call=self.call)
            price_up = usa_opt.binary_tree(step_num=50)
        else:
            raise Exception('only support European option and American option')
        # (2) 递归地利用二分法计算隐含波动率
        if price_up >= self.f:
            per = 1
            count = 0
            guess_sigma = (low + up) / 2
            while per > threshold and count < self.max_iter:
                count += 1
                guess_sigma = (low + up) / 2
                if self.option_type == 1:
                    europe_opt = european_option_pricing(r=self.r, sigma=guess_sigma, T=self.T, K=self.K, s0=self.s0,
                                                         call=self.call)
                    guess_price = europe_opt.bs_formula()
                elif self.option_type == 2:
                    usa_opt = american_option_pricing(r=self.r, sigma=guess_sigma, T=self.T, K=self.K, s0=self.s0,
                                                      call=self.call)
                    guess_price = usa_opt.binary_tree(step_num=50)
                else:
                    raise Exception('only support European option and American option')

                if self.f < guess_price:
                    up = guess_sigma
                else:
                    low = guess_sigma
                per = np.abs(guess_price - self.f)/self.f
            result = {'implied_sigma': guess_sigma, 'iter_times': count, 'error_rate': per}
        else:
            result = self.bisection(low=low, up=2*up, threshold=threshold)
        return result

if __name__=='__main__':
    df = {'K': [], 'call_imply_vol': [], 'put_imply_vol': []}
    for k, c, p in zip(np.arange(3.1, 4.7, 0.1), [0.9179, 0.819, 0.7189, 0.6193, 0.5186, 0.4209, 0.3229, 0.226, 0.1376,
                                                  0.0642, 0.022, 0.0054, 0.0017, 0.001, 0.0008, 0.0005],
                       [0.0005, 0.0007, 0.0007, 0.0009, 0.0015, 0.0019, 0.0037, 0.0074, 0.0179, 0.0452, 0.1029, 0.1871,
                        0.2806, 0.379, 0.4792, 0.5802]):
        imply_vol1 = implied_vol(f=c, s0=4.04, K=k, r=2.0291/100, T=14 / 360, option_type=1, call=True, max_iter=10**2)
        imply_vol2 = implied_vol(f=p, s0=4.04, K=k, r=2.0291 / 100, T=14 / 360, option_type=1, call=False,
                                 max_iter=10 ** 2)
        result1 = imply_vol1.bisection(low=0, up=1, threshold=0.001)
        result2 = imply_vol2.bisection(low=0, up=1, threshold=0.001)
        df['K'].append(k)
        df['call_imply_vol'].append(result1['implied_sigma'])
        df['put_imply_vol'].append(result2['implied_sigma'])
    df = pd.DataFrame(df).set_index('K')
    plt.plot(df)
    plt.xlabel('行权价')
    plt.ylabel('隐含波动率')
    plt.title('6月到期的沪深300ETF期权的隐含波动率')
    plt.legend(['看涨期权', '看跌期权'])
