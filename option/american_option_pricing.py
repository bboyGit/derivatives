import numpy as np
from numpy.linalg import solve
from europe_option_price import european_option_pricing

class american_option_pricing:
    """
    Desc: 美式期权定价
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

    def binary_tree(self, step_num):
        """
        :param step_num: 二叉树步数
        Note：与欧式期权不同，美式可以提前行权。二叉树的每个节点期权既有一个通过后一步节点期权价值贴现得到的隐含价值，
        也有一个当期行权得到的收益。因此在计算期权在某节点处的价值时要用上述两者中最大的那一个，这也是与欧式期权定价的区别。
        """
        # (1) 计算 u, d, p, q and dt
        dt = self.T / step_num
        u = np.exp(self.sigma * np.sqrt(dt))
        d = np.exp(-self.sigma * np.sqrt(dt))
        p = (np.exp(self.r * dt) - d) / (u - d)
        q = 1 - p
        # (2) 构造二叉树价格路径的矩阵(n步二叉树需要一个n+1*n+1矩阵来装价格路径)
        s = np.array([[0] * (1 + step_num)] * (1 + step_num), dtype=float)
        for i in range(0, 1 + step_num):  # 遍历每一步
            for j in range(i + 1):  # 遍历每一步中标的资产价格的每种可能
                s[j, i] = self.s0 * u ** (i - j) * d ** j
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
                for j in range(i + 1):
                    hiden_value = np.exp(-self.r * dt) * (p * f[j, i + 1] + q * f[j + 1, i + 1])
                    if self.call:
                        execute_value = max(0, s[j, i] - self.K)
                    else:
                        execute_value = max(0, self.K - s[j, i])
                    f[j, i] = max(hiden_value, execute_value)                     # 取期权隐含价值和直接行权中更高的那个
        f0 = f[0, 0]
        return f0

    def control_variate(self, step_num):
        """
        Desc: 基于二叉树的控制变量法假设欧式期权误差与美式期权误差相同
        :param step_num: 二叉树步数
        Reference: J.C. Hull and A.White, The use of the control variate technique in option pricing,
                  Journal of financial and quantitative analysis
        """
        # (1) 计算给定参数下基于BS公式和二叉树的欧式期权价格
        europe_opt_price = european_option_pricing(r=self.r, sigma=self.sigma, T=self.T, K=self.K, s0=self.s0, call=self.call)
        bs = europe_opt_price.bs_formula()
        europe_binary_tree = europe_opt_price.binary_tree(step_num=step_num)
        # (2) 计算给定参数下二叉树的美式期权价格
        usa_binary_tree = self.binary_tree(step_num=step_num)
        # (3) 由控制变量技巧计算调整后的美式期权价格
        usa_price = usa_binary_tree + bs - europe_binary_tree
        return usa_price

    def finite_difference(self, price_step, time_step, s_max):
        """
        Desc: 利用有限差分法计算期权价格
        :param price_step: 在价格层面上分裂的步数(从0到S_max的间隔数量)
        :param time_step: 在时间层面上的分裂的步数(从0到T)
        :param s_max: 设置的最高标的资产价格
        """
        # (1) 计算时间delta和价格delta
        dt = self.T/time_step
        ds = s_max/price_step
        # (2) 计算f(i,j-1), f(i,j) and f(i,j+1) where j=0:(price_step-1) 的系数aj, bj and cj
        p_div = np.arange(0, price_step)                             # 0 至 price_step - 1
        a = (self.r * p_div - self.sigma**2 * p_div**2) * dt/2       # index 为 0 至 price_step - 1
        b = 1 + self.r * dt + self.sigma**2 * p_div**2 * dt
        c = - dt * (self.r * p_div + self.sigma**2 * p_div**2)
        # (3) 构建x轴为时间y轴为标的资产价格网格
        s_grid = np.array([[s] * (time_step + 1) for s in np.arange(0, s_max + ds, ds)])
        f_grid = np.zeros([price_step + 1, time_step + 1])       # 行的index:0 至 price_step ; 列的index: 0 至 time_step
        # (4) 计算边界条件下的期权价格
        if self.call:
            f_grid[:, time_step] = [max(0, df) for df in s_grid[:, time_step] - self.K]  # 到期时的期权价格
            f_grid[0, :] = 0                                                             # 标的价格等于0时的期权价格
            f_grid[price_step, :] = s_max - self.K                                       # 标的价格等于最大预设情况时的期权价格
        else:
            f_grid[:, time_step] = [max(0, df) for df in self.K - s_grid[:, time_step]]
            f_grid[0, :] = self.K
            f_grid[price_step, :] = 0
        # (5) 迭代地计算从T-1期到0期的期权价格
        for t in range(time_step-1, -1, -1):
            y = f_grid[1:-1, t+1].copy()
            y[0] = y[0] - a[0] * f_grid[0, t+1]
            y[price_step - 2] = y[price_step - 2] - c[-1] * f_grid[price_step, t+1]
            x1 = np.array([[b[0], c[0]] + [0] * (price_step - 3)])
            x2 = np.array([[0]*i + [a[i+1], b[i+1], c[i+1]] + [0]*(price_step - 4 - i) for i in range(price_step-3)])
            x3 = np.array([[0] * (price_step - 3) + [a[-1], b[-1]]])
            x = np.concatenate([x1, x2, x3])
            f_t = solve(x, y)
            # 与直接行权相比较，看谁的收益更大
            
            f_grid[1: price_step, t] = f_t
        # (6) 获取第0期标的价格等于s0的期权价格既是最终的期权价格
        f0 = f_grid[:, 0].copy()
        idx = np.argmin(np.abs(s_grid[:, 0] - self.s0))
        final_f0 = f0[idx]
        return final_f0

if __name__=='__main__':
    usa_opt_price = american_option_pricing(r=0.1, sigma=0.4, T=0.4167, K=50, s0=50, call=False)
    # (1) 二叉树美式期权定价
    # put = []
    # for i in range(2, 500, 3):
    #     put_fee = usa_opt_price.binary_tree(step_num=i)
    #     put.append(put_fee)
    # put = np.array(put)
    # import matplotlib.pyplot as plt
    # plt.plot(put)
    # (2) 基于控制变量技巧的美式期权定价
    # put_20 = usa_opt_price.control_variate(step_num=20)
    # put_32 = usa_opt_price.control_variate(step_num=32)
    # print('put[-1] - put[7] = {} and put[-1] - put_20 = {}'.format(put[-1] - put[7], put[-1] - put_20))
    # print('put[-1] - put[10] = {} and put[-1] - put_32 = {}'.format(put[-1] - put[10], put[-1] - put_32))
    # (3) 基于有限差分法的美式期权定价
    usa_opt_price.finite_difference(price_step=100, time_step=100, s_max=100)
    
