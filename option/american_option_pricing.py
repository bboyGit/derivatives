import numpy as np

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
                    f[j, i] =max(hiden_value, execute_value)                     # 取期权隐含价值和直接行权中更高的那个
        f0 = f[0, 0]
        return f0

if __name__=='__main__':
    usa_opt_price = american_option_pricing(r=0.1, sigma=0.4, T=0.4167, K=50, s0=50, call=False)
    put = []
    for i in range(2, 50, 3):
        put_fee = usa_opt_price.binary_tree(step_num=i)
        put.append(put_fee)
    put = np.array(put)
    import matplotlib.pyplot as plt
    plt.plot(put)
