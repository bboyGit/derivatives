
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from option.iv_analysis.cubis_spine import calibrate_cs
from copy import deepcopy

class wing_model:

    def __init__(self, atm, ref, dc, dsm, uc, usm, ssr=1.0, vcr=0, scr=0, vr=None, sr=None, pc=None, cc=None):
        """
        :param atm: current underlying price
        :param ref: reference price of underlying
        :param dc: down cutoff
        :param dsm: down smoothing rate
        :param uc: up cutoff
        :param usm: up smoothing rate
        :param ssr: skew swimmingness rate, F = atm^ssr * ref^(1-ssr)
        :param vcr: volatility change rate, vc = vr - vcr * ssr * (atm - ref)/ref
        :param scr: slope change rate, sc = sr - scr * ssr * (atm - ref)/ref
        :param vr: volatility reference
        :param sr: volatility slope reference
        :param pc: put curvature
        :param cc: call curvature
        """
        self.atm = atm
        self.ref = ref
        if dc >= 0:
            raise ValueError('dc must be negative')
        else:
            self.dc = dc
        self.dsm = dsm
        self.uc = uc
        self.usm = usm
        self.ssr = ssr
        self.vcr = vcr
        self.scr = scr
        self.underlying = atm**ssr * ref**(1-ssr)
        self.vr = vr
        self.sr = sr
        self.pc = pc
        self.cc = cc

    def drop_cry_point(self, iv):
        """
        Desc: 用递归法把iv曲线的样本点中含有低高低(哭脸)形式的点去掉
        :param iv: DataFrame with columns of iv and ks
        """
        if len(iv) < 3:
            return iv
        else:
            iv = iv.sort_values('ln(k/s)')
            for i in range(len(iv) - 2):
                three_iv = iv.iloc[i:(i + 3), :]['iv']
                iv1 = three_iv.iloc[0]
                iv2 = three_iv.iloc[1]
                iv3 = three_iv.iloc[2]
                if iv2 > iv1 and iv2 > iv3:
                    # 哭脸,drop掉这三个点
                    print('cry face')
                    print(three_iv)
                    remain_idx = np.concatenate([np.arange(0, i + 1), np.arange(i + 2, len(iv))])
                    iv = iv.iloc[remain_idx, :]
                    return self.drop_cry_point(iv)
                else:
                    continue
        return iv

    def calibrate_atm_iv(self, iv: pd.DataFrame):
        """
        :param iv: 传入的待拟合iv, index执行价,两列columns是ln(K/S)和iv
        :return: atm iv
        """
        iv = self.drop_cry_point(iv)
        cs = calibrate_cs(iv['ln(k/s)'].values, iv['iv'].values.reshape(len(iv), 1),
                          knots=6, natural=False, smooth_index=0)
        X, knot = cs.form_spline()
        beta, y_hat = cs.fit(X)
        iv['cs_iv'] = y_hat
        interpolate_iv = cs.extrapolate(beta, knot, new_x=np.log(np.arange(0.9, 1.1, 0.01))).T
        interpolate_iv = interpolate_iv.set_index(0)
        interpolate_iv = interpolate_iv.rename(columns={1: 'cs_iv'})
        return iv, interpolate_iv

    def fit_pc_cc_vc1(self, iv, vc):
        """
        Desc: 只用处于3、4区间的数据来拟合pc cc和sr
        :param iv: drop cry point之后的点
        :param vc: atm iv
        :return: pc cc sc
        """
        # (1) 筛选出处于区间3、4中的点,并构造 y = X @ beta 中的矩阵 X
        iv34 = iv[(iv['ln(k/s)'] >= self.dc) & (iv['ln(k/s)'] <= self.uc)]
        X = np.array([[x**2 if x <= 0 else 0, x**2 if x > 0 else 0, x] for x in iv34['ln(k/s)']])
        y = iv34['iv'] - vc
        y = y.to_frame().values
        # (2) 最小二乘法求解 pc, cc, sc: y = X @ np.array([pc, cc, sc].T)
        mat1 = np.linalg.inv(X.T @ X) @ X.T
        beta = mat1 @ y
        beta = pd.Series(beta[:, 0], index=['pc', 'cc', 'sr'])
        beta['sc'] = beta['sr'] - self.scr * self.ssr * (self.atm - self.ref) / self.ref
        if pd.notna(self.sr):
            beta['sc'] = self.sr - self.scr * self.ssr * (self.atm - self.ref) / self.ref
        elif pd.notna(self.pc):
            beta['pc'] = self.pc
        elif pd.notna(self.cc):
            beta['cc'] = self.cc
        else:
            pass
        return beta

    def fit_pc_cc_vc2(self, iv, vc):
        """
        Desc: 用处于2、3、4、5区间的数据来拟合pc cc和sr
        :param iv: drop cry point之后的点
        :param vc: atm iv
        :return: pc cc sc
        """
        # (1) 筛选出处于区间2 3 4 5中的点,并构造 y = X @ beta 中的矩阵 X
        iv2345 = iv[(iv['ln(k/s)'] >= self.dc * (1 + self.dsm)) & (iv['ln(k/s)'] <= self.uc * (1 + self.usm))]
        X = []
        for x in iv2345['ln(k/s)']:
            if x <= self.dc:
                x_i = [-x**2/self.dsm + (1 + 1/self.dsm) * 2 * self.dc * x - (1 + 1/self.dsm) * self.dc**2,
                       0,
                       -x**2/(2 * self.dc * self.dsm) + (1 + 1/self.dsm) * x - self.dc/(2*self.dsm)]
            elif x <= 0:
                x_i = [x**2, 0, x]
            elif x <= self.uc:
                x_i = [0, x**2, x]
            else:
                x_i = [0,
                       -x**2/self.usm + (1 + 1/self.usm) * 2 * self.uc * x - (1 + 1/self.usm) * self.uc**2,
                       -x**2/(2 * self.uc * self.usm) + (1 + 1/self.usm) * x - self.uc/(2 * self.usm)]
            X.append(x_i)
        X = np.array(X)
        y = iv2345['iv'] - vc
        y = y.to_frame().values
        # (2) 最小二乘法求解 pc, cc, sc: y = X @ np.array([pc, cc, sc].T)
        mat1 = np.linalg.inv(X.T @ X) @ X.T
        beta = mat1 @ y
        beta = pd.Series(beta[:, 0], index=['pc', 'cc', 'sr'])
        beta['sc'] = beta['sr'] - self.scr * self.ssr * (self.atm - self.ref) / self.ref
        print(beta)
        if pd.notna(self.sr):
            beta['sc'] = self.sr - self.scr * self.ssr * (self.atm - self.ref)/self.ref
        elif pd.notna(self.pc):
            beta['pc'] = self.pc
        elif pd.notna(self.cc):
            beta['cc'] = self.cc
        else:
            pass

        return beta

    def calcu_wing_function(self, pc, cc, sc, vc):
        """
        Desc: 用前面计算出来的pc cc和vc来计算整条曲线
        :return: the whole wing model curve
        """
        f1 = vc + self.dc * (2 + self.dsm) * sc/2 + (1 + self.dsm) * pc * self.dc**2
        f2 = lambda x: -(pc/self.dsm + sc/(2 * self.dc * self.dsm)) * x**2 + \
                       (1 + 1/self.dsm) * (2 * pc * self.dc + sc) * x + \
                       vc - (1 + 1/self.dsm) * pc * self.dc**2 - sc * self.dc/(2*self.dsm)
        f3 = lambda x: pc * x**2 + sc * x + vc
        f4 = lambda x: cc * x**2 + sc * x + vc
        f5 = lambda x: -(cc/self.usm + sc/(2 * self.uc * self.usm)) * x**2 + \
                       (1 + 1/self.usm) * (2 * cc * self.uc + sc) * x + \
                       vc - (1 + 1/self.usm) * cc * self.uc**2 - sc * self.uc/(2*self.usm)
        f6 = vc + self.uc * (2 + self.usm) * sc/2 + (1 + self.usm) * cc * self.uc**2
        wing_function = [f1, f2, f3, f4, f5, f6]
        cut_point = [self.dc * (1 + self.dsm), self.dc, 0, self.uc, self.uc * (1 + self.usm)]
        return wing_function, cut_point

    def calibrate_mk_point(self, wing_function, cut_point, iv):
        """
        :param wing_function: the first output of calcu_wing_function
        :param cut_point: the second output of calcu_wing_function
        :param iv: market iv dataframe with 2 columns: iv and ln(k/s)
        :return: wing model iv dataframe with 2 columns: iv and ln(k/s)
        """
        iv = deepcopy(iv)
        wing_iv = []
        for ln_ks in iv['ln(k/s)']:
            if ln_ks <= cut_point[0]:
                wing_iv.append(wing_function[0])
            elif ln_ks <= cut_point[1]:
                wing_iv.append(wing_function[1](ln_ks))
            elif ln_ks <= cut_point[2]:
                wing_iv.append(wing_function[2](ln_ks))
            elif ln_ks <= cut_point[3]:
                wing_iv.append(wing_function[3](ln_ks))
            elif ln_ks <= cut_point[4]:
                wing_iv.append(wing_function[4](ln_ks))
            else:
                wing_iv.append(wing_function[5])
        wing_iv = pd.DataFrame(wing_iv, index=iv.index, columns=['wing_iv'])
        return wing_iv

    def extrapolate(self, wing_function, cut_point, ln_ks):
        """
        :param wing_function:
        :param cut_point:
        :param ln_ks: a list of k/s
        """
        wing_iv = []
        ln_ks = np.log(ln_ks)
        for i in ln_ks:
            if i <= cut_point[0]:
                wing_iv.append(wing_function[0])
            elif i <= cut_point[1]:
                wing_iv.append(wing_function[1](i))
            elif i <= cut_point[2]:
                wing_iv.append(wing_function[2](i))
            elif i <= cut_point[3]:
                wing_iv.append(wing_function[3](i))
            elif i <= cut_point[4]:
                wing_iv.append(wing_function[4](i))
            else:
                wing_iv.append(wing_function[5])
        wing_iv = pd.DataFrame(wing_iv, columns=['wing_iv'], index=ln_ks)
        return wing_iv

    def main(self, iv):
        iv = deepcopy(iv)
        # (1) 计算ln(K/S)
        iv['ln(k/s)'] = np.log(iv.index / self.underlying)
        # (2) 用3次样条对市场iv进行初次拟合，得到atm iv当作wing model中的参数vr
        if pd.isna(self.vr):
            iv, cs_iv = self.calibrate_atm_iv(iv)
            vr = cs_iv.loc[0, 'cs_iv']
        else:
            vr = self.vr
        vc = vr - self.ssr * self.vcr * (self.atm - self.ref) / self.ref
        # (3) 拟合wing model剩下的三个参数 pc, cc and vc
        beta = self.fit_pc_cc_vc2(iv, vc)
        # (4) 计算出整条wing model曲线
        wing_function, cut_point = self.calcu_wing_function(pc=beta['pc'], cc=beta['cc'], sc=beta['sc'], vc=vc)
        # (5) 计算出wing model拟合出来的市场的iv
        wing_iv = self.calibrate_mk_point(wing_function, cut_point, iv)
        iv['wing_iv'] = wing_iv['wing_iv']
        iv['iv-wing_iv'] = iv['iv'] - iv['wing_iv']
        extrapolate_wing = self.extrapolate(wing_function, cut_point, np.arange(0.8, 1.32, 0.02))
        extrapolate_wing['strike'] = np.exp(extrapolate_wing.index) * self.underlying
        return iv, extrapolate_wing

if __name__=="__main__":
    # (1) 读取iv数据
    iv = pd.DataFrame({'iv': [0.2373, 0.236, 0.23, 0.2265, 0.2252, 0.2225, 0.2212, 0.2204, 0.22, 0.2217, 0.2247, 0.2284, 0.2317, 0.2383, 0.2436, 0.2501, 0.2608, 0.2614, 0.2741, 0.2827, 0.2921, 0.2953, 0.2982],
                       'ln(k/atm)':[-0.18, -0.16, -0.14, -0.12, -0.09, -0.07, -0.05, -0.03, -0.01, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.11, 0.13, 0.15, 0.16, 0.18, 0.19, 0.21, 0.22]},
                       index = [11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000, 13250, 13500, 13750, 14000, 14250, 14500, 14750, 15000, 15250, 15500, 15750, 16000, 16250, 16500])
    # (2) 拟合wing
    underlying = cali_df[cali_df['instrument'] == future]['mid_price'].iloc[0]
    iv['ln(k/atm)'] = np.log(iv.index / underlying)
    wing1 = wing_model(atm=underlying * 1.05, ref=underlying, dc=-0.15, dsm=0.5, uc=0.15, usm=0.5, ssr=0)
    iv1, extrapolate_wing1 = wing1.main(iv)

    plt.plot(extrapolate_wing1['wing_iv'])
    plt.scatter(iv['ln(k/atm)'], iv['iv'], s=10, color='black')
    plt.xlabel('ln(K/S)')

    
