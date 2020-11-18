
import numpy as np
import pandas as pd
from option.fundamental_tools.implied_volatility import implied_vol
from option.fundamental_tools.greeks import Greeks
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
import os
os.chdir('d://lyh/project/derivative/')                              # 修改工作目录

class greek_analysis:
    """
    Desc: 画具体某一天真实行情的iv曲线、希腊值曲线、希腊值的skew曲线
    """
    def __init__(self, sec_name, month, curr_date, pre_date, maturity_date, rf, atm_strike):
        """
        :param sec_name: such as SR
        :param month: such as 101 or 2101
        :param curr_date: such as '2020-11-16'
        """
        self.sec_name = sec_name
        self.month = month
        self.curr_date = curr_date
        self.pre_date = pre_date
        self.ttm = (maturity_date - datetime.strptime(curr_date, '%Y-%m-%d')).days / 365
        self.rf = rf
        self.atm_strike = atm_strike

    def read_data(self):
        """
        :return: 字典，键为合约代码、值为合约对应的盘口一档信息
        """
        contract = self.sec_name + str(self.month)
        file_name = contract + ' from ' + self.pre_date + ' 210000 to ' + self.curr_date + ' 150000.xlsx'
        url = 'data/bid_ask_data/' + file_name
        data = pd.read_excel(url, sheet_name=None)
        return data

    def get_data_in_time_interval(self, data, start_time, end_time):
        """
        Desc: 获取一个既定很短时间范围内每个合约最后一次报价的bid ask
        :param data: output of read_data
        :return: 一个dataframe，每一行是一个合约。
        """
        df = []
        for key, value in data.items():
            value.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
            sub_df = value[(value['datetime'] >= start_time) & (value['datetime'] <= end_time)]
            if len(sub_df) > 0:
                sub_df = sub_df.iloc[[-1], :]
            else:
                continue
            sub_df.loc[:, 'code'] = key
            df.append(sub_df)
        df = pd.concat(df).set_index('datetime')
        df = df.dropna()
        return df

    def get_strike(self, df):
        """
        Desc: get strike, option_type and mid price
        :param df: output of get_data_in_time_interval
        """
        _strikes = []
        opt_type = []
        for i, _code in enumerate(df['code']):
            reg = re.search(r'[C,P][0-9]+.', _code)
            if pd.isnull(reg):
                _strikes.append(None)              # 期货合约
                opt_type.append(None)
            else:
                strike = int(reg.group()[1:-1])
                _strikes.append(strike)
                opt_type.append(reg.group()[0])
        df.loc[:, 'strike'] = _strikes
        df.loc[:, 'opt_type'] = opt_type
        df.loc[:, 'mid'] = (df['bid1'] + df['ask1'])/2
        return df

    def iv_curve(self, df):
        """
        :param df: output of get_strike
        """
        # (1) 计算iv
        iv = []
        df1 = df.dropna()
        s0 = df[df['strike'].isna()]['mid'].iloc[0]
        for k, opt_type, f in zip(df1['strike'], df1['opt_type'], df1['mid']):
            if opt_type == "C":
                call = 1
            elif opt_type == "P":
                call = 0
            else:
                raise ValueError('opt_type 只能是C或P')
            iv_obj = implied_vol(f=f, s0=s0, K=k, r=self.rf, T=self.ttm, q=0, option_type=1, call=call, max_iter=100)
            _iv = iv_obj.bisection(low=0, up=1, threshold=10**(-5))['implied_sigma']
            iv.append(_iv)
        df1.loc[:, 'mid_iv'] = iv
        df1 = pd.concat([df[df['strike'].isna()], df1])
        df1 = df1[((df1['mid_iv'] < 1) & (df1['mid_iv'] > 0.01)) | (df1['mid_iv'].isna())]
        sub_df = df1[df1['mid_iv'].notna()]
        # (2) 画iv曲线
        plt.scatter(sub_df[sub_df['opt_type'] == 'C']['strike'], sub_df[sub_df['opt_type'] == 'C']['mid_iv'])
        plt.scatter(sub_df[sub_df['opt_type'] == 'P']['strike'], sub_df[sub_df['opt_type'] == 'P']['mid_iv'])
        plt.legend(['call', 'put'])
        plt.title(self.sec_name + ' ' + str(self.month) + ' mid iv' + ' & ' + self.curr_date)
        plt.show()
        return df1

    def calcu_greeks(self, df, _greek):
        """
        :param df: output of iv_curve
        :return: 带有希腊值的df
        """
        # (1) calculate greeks
        greeks = []
        df1 = df.dropna()
        s0 = df[df['strike'].isna()]['mid'].iloc[0]
        for k, opt_type, f, iv in zip(df1['strike'], df1['opt_type'], df1['mid'], df1['mid_iv']):
            if opt_type == "C":
                call = 1
            elif opt_type == "P":
                call = 0
            else:
                raise ValueError('opt_type 只能是C或P')
            greek = Greeks(f=f, s0=s0, K=k, T=self.ttm, r=self.rf, call=call, option_type=1, sigma=iv)
            if _greek == 'delta':
                greeks.append(greek.delta())
            elif _greek == 'gamma':
                greeks.append(greek.gamma())
            elif _greek == 'vega':
                greeks.append(greek.vega())
            elif _greek == 'theta':
                greeks.append(greek.theta())
        df1.loc[:, _greek] = greeks
        df1 = pd.concat([df[df['strike'].isna()], df1])
        # (2) plot greek curve
        sub_df = df1[(df1['mid_iv'] < 1) & (df1['mid_iv'] > 0.01)]
        plt.scatter(sub_df[sub_df['opt_type'] == 'C']['strike'], sub_df[sub_df['opt_type'] == 'C'][_greek])
        plt.scatter(sub_df[sub_df['opt_type'] == 'P']['strike'], sub_df[sub_df['opt_type'] == 'P'][_greek])
        plt.legend(['call', 'put'])
        plt.title(self.sec_name + ' ' + str(self.month) + _greek + ' & ' + self.curr_date)
        plt.show()
        return df1

    def calcu_skew(self, df, feature):
        """
        :param df:  output of calcu_greeks
        :param feature: 希腊值或是iv
        """
        # (1) 计算skew
        if feature + '_skew' not in df.columns:
            df.loc[:, feature + '_skew'] = None
            df1 = df[df['opt_type'].notnull()]
            if 'iv' not in feature:
                df1.loc[df1['opt_type'] == 'C', feature + '_skew'] = df1.loc[(df1['opt_type'] == 'C') &
                                                                             (df1['strike'] == self.atm_strike), feature].iloc[0] / df1.loc[df1['opt_type'] == 'C', feature]
                df1.loc[df1['opt_type'] == 'P', feature + '_skew'] = df1.loc[(df1['opt_type'] == 'P') &
                                                                             (df1['strike'] == self.atm_strike), feature].iloc[0] / df1.loc[df1['opt_type'] == 'P', feature]
            else:
                df1.loc[df1['opt_type'] == 'C', feature + '_skew'] = df1.loc[df1['opt_type'] == 'C', feature] / df1.loc[(df1['opt_type'] == 'C') &
                                                                                                                        (df1['strike'] == self.atm_strike), feature].iloc[0]
                df1.loc[df1['opt_type'] == 'P', feature + '_skew'] = df1.loc[df1['opt_type'] == 'P', feature] / df1.loc[(df1['opt_type'] == 'P') &
                                                                                                                        (df1['strike'] == self.atm_strike), feature].iloc[0]
            df1 = pd.concat([df[df['opt_type'].isna()], df1])
        else:
            df1 = df.copy()
        # (2) skew plot
        plt.scatter(df1[df1['opt_type'] == 'C']['strike'], df1[df1['opt_type'] == 'C'][feature + '_skew'])
        plt.scatter(df1[df1['opt_type'] == 'P']['strike'], df1[df1['opt_type'] == 'P'][feature + '_skew'])
        plt.xlabel('strike')
        if 'iv' not in feature:
            plt.ylabel('atm/others')
        else:
            plt.ylabel('others/atm')
        plt.legend(['call', 'put'])
        plt.title(self.sec_name + str(self.month) + ' ' + feature + ' skew' + ' & ' + self.curr_date)
        plt.show()
        return df1

if __name__=="__main__":
    analyst = greek_analysis(sec_name="CU", month='2012', curr_date='2020-11-17', pre_date='2020-11-16',
                             maturity_date=datetime(2020, 11, 24), rf=0.015, atm_strike=53000)
    sr = analyst.read_data()
    df = analyst.get_data_in_time_interval(sr, datetime(2020, 11, 17, 14, 50), datetime(2020, 11, 17, 14, 55))
    df = analyst.get_strike(df)
    df = analyst.iv_curve(df)
    df = analyst.calcu_greeks(df, 'gamma')
    df = analyst.calcu_greeks(df, 'vega')
    df = analyst.calcu_greeks(df, 'theta')
    df = analyst.calcu_greeks(df, 'delta')
    df = analyst.calcu_skew(df, 'gamma')
    df = analyst.calcu_skew(df, 'vega')
    df = analyst.calcu_skew(df, 'theta')
    df = analyst.calcu_skew(df, 'delta')
    df = analyst.calcu_skew(df, 'mid_iv')
