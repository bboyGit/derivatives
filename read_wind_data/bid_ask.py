import numpy as np
import pandas as pd
from datetime import datetime
from WindPy import w
w.start()
from warnings import warn
from read_wind_data.generate_strike_series import strike_series

class read_bid_ask:

    def __init__(self, sec_name='SR.CZC', month=101, start_time="2020-11-05 21:00:00",
                 end_time="2020-11-06 15:00:00", lower_strike=4200, upper_strike=6300, strike_interval=100, file_format=1):
        """
        :param sec_name: 证券名称
        :param month: 合约月份
        :param start_time: 获取数据的开始时间
        :param end_time: 获取数据的结束时间
        :param strike_interval:
        :param file_format: 1-excel, 0-pikcle
        """
        self.sec_name = sec_name
        self.month = month
        self.start_time = start_time
        self.end_time = end_time
        if lower_strike >= upper_strike:
            raise Exception('lower_strike must smaller than upper_strike')
        self.lower_strike = lower_strike
        self.upper_strike = upper_strike
        self.strike_interval = strike_interval
        self.file_format = file_format

    def read_data(self):
        """
        Desc: 从wind读取期权和标的期货合约当个交易日的盘口数据并存入一个字典
        :return: 装有数据的字典，键是合约代码、值是合约对应的盘口数据
        """
        # (1) 获取行权价序列
        _strike = strike_series(lower_strike=self.lower_strike, upper_strike=self.upper_strike,
                                strike_interval=self.strike_interval)
        # (2) 构造期货合约代码列表
        contract, exchange = self.sec_name.split('.')
        _contract = contract + str(self.month)
        future_code = [_contract + '.' + exchange]
        # (3) 获取期权合约代码列表
        if exchange in ["CZC", "SHF"]:
            call_option_code = [_contract + 'C' + str(i) + '.' + exchange for i in _strike]
            put_option_code = [_contract + 'P' + str(i) + '.' + exchange for i in _strike]
        elif exchange == "DCE":
            call_option_code = [_contract + '-C-' + str(i) + '.' + exchange for i in _strike]
            put_option_code = [_contract + '-P-' + str(i) + '.' + exchange for i in _strike]
        else:
            raise Exception('unexpected exchange')
        sec_code = future_code + call_option_code + put_option_code
        # (4) 根据上一步中的代码以及起止时间从wind中读取盘口数据
        bid_ask_data = {}
        for _code in sec_code:
            _data = w.wst(_code, "bid1,ask1,bsize1,asize1", self.start_time, self.end_time, "")
            if _data.ErrorCode != 0:
                warn('{} wind API error: error code is {}'.format(_code, _data.ErrorCode))
            else:
                _df = pd.DataFrame(_data.Data).T
                _df.columns = _data.Fields
                _df.index = _data.Times
                bid_ask_data[_code] = _df
        return bid_ask_data

    def _write(self, bid_ask_data):
        """
        Desc: 将数据写到本地
        :param: bid_ask_data: output of read_data
        """
        contract, exchange = self.sec_name.split('.')
        _contract = contract + str(self.month)
        file_name = _contract + " from " + self.start_time + " to " + self.end_time
        file_name = file_name.replace(':', '')
        if self.file_format:
            excel_writer = pd.ExcelWriter('data/bid_ask_data/' + file_name + '.xlsx')
            for key, value in bid_ask_data.items():
                value.to_excel(excel_writer, sheet_name=key)
            excel_writer.save()
            excel_writer.close()
        else:
            pd.to_pickle(bid_ask_data, 'data/bid_ask_data/' + file_name + '.pkl')

    def exe(self):
        data = self.read_data()
        self._write(data)

def read_write(start_time, end_time, file_format=1):
    """
    Desc: 每个交易日读取各个我们需要存储的期权品种的数据并存入本地
    """
    # (1) 白糖
    for _month, lower, upper in zip([101, 103, 105, 107, 109], [4200, 4300, 4300, 4600, 4600], [6300, 5900, 5800, 5700, 5800]):
        print("SR{}.CZC".format(_month))
        read_bid_ask(sec_name='SR.CZC', month=_month, start_time=start_time, end_time=end_time,
                     lower_strike=lower, upper_strike=upper, strike_interval=100, file_format=file_format).exe()
    # (2) 棉花
    for _month, lower, upper in zip([101, 103, 105, 107, 109], [10000, 11000, 11400, 12000, 12200],
                                    [16200, 16200, 16200, 16400, 16400]):
        print("CF{}.CZC".format(_month))
        read_bid_ask(sec_name='CF.CZC', month=_month, start_time=start_time, end_time=end_time,
                     lower_strike=lower, upper_strike=upper, strike_interval=200, file_format=file_format).exe()
    # (3) 豆粕
    for _month, lower, upper in zip([2101, 2103, 2105, 2107, 2108, 2109], [2450, 2400, 2350, 2400, 2450, 2500],
                                    [3650, 3600, 3500, 3450, 3500, 3500]):
        print("M{}.DCE".format(_month))
        read_bid_ask(sec_name='M.DCE', month=_month, start_time=start_time, end_time=end_time,
                     lower_strike=lower, upper_strike=upper, strike_interval=50, file_format=file_format).exe()
    # (4) 玉米
    for _month, lower, upper in zip([2101, 2103, 2105, 2107, 2109], [1840, 1900, 1940, 2000, 2200],
                                    [2860, 2860, 2880, 2900, 2900]):
        print("C{}.DCE".format(_month))
        read_bid_ask(sec_name='C.DCE', month=_month, start_time=start_time, end_time=end_time,
                     lower_strike=lower, upper_strike=upper, strike_interval=20, file_format=file_format).exe()
    # (5) 橡胶
    for _month, lower, upper in zip([2101, 2103, 2104, 2105, 2106, 2107, 2109],
                                    [9100, 8900, 8400, 8500, 8500, 8700, 10250],
                                    [18500, 18250, 18250, 18250, 18000, 18000, 17500]):
        print("RU{}.SHF".format(_month))
        read_bid_ask(sec_name='RU.SHF', month=_month, start_time=start_time, end_time=end_time,
                     lower_strike=lower, upper_strike=upper,
                     strike_interval=pd.Series([100, 250, 500], index=[10000, 25000, np.inf]),
                     file_format=file_format).exe()
    # (6) 铜
    for _month, lower, upper in zip([2012, 2101, 2102, 2103, 2104, 2105],
                                    [33500, 34000, 34000, 34500, 35000, 36000],
                                    [57000, 57000, 57000, 57000, 57000, 57000]):
        print("CU{}.SHF".format(_month))
        read_bid_ask(sec_name='CU.SHF', month=_month, start_time=start_time, end_time=end_time,
                     lower_strike=lower, upper_strike=upper,
                     strike_interval=pd.Series([500, 1000, 2000], index=[40000, 80000, np.inf]),
                     file_format=file_format).exe()
    # (7) 黄金
    for _month, lower, upper in zip([2012, 2101, 2102, 2104, 2106],
                                    [304, 328, 296, 280, 320],
                                    [496, 480, 496, 496, 504]):
        print("AU{}.SHF".format(_month))
        read_bid_ask(sec_name='AU.SHF', month=_month, start_time=start_time, end_time=end_time,
                     lower_strike=lower, upper_strike=upper,
                     strike_interval=pd.Series([2, 4, 8], index=[200, 400, np.inf]),
                     file_format=file_format).exe()
    return

if __name__=="__main__":
    start = datetime.now()
    read_write("2020-11-17 21:00:00", "2020-11-18 15:00:00")
    end = datetime.now()
    print(end - start)
