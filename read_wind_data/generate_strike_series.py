import numpy as np
import pandas as pd

def strike_series(lower_strike, upper_strike, strike_interval):
    """
    Desc: 计算执行价序列
    :param lower_strike: 最低执行价
    :param upper_strike: 最高执行价
    :param strike_interval: 执行价间距. a scalar or a series. if it's sereis, index is break point and value is interval.
    :return: A strike list
    """
    if lower_strike >= upper_strike:
        raise Exception('lower_strike must smaller than upper_strike')
    if isinstance(strike_interval, pd.Series):
        threshold = strike_interval.index
        if threshold[-1] == np.inf and len(strike_interval) >= 2:
            if lower_strike >= threshold[-2]:
                strike_interval = strike_interval.iloc[-2]               # 最低执行价 > break_point的上限
                _strike = np.arange(lower_strike, upper_strike + 1, strike_interval)
            elif upper_strike <= threshold[0]:
                strike_interval = strike_interval.iloc[0]               # 最高执行价 < break_point的下限
                _strike = np.arange(lower_strike, upper_strike + 1, strike_interval)
            else:
                lower_idx = np.where(lower_strike <= threshold)[0][0]         # 大于等于lower_strike的第一个阈值所对应的index
                upper_idx = np.where(upper_strike >= threshold)[0][-1]       # 小于等于upper_strike的最后一个阈值所对应的index
                if lower_idx < upper_idx:
                    # 说明lower_strike和upper_strike中间框进来了不止一个break_point
                    real_use_strike = strike_interval.iloc[lower_idx: (upper_idx + 2)]  # 真正会用到的执行价序列
                    _strike = []
                    for idx, break_point, interval in zip(range(len(real_use_strike)), real_use_strike.index,
                                                          real_use_strike.values):
                        if idx == 0:
                            # 生成lower_strike到第一个break_point之间的interval
                            the_strike_sereis = np.arange(lower_strike, break_point, interval)
                            _strike.append(the_strike_sereis)
                        elif idx == len(real_use_strike) - 1:
                            # 生成最后一个strike_point到upper_strike之间的strike
                            the_strike_sereis = np.arange(real_use_strike.index[idx - 1], upper_strike + 1, interval)
                            _strike.append(the_strike_sereis)
                        else:
                            # 生成上一个break_point到这个break_point之间的strike
                            the_strike_sereis = np.arange(real_use_strike.index[idx - 1], break_point, interval)
                            _strike.append(the_strike_sereis)
                    _strike = np.concatenate(_strike)
                elif lower_idx == upper_idx:
                    # 说明lower_strike和upper_strike中间框进来了一个break_point
                    _strike = []
                    the_strike_series = np.arange(lower_strike, threshold[lower_idx], strike_interval.iloc[lower_idx])
                    _strike.append(the_strike_series)
                    the_strike_series = np.arange(threshold[lower_idx], upper_strike + 1, strike_interval.iloc[lower_idx + 1])
                    _strike.append(the_strike_series)
                    _strike = np.concatenate(_strike)
                else:
                    strike_interval = strike_interval.iloc[lower_idx]  # 说明lower_strike和upper_strike中间没有框进来break_point
                    _strike = np.arange(lower_strike, upper_strike + 1, strike_interval)
        else:
            raise Exception('The last index of strike_interval must be infinite and length of strike_price must bigger than 1')
    elif isinstance(strike_interval, int):
        _strike = np.arange(lower_strike, upper_strike + 1, strike_interval)
    else:
        raise Exception('strike_interval can only be pd.Seres or a scalar int')
    _strike = [int(i) for i in _strike]
    return _strike


if __name__=="__main__":
    strike_series(lower_strike=4200, upper_strike=6300, strike_interval=100)
    strike_series(lower_strike=4500, upper_strike=8500, strike_interval=pd.Series([100, 250, 500, 1000], index=[5000, 7000, 9000, np.inf]))
    strike_series(lower_strike=4500, upper_strike=12000, strike_interval=pd.Series([100, 250, 500, 1000], index=[5000, 7000, 9000, np.inf]))
