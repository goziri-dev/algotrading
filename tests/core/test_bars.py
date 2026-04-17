import numpy as np

from algotrading.core.bars import Bars


class TestBarsInit:
    def test_default_capacity(self):
        bars = Bars()
        assert bars._capacity == 10_000
        assert bars._size == 0

    def test_set_capacity(self):
        bars = Bars()
        bars.set_capacity(50)
        assert bars._capacity == 50
        assert len(bars._open) == 50
        assert len(bars._high) == 50
        assert len(bars._low) == 50
        assert len(bars._close) == 50
        assert len(bars._time) == 50


class TestBarsUpdate:
    def test_update(self):
        bars = Bars()
        bars.update(time=np.datetime64('2024-01-01T00:00:00'), open=100.0, high=110.0, low=90.0, close=105.0)
        assert bars._size == 1
        assert bars._time[0] == np.datetime64('2024-01-01T00:00:00')
        assert bars._open[0] == 100.0
        assert bars._high[0] == 110.0
        assert bars._low[0] == 90.0
        assert bars._close[0] == 105.0

    def test_update_increase_capacity(self):
        bars = Bars()
        bars.set_capacity(2)
        for i in range(3):
            bars.update(time=np.datetime64(f'2024-01-01T00:00:0{i}'), open=100.0 + i, high=110.0 + i, low=90.0 + i, close=105.0 + i)
        assert bars._size == 3
        assert bars._capacity >= 3
