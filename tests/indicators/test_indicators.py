import math
import numpy as np
import pytest

from algotrading.indicators.sma import SMA
from algotrading.indicators.ema import EMA
from algotrading.indicators.rsi import RSI
from algotrading.indicators.bbands import BBANDS


class TestSMAInit:
    def test_default_capacity(self):
        sma = SMA(period=3)
        assert sma._capacity == 10_000
        assert sma._size == 0

    def test_set_capacity(self):
        sma = SMA(period=3)
        sma.set_capacity(10)
        assert sma._capacity == 10
        assert len(sma._value) == 10
        assert len(sma._input) == 10


class TestSMAValues:
    def test_nan_during_warmup(self):
        sma = SMA(period=3)
        assert math.isnan(sma(1.0)[-1])
        assert math.isnan(sma(2.0)[-1])

    def test_first_value_is_mean_of_period(self):
        sma = SMA(period=3)
        sma(1.0)
        sma(2.0)
        assert sma(3.0)[-1] == pytest.approx(2.0)

    def test_sliding_window(self):
        sma = SMA(period=3)
        for price in [1.0, 2.0, 3.0]:
            sma(price)
        assert sma(4.0)[-1] == pytest.approx(3.0)
        assert sma(5.0)[-1] == pytest.approx(4.0)

    def test_previous_value_via_index(self):
        sma = SMA(period=3)
        for price in [1.0, 2.0, 3.0, 4.0]:
            sma(price)
        assert sma[-1] == pytest.approx(3.0)
        assert sma[-2] == pytest.approx(2.0)

    def test_size_increments(self):
        sma = SMA(period=3)
        for price in [1.0, 2.0, 3.0]:
            sma(price)
        assert sma._size == 3


class TestSMACapacity:
    def test_increases_when_full(self):
        sma = SMA(period=2)
        sma.set_capacity(2)
        sma(1.0)
        sma(2.0)
        sma(3.0)
        assert sma._capacity >= 3
        assert sma._size == 3


class TestEMAInit:
    def test_default_capacity(self):
        ema = EMA(period=3)
        assert ema._capacity == 10_000
        assert ema._size == 0

    def test_multiplier(self):
        ema = EMA(period=3)
        assert ema._k == pytest.approx(2.0 / 4.0)


class TestEMAValues:
    def test_nan_during_warmup(self):
        ema = EMA(period=3)
        assert math.isnan(ema(1.0)[-1])
        assert math.isnan(ema(2.0)[-1])

    def test_seed_equals_sma(self):
        ema = EMA(period=3)
        ema(1.0)
        ema(2.0)
        assert ema(3.0)[-1] == pytest.approx(2.0)

    def test_ema_formula(self):
        # k = 0.5 for period=3
        # seed = mean(1,2,3) = 2.0
        # ema(4) = 4*0.5 + 2.0*0.5 = 3.0
        # ema(5) = 5*0.5 + 3.0*0.5 = 4.0
        ema = EMA(period=3)
        for price in [1.0, 2.0, 3.0]:
            ema(price)
        assert ema(4.0)[-1] == pytest.approx(3.0)
        assert ema(5.0)[-1] == pytest.approx(4.0)

    def test_gives_more_weight_to_recent_prices(self):
        ema = EMA(period=3)
        sma = SMA(period=3)
        prices = [10.0, 10.0, 10.0, 20.0]
        for p in prices:
            ema(p)
            sma(p)
        assert ema[-1] > sma[-1]


class TestEMACapacity:
    def test_increases_when_full(self):
        ema = EMA(period=2)
        ema.set_capacity(2)
        ema(1.0)
        ema(2.0)
        ema(3.0)
        assert ema._capacity >= 3
        assert ema._size == 3


class TestRSIInit:
    def test_default_period_and_capacity(self):
        rsi = RSI()
        assert rsi._period == 14
        assert rsi._capacity == 10_000
        assert rsi._size == 0


class TestRSIValues:
    def test_nan_during_warmup(self):
        # period=3 needs period+1=4 prices; first 3 must all be nan
        rsi = RSI(period=3)
        assert math.isnan(rsi(10.0)[-1])
        assert math.isnan(rsi(11.0)[-1])
        assert math.isnan(rsi(12.0)[-1])

    def test_all_gains_returns_100(self):
        # changes [1,1,1] → rma_gain=1, rma_loss=0 → RSI=100
        rsi = RSI(period=3)
        rsi(10.0)
        rsi(11.0)
        rsi(12.0)
        assert rsi(13.0)[-1] == pytest.approx(100.0)

    def test_all_losses_returns_0(self):
        # changes [-1,-1,-1] → rma_gain=0, rma_loss=1 → RSI=0
        rsi = RSI(period=3)
        rsi(12.0)
        rsi(11.0)
        rsi(10.0)
        assert rsi(9.0)[-1] == pytest.approx(0.0)

    def test_seed_value(self):
        # prices [10,12,11,13], changes [+2,-1,+2]
        # rma_gain seeded = sma([2,0,2]) = 4/3
        # rma_loss seeded = sma([0,1,0]) = 1/3
        # RSI = 100 - 100/(1+4) = 80.0
        rsi = RSI(period=3)
        rsi(10.0)
        rsi(12.0)
        rsi(11.0)
        assert rsi(13.0)[-1] == pytest.approx(80.0)

    def test_wilder_smoothing(self):
        # continuing from seed above: up=4/3, down=1/3
        # price=10, change=-3 → gain=0, loss=3
        # up  = (4/3*2 + 0) / 3 = 8/9
        # down = (1/3*2 + 3) / 3 = 11/9
        # RSI = 100 - 100/(1 + 8/11) = 800/19
        rsi = RSI(period=3)
        for p in [10.0, 12.0, 11.0, 13.0]:
            rsi(p)
        assert rsi(10.0)[-1] == pytest.approx(800 / 19)

    def test_size_increments(self):
        rsi = RSI(period=3)
        rsi(10.0)
        rsi(12.0)
        rsi(11.0)
        assert rsi._size == 3


class TestRSICapacity:
    def test_increases_when_full(self):
        rsi = RSI(period=2)
        rsi.set_capacity(2)
        rsi(10.0)
        rsi(11.0)
        rsi(12.0)
        assert rsi._capacity >= 3
        assert rsi._size == 3


class TestBBands:
    def test_default_basis_is_sma(self):
        bb = BBANDS(period=3, mult=2.0)
        bb(1.0)
        bb(2.0)
        out = bb(3.0)
        assert out.mid == pytest.approx(2.0)

        out = bb(4.0)
        assert out.mid == pytest.approx(3.0)

    def test_smma_rma_basis(self):
        bb = BBANDS(period=3, mult=2.0, ma_type="SMMA (RMA)")
        bb(1.0)
        bb(2.0)
        out = bb(3.0)
        assert out.mid == pytest.approx(2.0)

        out = bb(4.0)
        # RMA(period=3): (prev * 2 + price) / 3
        assert out.mid == pytest.approx(8.0 / 3.0)

    def test_rejects_unsupported_ma_type(self):
        with pytest.raises(ValueError, match="Unsupported ma_type"):
            BBANDS(period=20, mult=2.0, ma_type="VWMA")
