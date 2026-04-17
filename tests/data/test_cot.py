from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from algotrading.data.cot import COTClient


class _FakeSocrata:
    def __init__(self, pages: list[list[dict[str, str]]] | None = None, latest: dict[str, str] | None = None):
        self.pages = pages or []
        self.latest = latest
        self.calls: list[dict[str, object]] = []

    def get(self, dataset_id: str, **kwargs):
        self.calls.append({"dataset_id": dataset_id, **kwargs})
        if kwargs.get("order") == "report_date_as_yyyy_mm_dd DESC":
            return [self.latest] if self.latest is not None else []

        offset = int(kwargs.get("offset", 0) or 0)
        limit = int(kwargs.get("limit", 50_000) or 50_000)
        index = offset // limit
        if index >= len(self.pages):
            return []
        return self.pages[index]

    def close(self) -> None:
        return None


def _row(d: str, noncomm_l=100, noncomm_s=60, comm_l=80, comm_s=90, oi=1000) -> dict[str, str]:
    return {
        "report_date_as_yyyy_mm_dd": d,
        "cftc_contract_market_code": "088691",
        "noncomm_positions_long_all": str(noncomm_l),
        "noncomm_positions_short_all": str(noncomm_s),
        "comm_positions_long_all": str(comm_l),
        "comm_positions_short_all": str(comm_s),
        "open_interest_all": str(oi),
    }


def test_fetch_historical_parses_requested_fields_and_computes_metrics() -> None:
    fake = _FakeSocrata(pages=[[_row("2026-01-07", 120, 70, 95, 80, 1000)]])
    client = COTClient(contract_code="088691", client=fake, page_size=10)

    reports = client.fetch_historical(start_date=date(2026, 1, 1), end_date=date(2026, 1, 31))

    assert len(reports) == 1
    r = reports[0]
    assert r.noncomm_longs == 120
    assert r.noncomm_shorts == 70
    assert r.noncomm_net == 50
    assert r.cot_spread == 35  # 50 - (95 - 80)
    # pct = longs / (longs + shorts) * 100
    assert r.noncomm_longs_pct == pytest.approx(120 / 190 * 100)
    assert r.noncomm_short_pct == pytest.approx(70 / 190 * 100)
    assert r.open_interest == 1000

    payload = r.to_strategy_data()
    assert payload["noncomm_longs"] == 120
    assert payload["noncomm_shorts"] == 70
    assert payload["noncomm_net"] == 50
    assert payload["cot_spread"] == 35
    assert payload["noncomm_longs_pct"] == pytest.approx(120 / 190 * 100)
    assert payload["noncomm_short_pct"] == pytest.approx(70 / 190 * 100)
    assert payload["noncomm_longs_oi_pct"] == pytest.approx(12.0)
    assert payload["noncomm_short_oi_pct"] == pytest.approx(7.0)
    assert payload["open_interest"] == 1000
    assert payload["chng_long"] == 0
    assert payload["chng_short"] == 0


def test_fetch_historical_paginates_until_last_partial_page() -> None:
    fake = _FakeSocrata(
        pages=[
            [_row("2026-01-07"), _row("2026-01-14")],
            [_row("2026-01-21")],
        ]
    )
    client = COTClient(contract_code="088691", client=fake, page_size=2)

    reports = client.fetch_historical()

    assert len(reports) == 3
    assert [r.report_date.isoformat() for r in reports] == ["2026-01-07", "2026-01-14", "2026-01-21"]


def test_report_at_returns_latest_report_on_or_before_timestamp() -> None:
    fake = _FakeSocrata(
        pages=[[
            _row("2026-01-07", noncomm_l=100),
            _row("2026-01-14", noncomm_l=120),
        ]]
    )
    client = COTClient(contract_code="088691", client=fake)
    client.fetch_historical()

    # release_lag_days=3 (default): Jan-14 report becomes available Jan 17 (Friday)
    # Querying Jan 16 (Thursday) should return Jan 7 report
    report_before_release = client.report_at(datetime(2026, 1, 16, tzinfo=timezone.utc))
    assert report_before_release is not None
    assert report_before_release.report_date.isoformat() == "2026-01-07"

    # Querying Jan 17 (the Friday it's released) should now return Jan 14 report
    report_on_release = client.report_at(datetime(2026, 1, 17, tzinfo=timezone.utc))
    assert report_on_release is not None
    assert report_on_release.report_date.isoformat() == "2026-01-14"

    data = client.strategy_data_at(datetime(2026, 1, 17, tzinfo=timezone.utc))
    assert data is not None
    assert data["noncomm_longs"] == 120


def test_report_at_accepts_numpy_datetime64() -> None:
    """bars.time[-1] is numpy.datetime64; must not silently fail the date comparison."""
    fake = _FakeSocrata(
        pages=[[
            _row("2026-01-07", noncomm_l=100),
            _row("2026-01-14", noncomm_l=120),
        ]]
    )
    client = COTClient(contract_code="088691", client=fake)
    client.fetch_historical()

    # Jan 17 as numpy datetime64 (nanoseconds, as bars.time stores)
    np_time = np.datetime64("2026-01-17T12:00:00", "ns")
    report = client.report_at(np_time)

    assert report is not None
    assert report.report_date.isoformat() == "2026-01-14"


def test_report_at_release_lag_zero_uses_report_date_directly() -> None:
    fake = _FakeSocrata(pages=[[_row("2026-01-14", noncomm_l=120)]])
    client = COTClient(contract_code="088691", client=fake, release_lag_days=0)
    client.fetch_historical()

    # With no lag the report is immediately available on its own date
    assert client.report_at(date(2026, 1, 14)) is not None
    # One day before -> nothing available yet
    assert client.report_at(date(2026, 1, 13)) is None


def test_fetch_historical_parses_datetime_style_report_date() -> None:
    fake = _FakeSocrata(
        pages=[[
            {
                "report_date_as_yyyy_mm_dd": "2026-01-14T00:00:00.000",
                "cftc_contract_market_code": "088691",
                "noncomm_positions_long_all": "120",
                "noncomm_positions_short_all": "70",
                "comm_positions_long_all": "95",
                "comm_positions_short_all": "80",
                "open_interest_all": "1000",
            }
        ]]
    )
    client = COTClient(contract_code="088691", client=fake)

    reports = client.fetch_historical()

    assert len(reports) == 1
    assert reports[0].report_date.isoformat() == "2026-01-14"


def test_refresh_live_rate_limited_and_merges_latest_report() -> None:
    fake = _FakeSocrata(
        pages=[[_row("2026-01-07")]],
        latest=_row("2026-01-21", noncomm_l=150),
    )
    client = COTClient(
        contract_code="088691",
        client=fake,
        live_refresh_interval=timedelta(hours=6),
    )
    client.fetch_historical()

    t0 = datetime(2026, 1, 22, 9, 0, tzinfo=timezone.utc)
    latest_1 = client.refresh_live(now_utc=t0)
    latest_2 = client.refresh_live(now_utc=t0 + timedelta(hours=1))
    latest_3 = client.refresh_live(now_utc=t0 + timedelta(hours=7))

    assert latest_1 is not None and latest_1.report_date.isoformat() == "2026-01-21"
    assert latest_2 is not None and latest_2.report_date.isoformat() == "2026-01-21"
    assert latest_3 is not None and latest_3.report_date.isoformat() == "2026-01-21"

    latest_calls = [c for c in fake.calls if c.get("order") == "report_date_as_yyyy_mm_dd DESC"]
    assert len(latest_calls) == 2
