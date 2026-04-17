from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from sodapy import Socrata


LEGACY_FUTURES_ONLY_DATASET_ID = "6dca-aqww"
DEFAULT_CFTC_HOST = "publicreporting.cftc.gov"


@dataclass(frozen=True)
class COTReport:
    report_date: date
    contract_code: str
    noncomm_longs: int
    noncomm_shorts: int
    comm_longs: int
    comm_shorts: int
    open_interest: int
    chng_long: int = 0
    chng_short: int = 0

    @property
    def noncomm_net(self) -> int:
        return self.noncomm_longs - self.noncomm_shorts

    @property
    def comm_net(self) -> int:
        return self.comm_longs - self.comm_shorts

    @property
    def cot_spread(self) -> int:
        return self.noncomm_net - self.comm_net

    @property
    def noncomm_longs_pct(self) -> float:
        """Longs as % of total noncomm directional positions (longs + shorts)."""
        total = self.noncomm_longs + self.noncomm_shorts
        if total <= 0:
            return 0.0
        return self.noncomm_longs * 100.0 / total

    @property
    def noncomm_short_pct(self) -> float:
        """Shorts as % of total noncomm directional positions (longs + shorts)."""
        total = self.noncomm_longs + self.noncomm_shorts
        if total <= 0:
            return 0.0
        return self.noncomm_shorts * 100.0 / total

    @property
    def noncomm_longs_oi_pct(self) -> float:
        """Longs as % of total open interest."""
        if self.open_interest <= 0:
            return 0.0
        return self.noncomm_longs * 100.0 / self.open_interest

    @property
    def noncomm_short_oi_pct(self) -> float:
        """Shorts as % of total open interest."""
        if self.open_interest <= 0:
            return 0.0
        return self.noncomm_shorts * 100.0 / self.open_interest

    def to_strategy_data(self) -> dict[str, float | int]:
        """Return the key fields expected by strategies/bars dynamic fields."""
        return {
            "noncomm_longs": self.noncomm_longs,
            "noncomm_shorts": self.noncomm_shorts,
            "noncomm_net": self.noncomm_net,
            "cot_spread": self.cot_spread,
            "noncomm_longs_pct": self.noncomm_longs_pct,
            "noncomm_short_pct": self.noncomm_short_pct,
            "noncomm_longs_oi_pct": self.noncomm_longs_oi_pct,
            "noncomm_short_oi_pct": self.noncomm_short_oi_pct,
            "open_interest": self.open_interest,
            "chng_long": self.chng_long,
            "chng_short": self.chng_short,
        }


class COTClient:
    """Fetch CFTC legacy futures-only COT reports via Socrata.

    The client supports both historical downloads and low-frequency live refresh,
    so a strategy can either pre-load COT history for backtests or query aligned
    values from ``update_bars()`` in live mode.
    """

    def __init__(
        self,
        contract_code: str,
        *,
        app_token: str | None = None,
        host: str = DEFAULT_CFTC_HOST,
        dataset_id: str = LEGACY_FUTURES_ONLY_DATASET_ID,
        live_refresh_interval: timedelta = timedelta(hours=6),
        release_lag_days: int = 3,
        page_size: int = 50_000,
        dotenv_path: str | Path = ".env",
        client: Socrata | None = None,
    ):
        if not contract_code.strip():
            raise ValueError("contract_code must not be empty")
        if page_size <= 0:
            raise ValueError("page_size must be > 0")
        if release_lag_days < 0:
            raise ValueError("release_lag_days must be >= 0")

        self.contract_code = contract_code.strip()
        self.host = host
        self.dataset_id = dataset_id
        self.live_refresh_interval = live_refresh_interval
        self.release_lag_days = release_lag_days
        self.page_size = page_size

        token = app_token or _token_from_env(dotenv_path)
        self._client = client or Socrata(host, token, timeout=30)

        self._reports: list[COTReport] = []
        self._last_live_poll_utc: datetime | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "COTClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def reports(self) -> list[COTReport]:
        return list(self._reports)

    def fetch_historical(
        self,
        *,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> list[COTReport]:
        """Download and cache historical reports for this contract code."""
        where_clauses = [
            f"cftc_contract_market_code = '{self.contract_code}'",
        ]
        if start_date is not None:
            where_clauses.append(
                f"report_date_as_yyyy_mm_dd >= '{_as_date(start_date).isoformat()}'"
            )
        if end_date is not None:
            where_clauses.append(
                f"report_date_as_yyyy_mm_dd <= '{_as_date(end_date).isoformat()}'"
            )

        offset = 0
        all_rows: list[dict[str, Any]] = []
        while True:
            rows = cast(
                list[dict[str, Any]],
                self._client.get(
                    self.dataset_id,
                    select=(
                        "report_date_as_yyyy_mm_dd,"
                        "cftc_contract_market_code,"
                        "noncomm_positions_long_all,"
                        "noncomm_positions_short_all,"
                        "comm_positions_long_all,"
                        "comm_positions_short_all,"
                        "open_interest_all,"
                        "change_in_noncomm_long_all,"
                        "change_in_noncomm_short_all"
                    ),
                    where=" AND ".join(where_clauses),
                    order="report_date_as_yyyy_mm_dd ASC",
                    limit=self.page_size,
                    offset=offset,
                ),
            )
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < self.page_size:
                break
            offset += self.page_size

        reports = [_report_from_row(row) for row in all_rows]
        self._reports = reports
        return list(reports)

    def fetch_latest(self) -> COTReport | None:
        """Fetch the latest available report for this contract code."""
        rows = cast(
            list[dict[str, Any]],
            self._client.get(
                self.dataset_id,
                select=(
                    "report_date_as_yyyy_mm_dd,"
                    "cftc_contract_market_code,"
                    "noncomm_positions_long_all,"
                    "noncomm_positions_short_all,"
                    "comm_positions_long_all,"
                    "comm_positions_short_all,"
                    "open_interest_all,"
                    "change_in_noncomm_long_all,"
                    "change_in_noncomm_short_all"
                ),
                where=f"cftc_contract_market_code = '{self.contract_code}'",
                order="report_date_as_yyyy_mm_dd DESC",
                limit=1,
            ),
        )
        if not rows:
            return None
        return _report_from_row(rows[0])

    def refresh_live(self, now_utc: datetime | None = None) -> COTReport | None:
        """Refresh cached data with the most recent report, rate-limited by interval."""
        now = now_utc or datetime.now(timezone.utc)
        if self._last_live_poll_utc is not None:
            if now - self._last_live_poll_utc < self.live_refresh_interval:
                return self._reports[-1] if self._reports else None

        latest = self.fetch_latest()
        self._last_live_poll_utc = now
        if latest is None:
            return None
        if not self._reports:
            self._reports = [latest]
            return latest

        if latest.report_date > self._reports[-1].report_date:
            self._reports.append(latest)
        elif latest.report_date == self._reports[-1].report_date:
            self._reports[-1] = latest
        return self._reports[-1]

    def report_at(
        self,
        at_time: date | datetime | np.datetime64,
        *,
        auto_live_refresh: bool = False,
    ) -> COTReport | None:
        """Return the latest report that was publicly available at ``at_time``.

        The CFTC publishes COT reports every Friday for data as of the prior
        Tuesday (``report_date``).  To avoid look-ahead bias, a report dated
        Tuesday is only considered available ``release_lag_days`` later
        (default 3 — i.e., the Friday of the same week).  Pass
        ``release_lag_days=0`` only if you want raw report-date alignment
        without the publication delay.
        """
        if auto_live_refresh:
            self.refresh_live()

        if not self._reports:
            return None

        cutoff = _as_date(at_time)
        lag = timedelta(days=self.release_lag_days)
        for report in reversed(self._reports):
            if report.report_date + lag <= cutoff:
                return report
        return None

    def strategy_data_at(
        self,
        at_time: date | datetime | np.datetime64,
        *,
        auto_live_refresh: bool = False,
    ) -> dict[str, float | int] | None:
        report = self.report_at(at_time, auto_live_refresh=auto_live_refresh)
        if report is None:
            return None
        return report.to_strategy_data()


def _as_date(value: date | datetime | np.datetime64) -> date:
    if isinstance(value, np.datetime64):
        # numpy stores bars.time as datetime64[ns]; convert to UTC date.
        ts = int(value.astype("datetime64[ms]").astype("int64"))
        return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).date()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.date()
    return value


def _report_from_row(row: dict[str, Any]) -> COTReport:
    raw_date = str(row["report_date_as_yyyy_mm_dd"])
    report_date = datetime.strptime(raw_date.split("T", 1)[0], "%Y-%m-%d").date()
    return COTReport(
        report_date=report_date,
        contract_code=str(row.get("cftc_contract_market_code", "")).strip(),
        noncomm_longs=_to_int(row.get("noncomm_positions_long_all")),
        noncomm_shorts=_to_int(row.get("noncomm_positions_short_all")),
        comm_longs=_to_int(row.get("comm_positions_long_all")),
        comm_shorts=_to_int(row.get("comm_positions_short_all")),
        open_interest=_to_int(row.get("open_interest_all")),
        chng_long=_to_int(row.get("change_in_noncomm_long_all")),
        chng_short=_to_int(row.get("change_in_noncomm_short_all")),
    )


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value).replace(",", ""))


def _token_from_env(dotenv_path: str | Path) -> str | None:
    env_token = (
        os.getenv("SCORATA_APP_TOKEN")
        or os.getenv("SOCRATA_APP_TOKEN")
        or os.getenv("SODAPY_APP_TOKEN")
    )
    if env_token:
        return env_token

    path = Path(dotenv_path)
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        k = key.strip()
        if k in {"SCORATA_APP_TOKEN", "SOCRATA_APP_TOKEN", "SODAPY_APP_TOKEN"}:
            v = value.strip().strip('"').strip("'")
            if v:
                return v

    return None
