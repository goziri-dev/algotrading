# algotrading

Un framework Python per progettare, testare, ottimizzare e tradare dal vivo strategie sistematiche tramite MetaTrader 5. Costruito da zero per offrire un ambiente unico e coerente in cui la stessa classe `Strategy` gira in modo identico nel backtest e contro un broker live.

---

## Caratteristiche principali

- **Backtest ed esecuzione live unificati.** La stessa istanza della strategia viene riscaldata dal backtester e passata al runner live. Indicatori, storico delle barre e posizioni aperte sopravvivono tutti alla transizione.
- **Multi-simbolo, multi-timeframe, multi-strategia.** Esegui un portafoglio di strategie — ciascuna sul proprio simbolo e mix di timeframe — contro un unico broker simulato o reale.
- **Modello di esecuzione realistico.** Spread, slippage, commissioni per lotto, margine, esecuzione di SL/TP, code di ordini limit/stop, gestione dei requote e dimensionamento specifico per contratto.
- **Risk sizing basato sull'equity.** Dimensiona le posizioni in base all'equity dell'intero portafoglio *oppure* alla sola equity della singola strategia, così il drawdown di una strategia non può ridurre la size di un'altra.
- **Ottimizzazione dei parametri con validazione walk-forward.** Grid search, random search, cross-validation time-series e walk-forward con finestre di training rolling o anchored.
- **Test di robustezza Monte Carlo** della curva di equity per separare l'abilità dalla fortuna del percorso.
- **Reportistica ricca.** Oltre 15 statistiche di sintesi, grafici statici matplotlib, HTML interattivi in Plotly e viste interattive suddivise trade-per-trade per l'audit.
- **Feed dati COT.** Ingestione integrata dei report Commitment of Traders della CFTC per filtri di regime/sentiment.
- **Completamente tipizzato, completamente testato.** Python 3.13, generics, e una suite pytest che copre broker, esecuzione, indicatori, ottimizzazione, plotting e il motore della sessione live.

---

## Esempio: SMACross su XAUUSD (gen 2024 → apr 2026)

```
==================================================
BACKTEST SUMMARY
==================================================
  Initial balance : 2,730.00
  Final balance   : 6,832.65
  Total PnL (net) : +4,102.65 (+150.28%)
  Total PnL (gross): +4,117.77 (+150.83%)
  Total costs     : 15.13  (spread=6.57  slippage=8.56  commission=0.00)
  Peak equity     : 7,476.86
  Trough equity   : 2,648.39
  Max drawdown    : -12.95% (-968.30)
  Trades          : 36  (W:33 / L:3)
  Win rate        : 91.7%
  Avg win         : +176.25 (+3.65%)
  Avg loss        : -571.15 (-8.39%)
  Profit factor   : 3.39
  Expectancy/trade: +113.96 (+2.64%)
  Avg hold time   : 21d 8h 41m
  Turnover        : 356,626.72
  Avg turnover    : 9,906.30 / trade
  Turnover / init : 130.6x
  Cost / gross    : 0.4%
  Exposure        : 92.3%
  CAGR            : 49.5%
  Sharpe (daily)  : 2.25
  Sortino (daily) : 3.33
  Calmar          : 3.82
==================================================
```

Il riepilogo è prodotto da `algotrading.backtest.print_backtest_summary`. I numeri riflettono un broker simulato con spread, slippage e contabilizzazione del margine realistici — non un replay senza attriti. Il codice sorgente della strategia si trova in [main.py](main.py).

---

## Avvio rapido

### Installazione

```bash
# uv è il package manager fissato dal progetto — vedi uv.lock
uv sync
```

Richiede Python 3.13+ e un terminale MetaTrader 5 installato localmente (Windows).

### Una strategia minima

```python
from dataclasses import dataclass
from datetime import datetime, timezone
import MetaTrader5 as mt5

from algotrading import Strategy, StrategyParams
from algotrading.backtest import print_backtest_summary
from algotrading.live.mt5 import MT5Session
from algotrading.indicators import SMA
from algotrading.utils import crossover, crossunder, fraction_to_qty


@dataclass
class MAParams(StrategyParams):
    fast: int = 10
    slow: int = 30


class MovingAverageCross(Strategy[MAParams]):
    def __init__(self, symbol: str = "XAUUSD", params: MAParams = MAParams()):
        super().__init__(symbol=symbol, params=params)
        self.fast = self.I(SMA(params.fast), source="close")
        self.slow = self.I(SMA(params.slow), source="close")

    def next(self) -> None:
        vpp = self.broker.value_per_point(self.symbol)
        qty = fraction_to_qty(self.equity, 99.9, self.price, vpp)
        if crossover(self.fast, self.slow):
            self.buy(qty=qty)
        elif crossunder(self.fast, self.slow):
            self.sell(qty=qty)


session = MT5Session(
    strategies=[MovingAverageCross()],
    primary_tf=mt5.TIMEFRAME_M15,
)

bt = session.backtest(
    initial_balance=10_000,
    slippage_pct=0.0024,
    date_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
)
print_backtest_summary(bt.broker)

# Promuovi la sessione già riscaldata al trading live — stesse istanze, stesso stato:
# session.go_live()
```

---

## Architettura

```
algotrading/
├── core/          Classe base Strategy, feed Bars, interfaccia Broker, Position,
│                  EntrySignal / PendingSignal, listener di esecuzione, runner.
├── backtest/      Broker simulato, motore di esecuzione, ottimizzazione (grid,
│                  random, walk-forward, cross-validation time-series),
│                  Monte Carlo, statistiche di sintesi, report matplotlib + Plotly.
├── live/mt5/      Adattatore broker MT5, loop della sessione live,
│                  cache delle specifiche di simbolo.
├── indicators/    SMA, EMA, RSI, ATR, BBANDS, ADX, Supertrend, HHLL, COTIndex.
│                  Supporta indicatori concatenati e multi-output (BBANDS, ADX).
├── data/          Client COT (Commitment of Traders) e modello dei report.
└── utils.py       crossover / crossunder / bars_since, filtro di sessione,
                   fraction_to_qty, risk_to_qty, MonthlyDrawdownTracker.

my_strategies/     Strategie di riferimento — bbands_reverse, rsi_reverse,
                   btc_scalper, hhll_debug, multi_strategy.

tests/             Suite pytest — backtest/, core/, data/, indicators/, live/.

docs/              Guide estese (vedi sotto).
```

### Note di design

- **Un'unica API Strategy per due modalità di esecuzione.** `Strategy.next()` viene invocato in modo identico dal backtester e dal runner live. L'unica cosa che cambia tra le modalità è quale implementazione di `Broker` la strategia interpella.
- **Le barre crescono solo in avanti.** Indicatori e strategie usano l'indicizzazione negativa (`self.fast[-1]`), che mappa in modo non ambiguo sullo storico completato. Una barra non viene mai revisionata dopo che `next()` l'ha vista.
- **I segnali sono valori, non side effect.** `self.buy(...)` e `self.sell(...)` emettono oggetti `EntrySignal` che il broker esegue. Gli ordini limit e stop vivono in una coda pending ispezionabile e cancellabile — il broker risolve i trigger contro i massimi/minimi di barra in simulazione e contro le esecuzioni reali in live.
- **Il sizing è disaccoppiato dai segnali.** `fraction_to_qty` e `risk_to_qty` sono funzioni pure di equity, prezzo e `value_per_point`, quindi la stessa logica di sizing funziona per ogni simbolo senza case speciali per FX, metalli o crypto.
- **Gli ID di strategia sono deterministici.** `self.id` è derivato dall'hash di `(classe, simbolo, params)`, ed è il meccanismo con cui il broker live instrada al set di posizioni corretto gli ordini delle riesecuzioni della stessa strategia.

---

## Documentazione

Le guide complete si trovano in [docs/](docs/):

- [Scrivere strategie](docs/writing-strategies.md) — l'API `Strategy` al completo: indicatori, multi-timeframe, sizing, SL/TP, ordini pending, callback, stato condiviso.
- [Eseguire strategie](docs/running-strategies.md) — `MT5Session` e `BacktestSession`, configurazioni multi-simbolo, specifiche di simbolo, plotting e come fare backtest senza una connessione MT5.
- [Ottimizzazione](docs/optimisation.md) — grid e random search, walk-forward analysis, cross-validation time-series, Monte Carlo.

---

## Test

```bash
uv run pytest
```

La suite copre:

- L'esecuzione del broker simulato, incluse le esecuzioni di SL/TP, i controlli di margine e la risoluzione delle code limit/stop.
- Ogni indicatore confrontato con valori di riferimento.
- L'intera pipeline di ottimizzazione e walk-forward.
- La generazione dei percorsi Monte Carlo.
- Le statistiche di sintesi e i path di plotting sia matplotlib che Plotly.
- Il motore della sessione live MT5 contro un broker mockato.

---

## Stack tecnologico

Python 3.13 · MetaTrader5 · numpy · pandas · matplotlib · plotly · lightweight-charts · tqdm · sodapy (API Socrata della CFTC) · uv · pytest.
