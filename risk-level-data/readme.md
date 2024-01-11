# Risk Level Index data

Here you can find the risk level data displayed on Risk DAO's bad debt dashboard (https://bad-debt.riskdao.org/), including detailed data for each protocol, ie: https://bad-debt.riskdao.org/risk-index?protocol=morpho-blue.


# Files

protocols_day_averages contains the latest weighted average risk value per protocol, computed with 30 days liquidity and volatility values.


The protocol specific files' structure is:
```
{
  "protocolAverageHistory": {
    "UnixTimeStamp": "RiskValue"
  },
  "VaultMarketWhatever": {
    "Collateral": {
      "numberOfDays_averageSpan": {
        "xDaysAverageVolatility": {
          "xDaysAverageLiquidity": "riskValue"
        }
      }
    }
  }
}

```
For instance, in the following data we can see that for the WETH vault / wstETH collateral, the average risk level over the last 3 days for the 30 days average volatility and the 30 days average liquidity was 3.356874388423344.

```
{
  "protocolAverageHistory": {
    "1704897577831": 4.96,
    "1704811177831": 4.93,
    "1704724777831": 4.91
  },
  "WETH": {
    "wstETH": {
      "3D_averageSpan": {
        "7": {
          "7": 3.295778471428988,
          "30": 3.356874388423344,
          "180": 2.9982574562463604
        },
        "30": {
          "7": 3.295778471428988,
          "30": 3.356874388423344,
          "180": 2.9982574562463604
        },
        "180": {
          "7": 3.295778471428988,
          "30": 3.356874388423344,
          "180": 2.9982574562463604
        }
      }
    }
  }
}

```
