# Risk Level Index data

Here you can find the risk level data displayed on Risk DAO's bad debt dashboard (https://bad-debt.riskdao.org/), including detailed data for each protocol, ie: https://bad-debt.riskdao.org/risk-index?protocol=morpho-blue.


# Files

protocols_day_averages contains the latest weighted average risk value per protocol, computed with 30 days liquidity and volatility values.

## Morpho File
Morpho files' structure is:
```
{
    "protocolAverageHistory": {
        "UnixTimeStamp": "RiskValue"
    },
    "marketID": {
        "numberOfDays_averageSpan": {
            "xDaysAverageVolatility": {
                "xDaysAverageLiquidity": "riskValue"
            }
        }
    }
}

```
For instance, in the following data we can see that for the market ID 0xc54d7acf14de29e0e5527cabd7a576506870346a78a11a6762e2cca66322ec41, the average risk level over the last 7 days for the 30 days average volatility and the 30 days average liquidity was 5.039145924635319.

```
{
    "protocolAverageHistory": {
        "1705334910014": 4.98,
        "1705248510014": 4.95,
        "1705162110014": 4.97,
        "1705075710014": 4.97,
        "1704989310014": 4.97,
        "1704902910014": 4.96,
        "1704816510014": 4.93,
        "1704730110014": 4.91
    },
    "0xc54d7acf14de29e0e5527cabd7a576506870346a78a11a6762e2cca66322ec41": {
        "7D_averageSpan": {
            "7": {
                "7": 4.9627383669524745,
                "30": 5.039145924635319,
                "180": 4.50958599294634
            },
            "30": {
                "7": 4.9627383669524745,
                "30": 5.039145924635319,
                "180": 4.50958599294634
            },
            "180": {
                "7": 4.9627383669524745,
                "30": 5.039145924635319,
                "180": 4.50958599294634
            }
        },
        "8D_averageSpan": {
            "7": {
                "7": 4.955997259013416,
                "30": 5.040050865143407,
                "180": 4.50687856248198
            },
            "30": {
                "7": 4.955997259013416,
                "30": 5.040050865143407,
                "180": 4.50687856248198
            },
            "180": {
                "7": 4.955997259013416,
                "30": 5.040050865143407,
                "180": 4.50687856248198
            }
        }
    }
}

```

## Other files
Other protocols specific files' structure is:
```
{
  "protocolAverageHistory": {
    "UnixTimeStamp": "RiskValue"
  },
  "VaultMarketWhatever": {
    "Collateral_marketID?": {
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
