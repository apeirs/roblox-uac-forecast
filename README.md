# Up-and-Coming top-10 forecaster

Forecasts Roblox's Up-and-Coming chart, places 1 to 10, for every hour of a 72-hour window
(make.fun roblox-up-and-coming-forecast-01).

For each place and each hour of the window it scores every game on the last chart with a conditional logit:
the game's log-rank distance to the place (a kernel that widens with the horizon), its rank gain over the last
12 hours, how long it has been on the chart, and its player count against the chart's median. A "newcomer"
option, whose share grows with the horizon and the place, stands for games not on the last chart. The
coefficients come from a maximum-likelihood fit on pairs of archive reads 1 to 80 hours apart, and are refitted
at run time on the input history's own read pairs (ridge toward the stored fit). The forecast for a place is the
softmax averaged over the window's hours. Every chart game gets a row, and the newcomer share is left unnamed,
so the scorer spreads it over games that join the chart.

Run (Python 3.13, standard library only, about 5 s):

    mkdir -p out && MAKEFUN_INPUT_DIR=path/to/input MAKEFUN_OUTPUT_DIR=out python3 main.py

Without the variables it reads /input and writes /output/forecast.json.
