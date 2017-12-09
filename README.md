# AlysSA v0.14b


* NextDayReturn(Data, DataColumn, PreviousDay, MoreLess='More', Days=1): Calculate the day return on the day after daily return of PreviousDay value occured.

* BarsInaRow(Data, DataColumn): How many times there was a X number of down/up candles in a row.

* SeasonalityPattern(DataFrame, Column): Plot chart of seasonality pattern with the actual year data. Always delete automatically first year.

* MonthsofExceededMove(DataFrame, Column, Window, Level, MoreLess='Less'): Calculate number of months when symbol exceeded Level move in Window days.

* DownloadFromStooq(Symbol, Interval, Open=False, High=False, Low=False, Close=True, Volume=False): Download data from stooq.plot

* SeasonalityDaily(DataFrame, Column): Calculate mean of daily returns and percentage of days with plus/minus daily returns.

* Rebase(DataFrame): Calucalte percent changes to the first value in a column.

* Fed_rates(): Load pickled df with details of FED funds rates 

* Plot_X_values_on_Y_dates(df, Column, Dates, fred_api, Range_start=-20, Range_end=60,Title='',PlotAll=False): Plot min and max boundaries for the average price action on selected days.	

* Bubbles(): Load pickled df with price bubbles data

* HeatmapMonthly(df, DateColumn, CloseColumn, Day=False): Heatmap of daily/monthly returns.

* CorrelationChart(df,col1,col2,window=20, CorrPrices=False):	Price/regression/correlation chart with basic statistics.

* SpreadChart(df,col1,col2): Spread chart.

* WordCloud(text,title='',language = 'English', additional_words=[''], UseStopwords = True): WordCloud from the text.

* CurrencyVsYieldSpreads(start_date='2015-01-01', symbol='all'): Major currencies vs. 10y yields spread.

* def NYSEMargin(): Plots NYSE Investor credit balances and SP500.