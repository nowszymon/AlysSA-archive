# AlysSA v0.1b


*NextDayReturn(Data, DataColumn, PreviousDay, MoreLess='More', Days=1): Calculate the day return on the day after daily return of PreviousDay value occured.

*BarsInaRow(Data, DataColumn): How many times there was a X number of down/up candles in a row.

*SeasonalityPattern(DataFrame, Column): Plot chart of seasonality pattern with the actual year data. Always delete automatically first year.

*MonthsofExceededMove(DataFrame, Column, Window, Level, MoreLess='Less'): Calculate number of months when symbol exceeded Level move in Window days.

*DownloadFromStooq(Symbol, Interval, Open=False, High=False, Low=False, Close=True, Volume=False): Download data from stooq.plot

*SeasonalityDaily(DataFrame, Column): Calculate mean of daily returns and percentage of days with plus/minus daily returns.