import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

"""
AlysSA v0.13b

NextDayReturn(Data, DataColumn, PreviousDay, MoreLess='More', Days=1): Calculate the day return on the day after daily return of PreviousDay value occured.
BarsInaRow(Data, DataColumn): How many times there was a X number of down/up candles in a row.
SeasonalityPattern(DataFrame, Column): Plot chart of seasonality pattern with the actual year data. Always delete automatically first year.
MonthsofExceededMove(DataFrame, Column, Window, Level, MoreLess='Less'): Calculate number of months when symbol exceeded Level move in Window days.
DownloadFromStooq(Symbol, Interval, Open=False, High=False, Low=False, Close=True, Volume=False): Download data from stooq.plot
SeasonalityDaily(DataFrame, Column): Calculate mean of daily returns and percentage of days with plus/minus daily returns.
Rebase(DataFrame): Calucalte percent changes to the first value in a column.
Fed_rates(): Load pickled df with details of FED funds rates 
Plot_X_values_on_Y_dates(df, Column, Dates, fred_api, Range_start=-20, Range_end=60,Title='',PlotAll=False): Plot min and max boundaries for the average price action on selected days.	
Bubbles(): Load pickled df with price bubbles data
HeatmapMonthly(df, DateColumn, CloseColumn, Day=False): Heatmap of daily/monthly returns.
CorrelationChart(df,col1,col2,window=20, CorrPrices=False):	Price/regression/correlation chart with basic statistics.
SpreadChart(df,col1,col2): Spread chart.
"""



def ExtractWeekDay(DataFrame, Column):
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'WeekDay'] = DataFrame.loc[i, Column].weekday()
	return DataFrame

def ExtractDay(DataFrame, Column):
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'Day'] = DataFrame.loc[i, Column].day
	return DataFrame
	
def ExtractMonth(DataFrame, Column):
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'Month'] = DataFrame.loc[i, Column].month
	return DataFrame
	
def ExtractYears(DataFrame, Column):
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'Year'] = DataFrame.loc[i, Column].year
	return DataFrame

def CalculateRollingPct(DataFrame, Column, Window):
	for i in range(Window+1,len(DataFrame)):
		DataFrame.loc[i, 'PCT'] = (DataFrame.loc[i,Column]/DataFrame.loc[i-Window, Column])-1
	return DataFrame
	
def PercentageChange(DataFrame, Column, Change):
	"""
	dataframe: pandas Dataframe, column: column with data, change: output column
	"""
	for i in range(1,len(DataFrame)):
		DataFrame.loc[i,Change] = (DataFrame.loc[i,Column]/DataFrame.loc[i-1,Column])-1
	return DataFrame

def RenameDays(DataFrame, Column):
		
	for day in range(0,len(DataFrame)):	
		if DataFrame.loc[day,Column] == 0:
			DataFrame.loc[day,Column] ="Monday"
		elif DataFrame.loc[day,Column] == 1:
			DataFrame.loc[day,Column] = "Tuesday"
		elif DataFrame.loc[day,Column] == 2:
			DataFrame.loc[day,Column] = "Wednesday"
		elif DataFrame.loc[day,Column] == 3:
			DataFrame.loc[day,Column] = "Thursday"
		elif DataFrame.loc[day,Column] == 4:
			DataFrame.loc[day,Column] = "Friday"
		elif DataFrame.loc[day,Column] == 5:
			DataFrame.loc[day,Column] = "Saturday"
		elif DataFrame.loc[day,Column] == 6:
			DataFrame.loc[day,Column] = "Sunday"
	return DataFrame
				
def RenameMonths(DataFrame):
	DataFrame.rename(index={1.0:'January', 2.0:'February', 3.0:'March', 4.0:'April', 5.0:'May',6.0:'June',7.0:'July',8.0:'August',9.0:'September', 10.0:'October',11.0:'November',12.0:'December'}, inplace=True)
	return DataFrame
	
def DownloadFromStooq(Symbol, Interval, Part = False, Date_from = '2000-01-01', Date_to = '2100-01-01', Open=False, High=False, Low=False, Close=True, Volume=False):
	"""
	Download data from stooq.plot
	"""
	
	import datetime
	
	Date_f = Date_from.replace('-','')
	Date_t = Date_to.replace('-','')
	
	if Part == False:
		url = 'http://stooq.com/q/d/l/?s={}&i={}'.format(Symbol,Interval)
	else:
		url = 'http://stooq.com/q/d/l/?s={}&d1={}&d2={}&i=d'.format(Symbol,Date_f,Date_t)

	data = pd.read_csv(url)
	
	data['Date'] = data['Date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
	
	if Open == False:
		del data['Open']
	if High == False:
		del data['High']
	if Low == False:
		del data['Low']
	if Close == False:
		del data['Close']
	if (('Volume' in data.columns) & (Volume == False)) :
		del data['Volume']
		
	data.rename(columns={'Close':Symbol}, inplace=True)
	
	return data
	
def NextDayReturn(Data, DataColumn, PreviousDay, MoreLess='More', Days=1):

	"""
	Calculate the day return on the day after daily return of PreviousDay value occured.

	Data: Pandas DataFrame, DataColumn: Column with data, PreviousDay: Return on previous day, MoreLess: Operator to test, Days: 1 is next day, 2 are to days after event and so on
	"""
	
	values = PercentageChange(Data, DataColumn, 'Pct_Change')
	
	if MoreLess == 'Less':
		ind = values[values['Pct_Change']<PreviousDay]['Pct_Change'].index + Days
	else:
		ind = values[values['Pct_Change']>PreviousDay]['Pct_Change'].index + Days
			
	final = pd.DataFrame()
	
	for i in ind:
		try:
			final.loc[i, 'Pct_Change'] = values.loc[i,'Pct_Change']
		except:
			continue
		
		final.reset_index(inplace=True, drop=True)
	
	print(final.describe())
	print()
	print('Up days: {}'.format((final[final['Pct_Change']>0]['Pct_Change'].size)),' ', round(((final[final['Pct_Change']>0]['Pct_Change'].size)/((final[final['Pct_Change']>0]['Pct_Change'].size)+(final[final['Pct_Change']<0]['Pct_Change'].size)))*100,2),'%' )
	print('Down days: {}'.format((final[final['Pct_Change']<0]['Pct_Change'].size)),' ', round(((final[final['Pct_Change']<0]['Pct_Change'].size)/((final[final['Pct_Change']>0]['Pct_Change'].size)+(final[final['Pct_Change']<0]['Pct_Change'].size)))*100,2),'%' )
	print()
	print(final.plot(kind="hist", bins=30, figsize=(10,7)))
	
def BarsInaRow(Data, DataColumn):
	"""
	How many times there was a X number of down/up candles in a row.
	
	Data: Pandas DataFrame, DataColumn: Column with data
	"""
	
	values = PercentageChange(Data, DataColumn, 'PCT')
	
	indUP = Data[Data['PCT']>0].index
	resultsUP = pd.DataFrame()
	agg = 1
	for i in range(0,len(indUP)-1):
		if indUP[i+1] == indUP[i] + 1:
			agg += 1 
		else:
			if agg != 0:
				resultsUP.loc[i, 'UP'] = agg
			agg = 1
			
	indDOWN = Data[Data['PCT']<0].index
	resultsDOWN = pd.DataFrame()
	agg = -1
	for i in range(0,len(indDOWN)-1):
		if indDOWN[i+1] == indDOWN[i] + 1:
			agg += -1 
		else:
			if agg != 0:
				resultsDOWN.loc[i, 'UP'] = agg
			agg = -1		
	
	final = pd.DataFrame()
	final = resultsUP
	final = final.append(resultsDOWN)
	res = pd.DataFrame(final['UP'].value_counts())
	res.sort_index(inplace=True)

	anchored_text = AnchoredText('                AlysSA v0.13b\n  Algorithmic\'n\'Statistical Analysis\nSzymon Nowak (www.1do1.biz.pl)', loc=1)
		
	print(final['UP'].value_counts().sort_index())
	print()

	ax = res.plot(kind="bar", figsize=(12,8), rot=45, legend=False, title='Number of downward(-)/upward(+) candles in a row')
	ax.title.set_size(20)
	ax.add_artist(anchored_text)
	
def SeasonalityPattern(DataFrame, Column, title=''):
	"""
	Plot chart of seasonality pattern with the actual year data. Always delete automatically first year.
	
	DataFrame: Pandas DataFrame, Column: Column with data
	"""
	
	values = ExtractYears(DataFrame, 'Date')
	
	listofyears = DataFrame['Year'].unique()
	listofyears = listofyears[1:]
	
	final = pd.DataFrame()
	for i in listofyears:
		oneyear = pd.DataFrame(DataFrame[DataFrame['Year']==i][Column])
		oneyear.reset_index(inplace=True,drop=True)
		final[i] = oneyear
	
	path = pd.DataFrame()
	for i in listofyears:
		for x in range(0,len(final)):
			path.loc[x,i] = final.loc[x,i]/final.loc[0,i]-1
			
	plotdata = pd.DataFrame()
	for i in range(0,len(path)):
		plotdata.loc[i,'Projection'] = path.iloc[i,:-1].mean()					
	plotdata[listofyears[-1]] = path[listofyears[-1]]
	
	plotdata = plotdata[:-5]

	anchored_text = AnchoredText('                AlysSA v0.13b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=4)	
	ax = plotdata.plot(y=['Projection',listofyears[-1]], secondary_y=['Projection'],figsize=(12,8), title='Seasonality Pattern Projection ({} r = {}%)'.format(title,round(plotdata.corr().iloc[0,1]*100)))
	ax.set_xlabel('Days')
	ax.add_artist(anchored_text)
	
def SeasonalityDaily(DataFrame, Column):
	"""
	Calculate mean of daily returns and percentage of days with plus/minus daily returns.
	
	DataFrame: Pandas DataFrame, Column: Column with data
	"""
	
	ExtractWeekDay(DataFrame, 'Date')
	RenameDays(DataFrame, 'WeekDay')
	PercentageChange(DataFrame, Column, '{}_pct'.format(Column))
	
	for i in range(0,len(DataFrame)):
		DataFrame.loc[i,'Ones'] = 1
		if DataFrame.loc[i, '{}_pct'.format(Column)] > 0:
			DataFrame.loc[i,'More'] = 1
	
	Plus = 	(DataFrame['More'].groupby(DataFrame['WeekDay']).sum())/(DataFrame['Ones'].groupby(DataFrame['WeekDay']).sum())*100
	Minus = (DataFrame['Ones'].groupby(DataFrame['WeekDay']).sum()-DataFrame['More'].groupby(DataFrame['WeekDay']).sum())/(DataFrame['Ones'].groupby(DataFrame['WeekDay']).sum())*100

	Plus = Plus.to_frame()
	Minus = Minus.to_frame()
	
	results = pd.concat([Plus, Minus], axis=1)
	results.columns = ['Plus', 'Minus']
	
	print('Percentage of days with plus/minus daily returns (%)')
	print()
	print(results)
	print()
	print(DataFrame['{}_pct'.format(Column)].groupby(DataFrame['WeekDay']).mean().plot(kind='bar', figsize=(10,7), title='Mean of daily returns'))
	
def MonthsofExceededMove(DataFrame, Column, Window, Level, MoreLess='Less'):
	"""
	Calculate number of months when symbol exceeded Level move in Window days.
	
	DataFrame: pandas DataFrame, Column: Column with data, Window: Period of change, Level: Percentage move, MoreLess: More or Less
	"""
	
	CalculateRollingPct(DataFrame, Column, Window)
	ExtractMonth(DataFrame, 'Date')
	if MoreLess ==  'Less':
		final = DataFrame[DataFrame['PCT']<Level].groupby('Month').count()[Column]
	else:
		final = DataFrame[DataFrame['PCT']>Level].groupby('Month').count()[Column]
	RenameMonths(final)
	final.plot(kind='bar',figsize=(10, 7), rot=45, legend=False, title='Number of months when {} was down more than {}% in {} days. '.format(Column, Level*100, Window))
	
def Rebase(DataFrame):
	"""
	Calucalte percent changes to the first value in a column.
	"""
	
	return(DataFrame.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100)))
	
def Fed_rates():

	import pickle
	
	infile = open('data/fed_rates','rb')
	data = pickle.load(infile)
	infile.close()
	return(data)
	
def Plot_X_values_on_Y_dates(df, Column, Dates, fred_api, Range_start=-20, Range_end=60,Title='',PlotAll=False):
	
	"""
	Plot min and max boundaries for the average price action on selected days.
	
	df: pandas DataFrame with values,Column: column in df with values, Dates: pandas DataFrame with dates, fred_api = api key for Fred
	"""
	
	from fredapi import Fred
	import pylab as plt
	
	df.reset_index(inplace=True)
	
	indexes = []
	for i in Dates:
		index = df[df['Date']==i]['index']
		indexes.extend(index)
		
	dates_pd = pd.DataFrame(Dates)
	dates_pd.reset_index(inplace=True)
	
	final = pd.DataFrame()
	for i in range(len(dates_pd)):
		for x in range(Range_start,Range_end):
			final.loc[x,i] = df.loc[indexes[i]+x][Column]
			
	for date in final:
		for i in range(len(final)):
			if i == abs(Range_start):
				continue
			#base = final.iloc[abs(Range_start),date]
			final.iloc[i,date] = (final.iloc[i,date]/final.iloc[abs(Range_start),date]-1)*100		
	
	for date in final:
		final.iloc[abs(Range_start),date] = 0	
	
	final = final.rename(columns=(lambda x:Dates[x]))	
	
	final['average'] = ''
	for i in range(len(final)):
		final.iloc[i,-1] = final.iloc[i,:-1].mean()

	final['min'] = ''
	for i in range(len(final)):
		final.iloc[i,-1] = final.iloc[i,:-1].min()
    
	final['max'] = ''
	for i in range(len(final)):
		final.iloc[i,-1] = final.iloc[i,:-1].max()
		
	final['average'] = pd.to_numeric(final['average'])
	final['min'] = pd.to_numeric(final['min'])
	final['max'] = pd.to_numeric(final['max'])	
	final['zero'] = '0'
	final['zero'] = pd.to_numeric(final['zero'])
	
	if PlotAll == False:
		anchored_text = AnchoredText('                AlysSA v0.13b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=4)	
		ax = final.plot(y=['min'],figsize=(12,8), label='min',color='Red',title=Title)
		final['max'].plot(ax=ax,label='max',color='Green')
		final['average'].plot(ax=ax,label='average', linewidth=3)
		final['zero'].plot(ax=ax,color='Black',linestyle='dashed',linewidth=1,label='')
		ax.set_xlabel('Days')
		ax.set_ylabel('[%]')
		# ax1 = ax.twinx()
		# ax1.set_ylim(ax.get_ylim())
		plt.legend()
		ax.fill_between(final.index,final['max'],final['min'],alpha=0.3,color='Pink')
		ax.add_artist(anchored_text)
	else:
		anchored_text = AnchoredText('                AlysSA v0.13b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=4)
		ax = final.plot(figsize=(12,8),legend=False, title=Title)
		ax.set_xlabel('Days')
		ax.set_ylabel('[%]')
		ax.add_artist(anchored_text)
		
def Bubbles():

	import pickle
	
	infile = open('data/bubbles','rb')
	data = pickle.load(infile)
	infile.close()
	return(data)
	
def HeatmapMonthly(df, DateColumn, CloseColumn, Day=False):

	"""
	Heatmap of daily/monthly returns.
	
	df: pandas DataFrame, DateColumn: column with dates, CloseColumn: column with close prices, Day=False: monthly returns
	"""

	import seaborn as sns
	
	df['Year'] = pd.DatetimeIndex(df[DateColumn]).year
	df['Month'] = pd.DatetimeIndex(df[DateColumn]).month
	df['Day'] = pd.DatetimeIndex(df[DateColumn]).weekday
	
	df['Return'] = df[CloseColumn].pct_change()*100
	
	if Day == False:
		avg = df[['Month', 'Return']].groupby(['Month'],as_index=False).mean()
	else:
		avg = df[['Day', 'Return']].groupby(['Day'],as_index=False).mean()
		
	avg['Year'] = 'Avg'
	new = pd.concat([df,avg])
	
	if Day == False:
		final = new.pivot_table(columns='Month',index='Year',values='Return')
	else:
		final = new.pivot_table(columns='Day',index='Year',values='Return')
		final = final.rename(columns={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday'})
		
	fig, ax = plt.subplots(figsize=(15,12))
	g = sns.heatmap(final, annot=True, cmap='PiYG')
	ax.invert_yaxis()	

def CorrelationChart(df,col1,col2,window=20, CorrPrices=False):

	"""
	Price/regression/correlation chart with basic statistics.
	
	df: pandas DataFrame, col1: Column with first instrument data, col2: Column with second instrument data, window: rolling correlation winow, CorrPrices=False: Correllate returns instead of prices
	"""
		
	import seaborn as sns
	
	pct = df
	pct = pct.pct_change()
	
	pct['corr'] = pct[col1].rolling(window).corr(pct[col2])
	last_corr = pct['corr'][-1]*100
	full_corr = df.corr()
	full_corr = full_corr.iloc[0,1]*100
	
	print('-'*40)
	print('Returns:')
	print('-'*40)
	
	print(pct.describe())

	print('-'*40)
	print('Prices:')
	print('-'*40)	
	
	print(df.describe())

	fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False, figsize=(15,12), gridspec_kw = {'height_ratios':[3, 1]})	
	ax1.plot(df[col1], color='orange', label='{}'.format(col1))
	axy = ax1.twinx()
	axy.plot(df[col2], label='{}'.format(col2))
	ax2.plot(pct['corr'], linewidth=0.1)
	
	ax1.set_title('{} vs. {} (r={}%)'.format(col1,col2,"%.2f" % full_corr), fontsize=15)
	ax2.set_title('Correlation({}) = {}%'.format(window,"%.2f" % last_corr), fontsize=15)
	
	ax1.grid(False)
	ax1.xaxis.grid(True)
	axy.grid(False)

	ax1.legend(loc=2)
	axy.legend()

	ax1.set_ylabel('{}'.format(col1))
	axy.set_ylabel('{}'.format(col2))

	ax2.fill_between(pct.index.values, 0, pct['corr'], where=pct['corr']>0, interpolate=True, color='green')
	ax2.fill_between(pct.index.values, 0, pct['corr'], where=pct['corr']<0, interpolate=True, color='red')

	anchored_text = AnchoredText('                AlysSA v0.13b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
	ax1.add_artist(anchored_text)
	
	if CorrPrices==False:
		jointplot = sns.jointplot(col1,col2,pct,kind='reg',size=15)
	else: 
		jointplot = sns.jointplot(col1,col2,df,kind='reg',size=15)	

def SpreadChart(df,col1,col2):

	"""
	Spread chart.
	
	df: pandas DataFrame, col1: Column with first instrument data, col2: Column with second instrument data
	"""
	
	import seaborn as sns

	df = Rebase(df)
	
	df['spread'] = df[col1] - df[col2]
	
	fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False, figsize=(15,12), gridspec_kw = {'height_ratios':[3, 1]})
	ax1.plot(df[col1],label='{}'.format(col1))
	ax1.plot(df[col2],label='{}'.format(col2))	
	ax2.plot(df['spread'], linewidth=0.1)

	ax1.set_title('{} vs. {}'.format(col1,col2),fontsize=15)
	ax2.set_title('Spread',fontsize=15)
	
	ax1.legend(loc=2)

	ax1.grid(False)
	ax1.xaxis.grid(True)
	ax2.grid(False)
	ax2.xaxis.grid(True)

	ax1.fill_between(df.index.values,df['XAUUSD'],df['CA_C.F'],where=df['XAUUSD']>df['CA_C.F'],interpolate=True,color=sns.xkcd_rgb["denim blue"], alpha=0.2)
	ax1.fill_between(df.index.values,df['XAUUSD'],df['CA_C.F'],where=df['XAUUSD']<df['CA_C.F'],interpolate=True,color=sns.xkcd_rgb["medium green"], alpha=0.2)
					 
	ax2.fill_between(df.index.values,0,df['spread'],where=df['spread']>0,interpolate=True,color=sns.xkcd_rgb["denim blue"], alpha=0.7)
	ax2.fill_between(df.index.values,0,df['spread'],where=df['spread']<0,interpolate=True,color=sns.xkcd_rgb["medium green"], alpha=0.7)

	
	anchored_text = AnchoredText('                AlysSA v0.13b\n  Algorithmic\'n\'Statistical Analysis\n       www.szymonnowak.com', loc=3)
	ax1.add_artist(anchored_text)
	