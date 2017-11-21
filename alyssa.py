import pandas as pd
from matplotlib.offsetbox import AnchoredText

"""
AlysSA v0.11b

NextDayReturn(Data, DataColumn, PreviousDay, MoreLess='More', Days=1): Calculate the day return on the day after daily return of PreviousDay value occured.
BarsInaRow(Data, DataColumn): How many times there was a X number of down/up candles in a row.
SeasonalityPattern(DataFrame, Column): Plot chart of seasonality pattern with the actual year data. Always delete automatically first year.
MonthsofExceededMove(DataFrame, Column, Window, Level, MoreLess='Less'): Calculate number of months when symbol exceeded Level move in Window days.
DownloadFromStooq(Symbol, Interval, Open=False, High=False, Low=False, Close=True, Volume=False): Download data from stooq.plot
SeasonalityDaily(DataFrame, Column): Calculate mean of daily returns and percentage of days with plus/minus daily returns.
Rebase(DataFrame): Calucalte percent changes to the first value in a column.
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
	
def DownloadFromStooq(Symbol, Interval, Open=False, High=False, Low=False, Close=True, Volume=False):
	"""
	Download data from stooq.plot
	"""
	
	import datetime
	url = 'http://stooq.com/q/d/l/?s={}&i={}'.format(Symbol,Interval)
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
	if Volume == False:
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

	anchored_text = AnchoredText('                AlysSA v0.1b\n  Algorithmic\'n\'Statistical Analysis\nSzymon Nowak (www.1do1.biz.pl)', loc=1)
		
	print(final['UP'].value_counts().sort_index())
	print()

	ax = res.plot(kind="bar", figsize=(12,8), rot=45, legend=False, title='Number of downward(-)/upward(+) candles in a row')
	ax.title.set_size(20)
	ax.add_artist(anchored_text)
	
def SeasonalityPattern(DataFrame, Column):
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
			path.loc[x,i] = final.loc[x,i]/final.loc[0,i]*100
			
	plotdata = pd.DataFrame()
	for i in range(0,len(path)):
		plotdata.loc[i,'Projection'] = path.iloc[i,:-1].mean()					
	plotdata[listofyears[-1]] = path[listofyears[-1]]
	
	anchored_text = AnchoredText('                AlysSA v0.1b\n  Algorithmic Statistical Analysis\nSzymon Nowak (www.1do1.biz.pl)', loc=4)	
	ax = plotdata.plot(figsize=(10,7), title='Seasonality Pattern Projection (r = {}%)'.format(round(plotdata.corr().iloc[0,1]*100)))
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
	