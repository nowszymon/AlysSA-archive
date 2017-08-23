import pandas as pd

"""
NextDayReturn(Data, DataColumn, PreviousDay, MoreLess='More', Days=1): Calculate the day return on the day after daily return of PreviousDay value occured.
BarsInaRow(Data, DataColumn): How many times there was a X number of down/up candles in a row.
"""

def PercentageChange(dataframe, column, change):
	"""
	dataframe: pandas Dataframe, column: column with data, change: output column
	"""
	dataframe[change] = ((dataframe[column]-dataframe[column].shift(1))/dataframe[column])
	return dataframe
	
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
	
	print(final['UP'].value_counts().sort_index())
	print()
	print(res.plot(kind="bar", figsize=(10,7), legend=False, title='Number of downward(-)/upward(+) candles in a row'))