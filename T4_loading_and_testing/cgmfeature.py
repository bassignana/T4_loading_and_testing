import pandas as pd
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

"""
    cgmquantify package
    Description:
    The cgmquantify package is a comprehensive library for computing metrics from continuous glucose monitors.
    Requirements:
    pandas, datetime, numpy, matplotlib, statsmodels
    Functions:
    importdexcom(): Imports data from Dexcom continuous glucose monitor devices
    interdaycv(): Computes and returns the interday coefficient of variation of glucose
    interdaysd(): Computes and returns the interday standard deviation of glucose
    intradaycv(): Computes and returns the intraday coefficient of variation of glucose 
    intradaysd(): Computes and returns the intraday standard deviation of glucose 
    TIR(): Computes and returns the time in range
    TOR(): Computes and returns the time outside range
    PIR(): Computes and returns the percent time in range
    POR(): Computes and returns the percent time outside range
    MGE(): Computes and returns the mean of glucose outside specified range
    MGN(): Computes and returns the mean of glucose inside specified range
    MAGE(): Computes and returns the mean amplitude of glucose excursions
    J_index(): Computes and returns the J-index
    LBGI(): Computes and returns the low blood glucose index
    HBGI(): Computes and returns the high blood glucose index
    ADRR(): Computes and returns the average daily risk range, an assessment of total daily glucose variations within risk space
    MODD(): Computes and returns the mean of daily differences. Examines mean of value + value 24 hours before
    CONGA24(): Computes and returns the continuous overall net glycemic action over 24 hours
    GMI(): Computes and returns the glucose management index
    eA1c(): Computes and returns the American Diabetes Association estimated HbA1c
    summary(): Computes and returns glucose summary metrics, including interday mean glucose, interday median glucose, interday minimum glucose, interday maximum glucose, interday first quartile glucose, and interday third quartile glucose
    plotglucosesd(): Plots glucose with specified standard deviation lines
    plotglucosebounds(): Plots glucose with user-defined boundaries
    plotglucosesmooth(): Plots smoothed glucose plot (with LOWESS smoothing)
            
"""

def importdexcom(filename):
    """
        Imports data from Dexcom continuous glucose monitor devices
        Args:
            filename (String): path to file
        Returns:
            (pd.DataFrame): dataframe of data with Time, Glucose, and Day columns
    """
    data = pd.read_csv(filename) 
    df = pd.DataFrame()
    df['Time'] = data['Timestamp (YYYY-MM-DDThh:mm:ss)']
    df['Glucose'] = pd.to_numeric(data['Glucose Value (mg/dL)'])
    df.drop(df.index[:12], inplace=True)
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%dT%H:%M:%S')
    df['Day'] = df['Time'].dt.date
    df = df.reset_index()
    return df


def importfreestylelibre(filename):
    """
        Imports data from Abbott FreeStyle Libre continuous glucose monitor devices
        Args:
            filename (String): path to file
        Returns:
            (pd.DataFrame): dataframe of data with Time, Glucose, and Day columns
    """
    data = pd.read_csv(filename, header=1, parse_dates=['Device Timestamp'])
    df = pd.DataFrame()

    historic_id = 0

    df['Time'] = data.loc[data['Record Type'] == historic_id, 'Device Timestamp']
    df['Glucose'] = pd.to_numeric(data.loc[data['Record Type'] == historic_id, 'Historic Glucose mg/dL'])
    df['Day'] = df['Time'].dt.date
    return df


def interdaycv(df):
    """
        Computes and returns the interday coefficient of variation of glucose
        The higher the coefficient of variation, the greater the level of dispersion around the mean. It is generally expressed as a percentage. The lower the value of the coefficient of variation, the more precise the estimate.
        più bassa è, meglio è, anche in relazione alla magnitudine del valore medio: 50/100 >> 50/1000
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            cvx (float): interday coefficient of variation averaged over all days
            
    """
    try:
        cvx = (np.std(df['Glucose']) / (np.mean(df['Glucose'])))*100
        return cvx
    except:
        return float('nan')
def interdaysd(df):
    """
        Computes and returns the interday standard deviation of glucose
         it is a measure of the average distance between the values of the data in the set and the mean.
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            interdaysd (float): interday standard deviation averaged over all days
            
    """
    try:
        interdaysd = np.std(df['Glucose'])
        return interdaysd
    except:
        return float('nan')


def intradaycv(df):
    """
        Computes and returns the intraday coefficient of variation of glucose 
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            intradaycv_mean (float): intraday coefficient of variation averaged over all days
            intradaycv_medan (float): intraday coefficient of variation median over all days
            intradaycv_sd (float): intraday coefficient of variation standard deviation over all days
            
    """
    intradaycv = []
    for i in pd.unique(df['Day']):
        intradaycv.append(interdaycv(df[df['Day']==i]))
    
    try:
        intradaycv_mean = np.mean(intradaycv)
    except:
        intradaycv_mean = float('nan')
    
    try:
        intradaycv_median = np.median(intradaycv)
    except:
        intradaycv_median = float('nan')

    try:
        intradaycv_sd = np.std(intradaycv)
    except:
        intradaycv_sd = float('nan')

    return intradaycv_mean, intradaycv_median, intradaycv_sd


def intradaysd(df):
    """
        Computes and returns the intraday standard deviation of glucose 
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            intradaysd_mean (float): intraday standard deviation averaged over all days
            intradaysd_medan (float): intraday standard deviation median over all days
            intradaysd_sd (float): intraday standard deviation standard deviation over all days
            
    """
    intradaysd =[]

    for i in pd.unique(df['Day']):
        intradaysd.append(np.std(df[df['Day']==i].drop('Time', axis = 1)))
    
    try:
        intradaysd_mean = np.mean(intradaysd)
    except:
        intradaysd_mean = float('nan')

    try:
        intradaysd_median = np.median(intradaysd)
    except:
        intradaysd_median = float('nan')

    try:
        intradaysd_sd = np.std(intradaysd)
    except:
        intradaysd_sd = float('nan')

    return intradaysd_mean, intradaysd_median, intradaysd_sd

def TIR(df, sd=1, sr=5):
    """
        Computes and returns the time in range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            TIR (float): time in range, units=minutes
            
    """
    try:
        up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
        dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
        TIR = len(df[(df['Glucose']<= up) & (df['Glucose']>= dw)])*sr 
        return TIR
    except:
        return float('nan')
    
def TOR(df, sd=1, sr=5):
    """
        Computes and returns the time outside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing  range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            TOR (float): time outside range, units=minutes
            
    """
    try:
        up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
        dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
        TOR = len(df[(df['Glucose']>= up) | (df['Glucose']<= dw)])*sr
        return TOR
    except:
        return float('nan')

def POR(df, sd=1, sr=5):
    """
        Computes and returns the percent time outside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            POR (float): percent time outside range, units=%
            
    """
    try:
        up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
        dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
        TOR = len(df[(df['Glucose']>= up) | (df['Glucose']<= dw)])*sr
        POR = (TOR/(len(df)*sr))*100
        return POR
    except:
        return float('nan')

def PIR(df, sd=1, sr=5):
    """
        Computes and returns the percent time inside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            PIR (float): percent time inside range, units=%
            
    """
    try:
        up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
        dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
        TIR = len(df[(df['Glucose']<= up) & (df['Glucose']>= dw)])*sr
        PIR = (TIR/(len(df)*sr))*100
        return PIR
    except:
        return float('nan')

def MGE(df, sd=1):
    #TODO: non sarebbe più sensato calcolare la media oltre la soglia superiore e separatamente la media oltre la soglia inferiore? cosa mi dice questa misura?
    """
        Computes and returns the mean of glucose outside specified range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
        Returns:
            MGE (float): the mean of glucose excursions (outside specified range)
            
    """
    try:
        up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
        dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
        MGE = np.mean(df[(df['Glucose']>= up) | (df['Glucose']<= dw)])
        return MGE
    except:
        return float('nan')

def MGN(df, sd=1):
    """
        Computes and returns the mean of glucose inside specified range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
        Returns:
            MGN (float): the mean of glucose excursions (inside specified range)
            
    """
    try:
        up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
        dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])
        MGN = np.mean(df[(df['Glucose']<= up) & (df['Glucose']>= dw)])
        return MGN
    except:
        return float('nan')
    
def MAGE(df, std=1):
    """
        Computes and returns the mean amplitude of glucose excursions
        
        The mean amplitude of glycemic excursion (MAGE) is the mean of blood glucose values exceeding one SD from the 24‐hour mean blood glucose and is used as an index of glycemic variability. According to the scientific literature, the value of MAGE in patients without DM are nearly 30 to 40 mg/dL and the cutoff value of MAGE for cardiovascular events are considered nearly 60 to 70 mg/dL.
        
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
        Returns:
            MAGE (float): the mean amplitude of glucose excursions 
        Refs:
            Sneh Gajiwala: https://github.com/snehG0205/NCSA_genomics/tree/2bfbb87c9c872b1458ef3597d9fb2e56ac13ad64
            
    """
    try:
        #extracting glucose values and incdices
        glucose = df['Glucose'].tolist()
        ix = [1*i for i in range(len(glucose))]
        stdev = std
    
        # local minima & maxima
        a = np.diff(np.sign(np.diff(glucose))).nonzero()[0] + 1
        # local min
        valleys = (np.diff(np.sign(np.diff(glucose))) > 0).nonzero()[0] + 1
        # local max
        peaks = (np.diff(np.sign(np.diff(glucose))) < 0).nonzero()[0] + 1
        # +1 -- diff reduces original index number
    
        #store local minima and maxima -> identify + remove turning points
        excursion_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
        k=0
        df.reset_index(drop = True, inplace = True)
        for i in range(len(peaks)):
            excursion_points.loc[k] = [peaks[i]] + [df['Time'][k]] + [df['Glucose'][k]] + ["P"]
            k+=1
    
        for i in range(len(valleys)):
            excursion_points.loc[k] = [valleys[i]] + [df['Time'][k]] + [df['Glucose'][k]] + ["V"]
            k+=1
    
        excursion_points = excursion_points.sort_values(by=['Index'])
        excursion_points = excursion_points.reset_index(drop=True)
    
    
        # selecting turning points
        turning_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
        k=0
        for i in range(stdev,len(excursion_points.Index)-stdev):
            positions = [i-stdev,i,i+stdev]
            for j in range(0,len(positions)-1):
                if(excursion_points.Type[positions[j]] == excursion_points.Type[positions[j+1]]):
                    if(excursion_points.Type[positions[j]]=='P'):
                        if excursion_points.Glucose[positions[j]]>=excursion_points.Glucose[positions[j+1]]:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                            k+=1
                        else:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                            k+=1
                    else:
                        if excursion_points.Glucose[positions[j]]<=excursion_points.Glucose[positions[j+1]]:
                            turning_points.loc[k] = excursion_points.loc[positions[j]]
                            k+=1
                        else:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                            k+=1
    
        if len(turning_points.index)<10:
            turning_points = excursion_points.copy()
            excursion_count = len(excursion_points.index)
        else:
            excursion_count = len(excursion_points.index)/2
    
    
        turning_points = turning_points.drop_duplicates(subset= "Index", keep= "first")
        turning_points=turning_points.reset_index(drop=True)
        excursion_points = excursion_points[excursion_points.Index.isin(turning_points.Index) == False]
        excursion_points = excursion_points.reset_index(drop=True)
    
        # calculating MAGE
        mage = turning_points.Glucose.sum()/excursion_count
    
        return round(mage,3)
    except:
        return float('nan')



def J_index(df): #TODO controllare che questa sia l'implementazione giusta per il CGM
    """
        Computes and returns the J-index
        più basso è il valore, meglio è
        
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            J (float): J-index of glucose
            
    """
    try:
        J = 0.001*((np.mean(df['Glucose'])+np.std(df['Glucose']))**2)
        return J
    except:
        return float('nan')

def LBGI_HBGI(df):
    """
    misura di richio di eventi di ipo o iperglicemia, più basso è meglio è
        Connecter function to calculate rh and rl, used for ADRR function
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            LBGI (float): Low blood glucose index
            HBGI (float): High blood glucose index
            rl (float): See calculation of LBGI
            rh (float): See calculation of HBGI
            
    """
    try:
        f = ((np.log(df['Glucose'])**1.084) - 5.381)
        rl = []
        for i in f: 
            if (i <= 0):
                rl.append(22.77*(i**2))
            else:
                rl.append(0)
    
        LBGI = np.mean(rl)
    
        rh = []
        for i in f: 
            if (i > 0):
                rh.append(22.77*(i**2))
            else:
                rh.append(0)
    
        HBGI = np.mean(rh)
        
        return LBGI, HBGI, rh, rl
    except:
        return float('nan')


def LBGI(df):
    """
        Computes and returns the low blood glucose index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            LBGI (float): Low blood glucose index
            
    """
    try:
        f = ((np.log(df['Glucose'])**1.084) - 5.381)
        rl = []
        for i in f: 
            if (i <= 0):
                rl.append(22.77*(i**2))
            else:
                rl.append(0)
    
        LBGI = np.mean(rl)
        return LBGI
    except:
        return float('nan')


def HBGI(df):
    """
        Computes and returns the high blood glucose index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            HBGI (float): High blood glucose index
            
    """
    f = ((np.log(df['Glucose'])**1.084) - 5.381)
    rh = []
    for i in f: 
        if (i > 0):
            rh.append(22.77*(i**2))
        else:
            rh.append(0)

    HBGI = np.mean(rh)
    return HBGI

def ADRR(df):
    """
     low risk, 0–19; moderate risk, 20–40; and high risk, 40 and above
     
        Computes and returns the average daily risk range, an assessment of total daily glucose variations within risk space
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            ADRRx (float): average daily risk range
            
    """
    try:
        ADRRl = []
        for i in pd.unique(df['Day']):
            LBGI, HBGI, rh, rl = LBGI_HBGI(df[df['Day']==i])
            LR = np.max(rl)
            HR = np.max(rh)
            ADRRl.append(LR+HR)
        
        ADRRx = np.mean(ADRRl)
        return ADRRx
    except:
        return float('nan')

def uniquevalfilter(df, value):
    """
        Supporting function for MODD and CONGA24 functions
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            value (datetime): time to match up with previous 24 hours
        Returns:
            MODD_n (float): Best matched with unique value, value
            
    """
    try:
        xdf = df[df['Minfrommid'] == value]
        n = len(xdf)
        diff = abs(xdf['Glucose'].diff())
        MODD_n = np.nanmean(diff)
        return MODD_n
    except:
        return float('nan')
        
def MODD(df): #TODO: i can try this concept binning multiple minutes togheather
    """
        mean of all valid absolute value differences between glucose concentrations measured at the same time of day on two consecutive days
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Requires:
            uniquevalfilter (function)
        Returns:
            MODD (float): Mean of daily differences
            
    """
    try:
        df['Timefrommidnight'] =  df['Time'].dt.time
        lists=[]
        for i in range(0, len(df['Timefrommidnight'])):
            lists.append(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[0:2])*60 + int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[3:5]) + round(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[6:9])/60))
        df['Minfrommid'] = lists
        df = df.drop(columns=['Timefrommidnight'])
        
        #Calculation of MODD and CONGA:
        MODD_n = []
        uniquetimes = df['Minfrommid'].unique()
    
        for i in uniquetimes:
            MODD_n.append(uniquevalfilter(df, i))
        
        #Remove zeros from dataframe for calculation (in case there are random unique values that result in a mean of 0)
        MODD_n[MODD_n == 0] = np.nan
        
        MODD = np.nanmean(MODD_n)
        return MODD
    except:
        return float('nan')

def CONGA24(df):
    """
        Computes and returns the continuous overall net glycemic action over 24 hours
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Requires:
            uniquevalfilter (function)
        Returns:
            CONGA24 (float): continuous overall net glycemic action over 24 hours
            
    """
    try:
        df['Timefrommidnight'] =  df['Time'].dt.time
        lists=[]
        for i in range(0, len(df['Timefrommidnight'])):
            lists.append(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[0:2])*60 + int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[3:5]) + round(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[6:9])/60))
        df['Minfrommid'] = lists
        df = df.drop(columns=['Timefrommidnight'])
        
        #Calculation of MODD and CONGA:
        MODD_n = []
        uniquetimes = df['Minfrommid'].unique()
    
        for i in uniquetimes:
            MODD_n.append(uniquevalfilter(df, i))
        
        #Remove zeros from dataframe for calculation (in case there are random unique values that result in a mean of 0)
        MODD_n[MODD_n == 0] = np.nan
        
        CONGA24 = np.nanstd(MODD_n)
        return CONGA24
    except:
        return float('nan')

def GMI(df):
    """
        Computes and returns the glucose management index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            GMI (float): glucose management index (an estimate of HbA1c)
            
    """
    try:
        GMI = 3.31 + (0.02392*np.mean(df['Glucose']))
        return GMI
    except:
        return float('nan')
    
def eA1c(df):
    """
        Computes and returns the American Diabetes Association estimated HbA1c
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            eA1c (float): an estimate of HbA1c from the American Diabetes Association
            
    """
    try:
        eA1c = (46.7 + np.mean(df['Glucose']))/ 28.7 
        return eA1c
    except:
        return float('nan')
    
def summary(df): 
    """
        Computes and returns glucose summary metrics
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            meanG (float): interday mean of glucose
            medianG (float): interday median of glucose
            minG (float): interday minimum of glucose
            maxG (float): interday maximum of glucose
            Q1G (float): interday first quartile of glucose
            Q3G (float): interday third quartile of glucose
            
    """
    try:
        meanG = np.nanmean(df['Glucose'])
    except:
        meanG = float('nan')

    try:
        medianG = np.nanmedian(df['Glucose'])    
    except:
        medianG = float('nan')

    try:
        minG = np.nanmin(df['Glucose'])
    except:
        minG = float('nan')

    try:
        maxG = np.nanmax(df['Glucose'])
    except:
        maxG = float('nan')

    try:
        Q1G = np.nanpercentile(df['Glucose'], 25)
    except:
        Q1G = float('nan')

    try:
        Q3G = np.nanpercentile(df['Glucose'], 75)
    except:
        Q3G = float('nan')

    return meanG, medianG, minG, maxG, Q1G, Q3G

def plotglucosesd(df, sd=1, size=15):
    """
        Plots glucose with specified standard deviation lines
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation lines to plot (default=1)
            size (integer): font size for plot (default=15)
        Returns:
            plot of glucose with standard deviation lines
            
    """
    glucose_mean = np.mean(df['Glucose'])
    up = np.mean(df['Glucose']) + sd*np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd*np.std(df['Glucose'])

    plt.figure(figsize=(20,5))
    plt.rcParams.update({'font.size': size})
    plt.plot(df['Time'], df['Glucose'], '.', color = '#1f77b4')
    plt.axhline(y=glucose_mean, color='red', linestyle='-')
    plt.axhline(y=up, color='pink', linestyle='-')
    plt.axhline(y=dw, color='pink', linestyle='-')
    plt.ylabel('Glucose')
    plt.show()

def plotglucosebounds(df, upperbound = 180, lowerbound = 70, size=15):
    """
        Plots glucose with user-defined boundaries
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            upperbound (integer): user defined upper bound for glucose line to plot (default=180)
            lowerbound (integer): user defined lower bound for glucose line to plot (default=70)
            size (integer): font size for plot (default=15)
        Returns:
            plot of glucose with user defined boundary lines
            
    """
    plt.figure(figsize=(20,5))
    plt.rcParams.update({'font.size': size})
    plt.plot(df['Time'], df['Glucose'], '.', color = '#1f77b4')
    plt.axhline(y=upperbound, color='red', linestyle='-')
    plt.axhline(y=lowerbound, color='orange', linestyle='-')
    plt.ylabel('Glucose')
    plt.show()

def plotglucosesmooth(df, size=15):
    """
        Plots smoothed glucose plot (with LOWESS smoothing)
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            size (integer): font size for plot (default=15)
        Returns:
            LOWESS-smoothed plot of glucose
            
    """
    filteres = lowess(df['Glucose'], df['Time'], is_sorted=True, frac=0.025, it=0)
    filtered = pd.to_datetime(filteres[:,0], format='%Y-%m-%dT%H:%M:%S') 
    
    plt.figure(figsize=(20,5))
    plt.rcParams.update({'font.size': size})
    plt.plot(df['Time'], df['Glucose'], '.')
    plt.plot(filtered, filteres[:,1], 'r')
    plt.ylabel('Glucose')
    plt.show()