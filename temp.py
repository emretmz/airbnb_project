import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pandas.plotting import table
from matplotlib.colors import ListedColormap

"""Q1: How to get the data?"""
url=r"C:\Users\conko\.spyder-py3\airbnb\airbnb.csv"
airbnb=pd.read_csv(url,sep=",")
print(airbnb.head(5))
    

"""Q2: What is the shape of the data? Column and row numbers..."""
print(airbnb.shape[0]) #row numbers
print(airbnb.shape[1]) #column numbers
    
"""Q3: Get basic information about data"""
print(airbnb.info())
    
"""Q4: Get the number of columns"""
print(len(airbnb.columns))
    
"""Q5: Get the index info of data"""
print(airbnb.index)#starts with 0 and stops at 7833',increases 1 by 1
    
"""Q6: Get the last 10 rows"""
print(airbnb.tail(5))#last 10 row
    
"""Q7: Get the biggest hots/hotels by bed numbers."""
air_bed=airbnb.groupby(by="host_name").sum().sort_values(by="beds",ascending=False).head(5) 
    
"""Q8: Get the highest prices in the table."""
air_zcode=airbnb.groupby(by="zipcode").sum().sort_values(by="price",ascending=False).head(5) 
    
"""Q9: Get the highest capacities hosts/hotels in the table."""
air_room=airbnb.groupby(by="zipcode").sum().sort_values(by="bedrooms",ascending=False).head(5)
    
"""Q10: Get the host names of the highest scored in the table"""
air_name=airbnb.groupby(by="host_name").sum().sort_values(by="review_scores_value",ascending=False).head(5)
    
"""Q:11 How many beds stay totally in Amsterdam which are recorded in airbnb?"""
print(airbnb.beds.sum())
    
"""Turn the item price into a float"""
print(airbnb["price"].dtype)
#airbnb[airbnb.price.astype(float)]
airbnb['price'].dtypes
    
"""Turn the item zipcode into a string"""
airbnb['zipcode']=airbnb['zipcode'].convert_dtypes()
airbnb['zipcode'].dtypes
    
"""Q:12 What is the average price per bed?"""
print(airbnb.groupby(by="host_name").sum().mean()["price"])
    
"""Data types of all columns """
print(airbnb.dtypes)
    
"""Get the number of the unique prices"""
print(airbnb.price.unique())
print(len(airbnb.bedrooms.unique()))
    
"""Q:13 What is the top 5 host's bed numbers?"""
print(airbnb.bedrooms.value_counts().head(5))
  
"""Q:14 Describe the data and get the measures of the central tendency"""
print(airbnb.describe())
print(airbnb.describe(include="all"))
    
"""Q:15 Get the measures of the central tendency"""
print(airbnb["bedrooms"].describe())
    
"""Q:16 What is the mean of minimum nights in the hosts?"""
print(airbnb["minimum_nights"].mean())
    
"""Q:17 What is the last 5 minimum nights?"""
print(airbnb.minimum_nights.value_counts().tail())
    
"""Q:18 What is the least realized price?"""
print(airbnb.price.value_counts().tail())
    
"""Q:19 What is the common hosts which have more than 50$ prices?"""
print(airbnb[airbnb.price>50].nunique())
    
"""Q:20 How many rooms cost more than 50$?"""
print(airbnb["price"].dtype) 
print(airbnb[airbnb["price"]>100].nunique())
    
"""get only price and host_name columns"""
name_price=airbnb[["host_name","price"]]
"""sort top 5 price values with..."""
print(airbnb.sort_values(by="price",ascending=True).head(5))
"""sort the top 5 host_name and zipcode columns..."""
print(airbnb.sort_values(["host_name","zipcode"],ascending=(True,True)).head(5))
    
"""Q:21 What is the quantity of the most expensive hosting ordered?"""
print(airbnb.sort_values(by="price",ascending=False).head(1))
    
"""get a specific host_name's rows"""
print(airbnb[airbnb["host_name"]=="Ozlem"])
    
"""Q:22 What is the total number of the non-unique zipcodes?"""
print(airbnb["zipcode"].nunique())
    
"""Q:23 Which hosts are located in 1054 zipcode?"""
print(airbnb[airbnb["zipcode"].str.startswith('1054')])
    
"""get the first 7 columns"""
print(airbnb.iloc[:,:7].head(5))#
"""get the last 3 columns"""
print(airbnb.iloc[:,:-3].head(5))
     
"""Q:24 Get the price rows which have these host names Daniel.Brittta and Ozlem"""
result=airbnb.loc[airbnb["host_name"].isin(["Daniel","Britta","Ozlem"]),["host_name","price"]]
    
"""Q:25 Visualize the prices and review scores value of first 100 rows columns."""
tbl=airbnb.loc[:,("host_id","review_scores_value","price")].head(100)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(tbl['price'],tbl['review_scores_value'])
plt.show()
    
"""histogram of roomtype"""
plt.hist(airbnb['room_type'], bins=10)
plt.show()
    
"""histogram of host_response_time"""
plt.hist(airbnb['host_response_time'].value_counts(), bins=10)
plt.show()
    
"""pie chart of the host_response_time"""
airbnb['host_response_time'].value_counts().head(30).plot(kind='pie',figsize=(6,6))
    
"""Q:26 Visualize the  correlation of the airbnb """
def visualize_corr_of_airbnb_0():
    airbnb.corr()
    sns.heatmap(airbnb.corr())

def visualize_corr_of_airbnb_1():
    """Color Palette of Heatmap"""
    sns.heatmap(airbnb.corr(),cmap="Blues",linewidths=1,xticklabels=True,yticklabels=True)
    
def visualize_corr_of_airbnb_2():
    """Diverging palette"""
    sns.heatmap(airbnb.corr(),cmap="RdBu",linewidths=1)
    
def visualize_corr_of_airbnb_3():
    """Controlling the colorbar:center,vmax,vmin"""
    sns.heatmap(airbnb.corr(),cmap="RdBu",linewidths=1,center=0,vmin=-1,vmax=1,fmt=".0f",square=True)
def visualize_corr_of_airbnb_4():
    """Annotations"""
    sns.heatmap(airbnb.corr(),cmap="RdBu",annot=True)

"""Annot_kws"""
#sns.heatmap(airbnb.corr(),cmap="RdBu",annot=True,fmt=".0f",annot_kws={'fontsize':16,'fontweight':'bold','fontfamily':'serif','color':'black'})
 
"""Xticklabels masking"""#choose the specific labels to print but actually the map is same
#labels=['host_id','zipcode','price']
#sns.heatmap(airbnb.corr(),cmap="RdBu",xticklabels=labels,yticklabels=labels)
"""Labels false"""#hide x or y labels
#labels=['host_id','zipcode','price']
#sns.heatmap(airbnb.corr(),cmap="RdBu",xticklabels=False,yticklabels=labels)
 
"""Q:27 Show linear regression of price and minimum nigths"""
"""""""""""""""""""""""LINEER REGRESSION"""""""""""""""""""""""""""""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def lin_reg_price_min_night():
    #X:independent
    #Y:dependent
    
    """simple linear regression"""   
    my_price_min_nights=airbnb.loc[:,('minimum_nights','price')]
    """check whether it has null"""
    my_price_min_nights.isnull().sum()
    """fill empty rows with mean of column"""
    my_price_min_nights.fillna(value=my_price_min_nights.mean(),inplace=True)
    """check the info of it"""
    my_price_min_nights.describe()
    
    """normalization wtih Minmax Scaler"""
    my_price_min_nights.minimum_nights=(my_price_min_nights.minimum_nights-my_price_min_nights.minimum_nights.min())/(my_price_min_nights.minimum_nights.max()-my_price_min_nights.minimum_nights.min())
    my_price_min_nights.price=(my_price_min_nights.price-my_price_min_nights.price.min())/(my_price_min_nights.price.max()-my_price_min_nights.price.min())
    
    """normalization with Sklearn"""
    #Numpy dizilerini cizdir o yuzden x ve y nin value larini al
    Y=my_price_min_nights.minimum_nights
    X=my_price_min_nights.price
    
    x_x=X.values.reshape(-1,1)
    y_y=Y.values.reshape(-1,1)
    
    normalizer = preprocessing.Normalizer().fit(x_x) 
    normalizer.transform(x_x)
    
    normalizer = preprocessing.Normalizer().fit(y_y) 
    normalizer.transform(y_y)
    
    """visualize the relationship of price and min_nights"""#This is not regression
    plt.scatter(X.values,Y.values)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('The relationship of price x min_nights y')
    plt.show()

    """get the basic stats of columns"""
    my_price_min_nights.describe()

    Y=my_price_min_nights.minimum_nights
    X=my_price_min_nights.price

    type(X) #Series
    type(Y) #Series

    #The regression between x and y
    plt.scatter(X.values,Y.values)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('The relationship between x and y')
    plt.show()

    #Creat Regression Model 
    lr_model=LinearRegression()
    lr_model.fit(X.values.reshape(-1,1),Y.values.reshape(-1,1))#Sutun sayisi 1 olacak,satir sayisi onemli degil

    lr_model.coef_   #.0.003
    lr_model.intercept_ #0.012
    print(f"The model of regression: Y= {lr_model.coef_} + {lr_model.intercept_}*x ") 

    #Estimate these values
    lr_model=LinearRegression()
    lr_model.fit(X.values.reshape(-1,1),Y.values.reshape(-1,1))#Sutun sayisi 1 olacak,satir sayisi onemli degil

    y_predicted=lr_model.predict(X.values.reshape(-1,1))
    r2_score(Y,y_predicted)#model ile ne kadar verileri acikladik,cok dusuk

    #Y predicted sutununu print et
    df_y=pd.DataFrame({'y':Y.values.flatten(),'y_predicted':y_predicted.flatten()}) #flatten komutu tum verileri tek boyuta indirger,duzlestirir
    y_predicted
    df_y # y ile tahmin edilen y tablosunu incele

    #ortalama karesel hata
    mean_squared_error(Y,y_predicted) #gayet kucuk,sonuc guzel

    #Grafik
    #2 boyutta calistigimiz icin 2 random nokta bize yeter
    bo=lr_model.intercept_[0].round(3)
    bo
    b1=lr_model.coef_ [0][0].round(3)
    b1

    random_x=np.array([0,1])
    plt.plot(random_x, bo*random_x, color='red',label='regression')
    plt.scatter(X.values,Y.values)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regression of minimum_nights and price')
    
    return lr_model

lin_reg_price_min_night()

"""Q:28 Get the linear Regression of the columns which are number of review and review scores rating."""
lr_model=lin_reg_price_min_night()

from sklearn import linear_model
#X:bagimsiz
#Y:bagimli

my_rating_and_reviews=airbnb.loc[:,('number_of_reviews','review_scores_rating')]
my_rating_and_reviews.isnull().sum()
#
Y=my_rating_and_reviews.review_scores_rating
X=my_rating_and_reviews.number_of_reviews

#%matplotlib inline
plt.title("The relationship between number of reviews and review scores rating")
plt.xlabel('num of reviews')
plt.ylabel('review scores rating')
plt.scatter(my_rating_and_reviews.number_of_reviews,my_rating_and_reviews.review_scores_rating,color='red',marker='+')

"""Fill empty rows with mean of column"""
my_rating_and_reviews.fillna(value=my_rating_and_reviews.mean(),inplace=True)
my_rating_and_reviews.describe()

"""Normalizer"""
x_x=X.values.reshape(-1,1)
y_y=Y.values.reshape(-1,1)

normalizer = preprocessing.Normalizer().fit(x_x) 
normalizer.transform(x_x)

normalizer = preprocessing.Normalizer().fit(y_y) 
normalizer.transform(y_y)

"""modelleme"""
reg=linear_model.LinearRegression()
reg.fit(X.values.reshape(-1,1),Y.values.reshape(-1,1))
"""Predict"""
reg.predict(X.values.reshape(-1,1))
"""Coef and Intercept"""
reg.coef_ #0.0013
reg.intercept_ #93.32
"""predicted"""
y_predicted=reg.predict(X.values.reshape(-1,1))
r2_score(Y,y_predicted)#low,that's fine!

#Y predicted column is printed 
df_y_rev_scores=pd.DataFrame({'y':Y.values.flatten(),'y_predicted':y_predicted.flatten()}) #flatten komutu tum verileri tek boyuta indirger,duzlestirir
print(y_predicted)
print(df_y_rev_scores) # y ile tahmin edilen y tablosunu incele

#mean of the squared error
mean_squared_error(Y,y_predicted) #low,that's fine!

#Grafik
#2 boyutta calistigimiz icin 2 random nokta bize yeter
bo=lr_model.intercept_[0].round(3)
b1=lr_model.coef_[0][0].round(3)

random_x=np.array([0,1])
plt.plot(random_x, bo*random_x, color='red',label='regression')
plt.scatter(X.values,Y.values)
plt.legend()
plt.xlabel('number of reviews')
plt.ylabel('review scores rating')

"""Q:29 What is the relationship between number of review and review scores rating"""
#1.Strong Negative Correlation 2.Moderate Negative Correlation 3.Strong Positive Correlation 4.Moderate Positive Correlation 5.No Correlation
import scipy.stats as st

"""create the shape"""
my_reviews_and_rating=airbnb.loc[:,('number_of_reviews','review_scores_rating')]
print(my_reviews_and_rating.head())

"""fill empty rows with mean of column"""
my_reviews_and_rating.number_of_reviews=my_reviews_and_rating.number_of_reviews.fillna(value=my_reviews_and_rating.number_of_reviews.mean())
my_reviews_and_rating.review_scores_rating=my_reviews_and_rating.review_scores_rating.fillna(value=my_reviews_and_rating.review_scores_rating.mean())

# Compute correlations
corr = my_reviews_and_rating.corr()
print(corr)

"""visualize the correlation"""
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.set_style(style = 'white')
# Set up  matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Add diverging colormap
cmap = sns.diverging_palette(10, 250, as_cmap=True)
# Draw correlation plot
sns.heatmap(corr,mask=mask,cmap=cmap, 
        square=True,
        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

"""the correlation coefficient by Numpy"""
X_num_rev=my_reviews_and_rating.number_of_reviews
Y_rev_score_rate=my_reviews_and_rating.review_scores_rating
corr_np=np.corrcoef(X_num_rev,Y_rev_score_rate)
print(corr_np)

"""the pearsonr correlation by Scipy"""
corr_pearsonr=st.pearsonr(X_num_rev,Y_rev_score_rate)
print(corr_pearsonr)

"""get a scatterplot of X_num_rev and Y_rev_score_rate"""
sns.scatterplot(X_num_rev, Y_rev_score_rate)

"""get a corr with Spearman r"""
from scipy.stats import spearmanr
spearmanr(X_num_rev, Y_rev_score_rate)

"""Q:30 Visualize some of columns"""
"""Bar plot of first 100 host_response_time's value_counts """
airbnb['host_response_time'].head(100).value_counts().sum()
airbnb.zipcode.head(50).value_counts().plot.bar()

"""value_counts() gets the number of very value"""
airbnb['zipcode'].value_counts().head(30)

"the most crowded 30 zipcodes about hosting in pie"
airbnb['zipcode'].value_counts().head(10).plot(kind='pie',figsize=(6,6))

"...in bar plotting"
airbnb['zipcode'].value_counts().head(10).plot(kind='bar' )

"""Q:31 Visualize the relationship between price and number of reviews"""
airbnb.plot(kind='scatter',x='price',y='number_of_reviews',figsize=(6,6))

"""Q:32 Get the histogram of price column"""
hist_of_price=airbnb['price'].plot(kind='hist',bins=100,figsize=(14,10) )

"""Q:33 Group by the mean of the first 5 zipcode's price"""
print(airbnb.groupby(by="zipcode").price.mean().head(5))

"""Q:34 Get the stats description of the grouped airbnb by the bed numbers"""
print(airbnb.groupby("beds").review_scores_location.describe())

"""get the mean of airbnb which is grouped by host since year"""
print(airbnb.groupby(by="host_since_year").mean())
"""get the median of airbnb which is grouped by host since year"""
print(airbnb.groupby(by="host_since_year").median())

"""get a list of host_id and host_response_time"""
list_of_response=airbnb.loc[:,['host_id','host_response_time']]
res=list_of_response.host_id.isna().sum()
print(res)#0
res2=list_of_response.host_response_time.isna().sum()
print(res2) #732
print('nan Sayisi:{}'.format(res))
print('nan Sayisi:{}'.format(res2))

"""Q:35 How many hosts have more than 1 bed?"""
list_of_price=airbnb.loc[:,['host_name','price','beds']]
count=0
for i in airbnb.beds:
    if i>1:
        count+=1
        
print("Count of hosts who have more than 1 bed:",count)        

"""Q:36 Calculate the price per bed of every rows?"""
"""check the data types"""
print(airbnb.beds.dtype)
print(airbnb.price.dtype)
"""last price column"""
airbnb['last_price']=airbnb['price']/airbnb['beds']
"""get a list of price per unit"""
my_last_price=airbnb.loc[:,['host_name','beds','price','last_price']]

"""Q:37 What is the average price of the all beds in Amsterdam?"""
avg_price=my_last_price.last_price.mean()
print("Average price of beds in Amsterdam:",avg_price)

"""Q:38 Get a list of hosts who price more than average"""
mask_expensive=(my_last_price.last_price>=avg_price)
my_last_price.loc[mask_expensive,'last_price']
mask_exp=my_last_price.dropna(subset = ["last_price"])
"""check the data"""
print(my_last_price.last_price.isna().sum()) #13
print(airbnb.price.isna().sum())

"""Q:39 Get a list of 13 host names who has no price"""
son=my_last_price[my_last_price['last_price'].isnull()].host_name.tolist()

"""fill na"""
my_last_price['beds'].fillna(1,inplace=True)
print(my_last_price[my_last_price['beds'].isnull()])
"""check the data"""
my_last_price.groupby('beds')['price'].describe() 
print(airbnb.shape)
print(airbnb._get_numeric_data())

"""mask the dataframe with columns which are numeric"""
print(airbnb.select_dtypes(include= np.number))


for x in airbnb.columns: 
    lambda x : print(x.dtype) if x != None else x

"""Q:40 What is the distribution of the airbnb"""

"""histogram gets only numerical data"""
for i in airbnb.columns:
    print(i)
    print(airbnb[i].dtype)
    plt.title("Histogram of numerical columns of airbnb")
    plt.hist(x=airbnb[i],bins=100,label=i)   
    plt.show()

"""most popular zipcode"""
print(airbnb['zipcode'].mode())#1054
"""trim letters in zipcode"""
print(airbnb['zipcode'].isnull().sum())#173
my_trimmed_airbnb_zip=airbnb['zipcode'].str.split(n=1)#creates a list which has 2 columns with zipcode, for ex:(1054,AM)

new_zip=[]
deleted_element=0
for i in range(len(my_trimmed_airbnb_zip)) : 
    try:
        print(my_trimmed_airbnb_zip[i].pop(1))
        print(my_trimmed_airbnb_zip[i])
        new_zip.append(my_trimmed_airbnb_zip[i])
    except:
        deleted_element+=1
        print("Out of range")
        new_zip.append(1054) #most frequent value
        print("added 1054")        
print(f"deleted element number:{deleted_element}")       
#deleted element number:3770 This means that we had 3770 zipcode with letters,others are only numbers
"""print the new trimmed zipcode/how to print a column"""
for i in range(len(new_zip)):
    print(f"Element {i}:{new_zip[i]}")
type(new_zip[4])#list, this means that some values's type is list,but it must be int    
"""add trimmed zipcode in airbnb dataframe"""
airbnb['new_zip']=new_zip
"""get info with data frame which is with trimmed zipcode"""
desc_zipcode_price=airbnb.groupby(by="zipcode").price.describe()
xprc=airbnb['price'].head(10).hist(by=airbnb['host_name'],bins=50,alpha=0.5,figsize=(8,4),xrot=45)
"""change zipcode's data type from object to string"""
airbnb.zipcode=airbnb.zipcode.astype(str)
print(airbnb.zipcode.dtype)#object

"""distribution of the zipcode"""
#airbnb.new_zip=airbnb.new_zip.astype(int)
#print(airbnb.new_zip.dtype)
#airbnb.hist(column='new_zip')#non numerical error
sorted_zip=airbnb.groupby('zipcode')['price'].sum().sort_values(ascending=False).plot(kind='bar')
  
"""what is the dtype of zipcode?"""
print(airbnb['zipcode'].dtype) #object
"""To get the histogram,i change dtype of zipcode to int"""
print(airbnb['zipcode'].np.isna().sum())
print(airbnb.zipcode[1].dtype)
"""histogram of review_scores_value"""
airbnb.hist(column='review_scores_value')
"""histogram of price"""
airbnb.hist(column='last_price',alpha=0.7,figsize=(10,4),grid=True,orientation='horizontal',color='#FFCF56',legend=True)

#histogram with matplotlib  
airbnb['last_price'].plot(kind='hist',bins=30,title='Histogram of last price',color='#A0E8AF')

over_100=airbnb[airbnb['last_price']<500]
over_100['last_price'].plot(kind='hist',bins=30,title='Histogram of last price',color='#A0E8AF')

"""numeric olmayan bir sutunun plot ile gosterimi""" 
airbnb.zipcode.value_counts().plot()    

"""standart deviation of 'review_scores_value'"""
print(airbnb.review_scores_value.std)

"""standart deviation of last_price"""
print(airbnb.last_price.std())

"""sutunlarin standart sapmasini almak icin axis=0 olmalidir"""
std_of_columns=airbnb.std(axis=0)

"""correlation of minimum_nights and last_price"""
min_night_price_corr=airbnb[['minimum_nights','last_price']].corr()    
print(min_night_price_corr)
sns.heatmap(min_night_price_corr, annot = True, fmt='.2g',cmap= 'coolwarm')
    
"""correlation of review_scores_value and last_price"""
score_price_corr=airbnb[['review_scores_value','last_price']].corr()    
print(score_price_corr)
sns.heatmap(score_price_corr, annot = True, fmt='.2g',cmap= 'coolwarm')
"""alternative"""
rev_score_value_and_price_corr=airbnb["review_scores_value"].corr(airbnb["price"])
rev_score_value_and_price_corr.style.background_gradient(cmap='coolwarm')

"""Q:41:Plot the distribution of price,which zipcode is the most expensive?"""
print(airbnb["zipcode"].index) #range index
print(airbnb["zipcode"].array)

float_of_all_zipcode=airbnb["new_zip"].to_numpy()
float_of_all_zipcode[0:20]

list(map(lambda x:x,airbnb["new_zip"][0:30]))
print(len(float_of_all_zipcode))
print(float_of_all_zipcode.dtype)

 
"""
SORU: zipcode'larin sehirmerkezine yakinligi ile fiyat arasindaki corelasyon ve bunla ilgili bir harita yapimi)
"""


"""
Yillara nazaran degisen fiyat araligi ve enflasyon orani ile arasindaki corelasyon
kac yillik host ve musteri memnuniyeti arasindaki iliski(Tecrubeli hostlar daha fazla musteri cekebiliyor mu?)
En cok tercih edilen ve review alan neigborrhoed hangisi(column 6)
(column 7) Amsterdam harici fiyat dagilimi
Koordinatlar ile sehir merkezine yakin olan zipcodelarin fiyat dagilimi
minimum_nights ile price arasindaki iliski(fiyat uygun oldukca minimum_nights artiyor mu?)
roomtype ile fiyat arasindaki iliski
property_type ile review_scores_value arasidnaki corelasyon
"""

"""Multiple Linear Regression"""
import math
import numpy as np
from haversine import haversine, Unit

class Haversine:
    '''
    use the haversine class to calculate the distance between
    two lon/lat coordnate pairs.
    output distance available in kilometers, meters, miles, and feet.
    example usage: Haversine([lon1,lat1],[lon2,lat2]).feet
    
    '''
    def __init__(self,coord1,coord2):
        lon1,lat1=coord1
        lon2,lat2=coord2
        
        R=6371000                               # radius of Earth in meters
        phi_1=math.radians(lat1)
        phi_2=math.radians(lat2)

        delta_phi=math.radians(lat2-lat1)
        delta_lambda=math.radians(lon2-lon1)

        a=math.sin(delta_phi/2.0)**2+\
           math.cos(phi_1)*math.cos(phi_2)*\
           math.sin(delta_lambda/2.0)**2
        c=2*math.atan2(math.sqrt(a),math.sqrt(1-a))
        
        self.meters=R*c                         # output distance in meters
        self.km=self.meters/1000.0              # output distance in kilometers
        self.miles=self.meters*0.000621371      # output distance in miles
        self.feet=self.miles*5280               # output distance in feet

   
airbnb_coordinates=airbnb.loc[:,('latitude', 'longitude')]
airbnb_coordinates.head()

latitude_DamSquare=52.373 
longitude_DamSquare=4.893

Haversine([-84.412977,39.152501],[-84.412946,39.152505]).km  

Haversine([latitude_DamSquare,longitude_DamSquare],[52.373021,4.868461]).km

Haversine([latitude_DamSquare,longitude_DamSquare],[airbnb['latitude'][1],airbnb['longitude'][1]]).km


airbnb['distance'] = airbnb.apply(lambda x: Haversine([latitude_DamSquare,longitude_DamSquare],[x['latitude'], x['longitude']]).km, axis=1)

"""histogram of distance"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

x = np.random.normal(airbnb['distance'])
plt.hist(x, density=True, bins=200)
plt.ylabel('Distribution')
plt.xlabel('Km');
plt.hist(x)
plt.show() 

"""correlation of last_price and distance"""
data1=airbnb['distance']
data2=airbnb['last_price']

data2.isnull().sum()#13
data2.mean()#80.84217495331474

data2=data2.fillna(80.84217495331474)
data2.isnull().sum()

from scipy import stats
stats.pearsonr(data1, data2)
"""(-0.11323005689027815, 8.9946956382177e-24)"""

from matplotlib import pyplot
pyplot.scatter(data1, data2)
pyplot.show()

data2.median() #75
data2.mean()#80.8
data2.mode() #100
data2.value_counts()

mask_price=data2[data2>250]


"""get the index list of prices which are higher than 250"""

index = data2.index
index
condition = data2 >=251
price_indices = index[condition]
len(price_indices) #37

price_indices = price_indices.tolist()
print(price_indices)

count=0
for x in price_indices: 
    count+=1
    print(f" value order: {count}, index: {x} value:{data2[x]}")
    
    
data2_masked = data2.apply(lambda x : 80.8 if x >=251 else x)

mask_price=data2_masked[data2_masked>250]
print(mask_price)
 
from matplotlib import pyplot
pyplot.scatter(data1, data2_masked)
pyplot.show()    

from scipy import stats
stats.pearsonr(data1, data2_masked)
"""(-0.2907218912105367, 2.1306340946958336e-152)"""
correlation = data1.corr(data2_masked)

"""""""""""""""""""""""""""""""""Multiple Linear Regression"""""""""""""""""""""""""""""""""""""""
X = airbnb[['distance','review_scores_value']]
y = airbnb['last_price']

airbnb['last_price'].isnull().sum()
airbnb['last_price'].mean()
airbnb['last_price']=airbnb['last_price'].fillna(80.84)
airbnb['last_price'].isnull().sum()

airbnb['review_scores_value'].isnull().sum()
airbnb['review_scores_value'].mean()
airbnb['review_scores_value']=airbnb['review_scores_value'].fillna(9.04)
airbnb['review_scores_value'].isnull().sum()

airbnb['distance'].isnull().sum()

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the last price of a room where the distance is 0.1 km to Dam Square, and the review scores value is 10:
predicted_price = regr.predict([[0.1, 10]])
print(predicted_price) #102.17

"""coefficient"""
#Coefficient:  The coefficient is a factor that describes the relationship with an unknown variable.
print(regr.coef_) #[-5.94978158  3.26885832] #Represent distance and review score value

"""these values tell us that if the distance increase by 1 km, the last prices increases by -5.94978158.
and if the review score value increases by 1 point, the last prices increases by 3.26885832."""


predicted_price = regr.predict([[10,10]])
print(predicted_price)


import seaborn as sns
sns.set_theme(style="whitegrid")
mydata = airbnb[['distance','last_price']]

"""standart deviation of  distance column"""
print(airbnb['distance'].std())#2.10610204775874

"""trimmed variance of distance column"""
print(stats.tvar(airbnb['distance']))#4.435665835573577

"""coefficient of Variation"""
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 
print(cv(airbnb['distance'])) #66.684

"""covariance of distance column and review_scores_value"""
X = airbnb['distance']
y = airbnb['review_scores_value']
print(np.cov(X,y))

"""[[4.43566584 0.02235222]
 [0.02235222 0.60763976]]"""
    
    
"""Correlation of distance and review_scores_value"""    
X = airbnb[['distance','review_scores_value']]
print(X.corr(method ='kendall'))
"""    distance  review_scores_value
distance             1.000000             0.030985
review_scores_value  0.030985             1.000000"""

X = airbnb[['distance','review_scores_value']]
print(X.corr(method ='pearson'))
"""distance  review_scores_value
distance             1.000000             0.013615
review_scores_value  0.013615             1.000000"""


"""""""""""""""""""""""""""""Hypothesis Testing of Correlation"""""""""""""""""""""""""""""""""""""""""""""

"""We are going to test whether out correlation of distance and review_score_location is correct."""
correlation = airbnb['distance'].corr(airbnb['review_scores_location'])
print(correlation) #-0.4130267080557865

"""We have one numeric column and one categorical column(review score) and our categorical column has more than 5 different calue.
Then we have to implement a Normalization Test before we start T-Test"""

"""Normal Distribution function"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

"""Normal distribution of column 'distance'"""

x, y = ecdf(airbnb["review_scores_location"])

plt.figure(figsize=(8,7))
sns.set()
plt.plot(x, y, marker=".", linestyle="none")
plt.xlabel("review_scores_location (F)")
plt.ylabel("Cumulative Distribution Function")

samples = np.random.normal(np.mean(airbnb["review_scores_location"]), np.std(airbnb["review_scores_location"]), size=10000)
x_theor, y_theor = ecdf(samples)
plt.plot(x_theor, y_theor)
plt.legend(('Normal Distribution of review scores location', 'Empirical Data'), loc='lower right')
print(stats.normaltest(airbnb["review_scores_location"]))


"""Normal distribution of column 'distance'"""
x, y = ecdf(airbnb["distance"])

plt.figure(figsize=(8,7))
sns.set()
plt.plot(x, y, marker=".", linestyle="none")
plt.xlabel("distance (F)")
plt.ylabel("Cumulative Distribution Function")

samples = np.random.normal(np.mean(airbnb["distance"]), np.std(airbnb["distance"]), size=10000)
x_theor, y_theor = ecdf(samples)
plt.plot(x_theor, y_theor)
plt.legend(('Normal Distribution of distance', 'Empirical Data'), loc='lower right')
print(stats.normaltest(airbnb["distance"]))

""" If the p value is less than our alpha (significance value), we can reject the hypothesis that this sample data is normally 
distributed. If greater, we cannot reject the null hypothesis and must conclude the data is normally distributed. """


"""Independent Sample T-Test (Student's Test)"""
#Null Hypothesis (H0): The result of correlation between distance and review location score is not right.
#Alternative Hypothesis (H1):This is right conclution.
  
airbnb['review_scores_location'] =airbnb['review_scores_location'] .fillna(9.04)         
 
correlation=airbnb['distance'].corr(airbnb['review_scores_location']) #-0.41 Negative correlation means if the distance decrease,review score is increase.
print(correlation)

from scipy.stats import variation
x = np.array([airbnb['distance'],airbnb['review_scores_location']]) 
print(variation(x, axis=1, nan_policy='omit'))#[0.66680441 0.08208538]



from scipy import stats as st
a =  airbnb['distance'].to_numpy()
b =  airbnb['review_scores_location'].to_numpy()

result= st.ttest_1samp(b, popmean=0)
print(result)

# Use scipy.stats.ttest_ind.
from scipy import stats
a =  airbnb['last_price'].to_numpy()
b =  airbnb['distance'].to_numpy()

t, p = stats.ttest_ind(a,b, equal_var=False)#ttest_ind:t = -25.2209  p = 1.48107e-137
print("ttest_ind:            t = %g  p = %g" % (t, p))

#Result: Ho is rejected.
"""Normaliation and Standart Normalization"""

"""Probablity"""
"""If there are correlation between two columns,next step you should search/calculate probability as a statistical significant"""
"""correlation is not enough to prove relationship between two columns,that's why we need to look probablilty
correlation doesnt mean coezation"""

"""Uniform(Tekduze) Distribution"""
import matplotlib.pyplot as plt
s=airbnb['distance']
count, bins, ignored = plt.hist(s, 100, density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()






