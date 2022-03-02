'''Regression Problem:- This script aims to predict 'price' based on a set of features
    Dataset from kaggle house sales prediction
    Link: https://www.kaggle.com/harlfoxem/housesalesprediction'''
##Import all libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error,explained_variance_score

if __name__ == '__main__':
    ##Step1:Read data from kc_house_data.csv
    df=pd.read_csv("..//Data//kc_house_data.csv")
    print(df.head())
    ##step2:check for missing values per column
    print(df.isnull().sum()) ## No missing values found

    ##step3:Data Visualization###

    ##find the correlation between each column use df.corr()
    print((df.corr()['price']).sort_values()) ###Since we neeed correlation with price choose that column only
    ##Since sqft_living is highly correlated with price, Let's create a scattter plot
    # plt.figure(figsize=(10,5)) ##To expand figure size
    # sns.scatterplot(x='price',y='sqft_living',data=df)
    # plt.show()
    # ##To see how well no. of bedrooms correlated with price, Let's create a scattter plot
    # plt.figure(figsize=(12,8))
    # sns.scatterplot(x=df['price'],y=df['bedrooms'])
    # plt.show()
    # ##We can also use boxplot to see the distribution
    # plt.figure(figsize=(10,5))
    # sns.boxplot(y='price',x='bathrooms',data=df)
    # plt.show()
    # ##We can also use barplot to see the distribution
    # plt.figure(figsize=(10,5))
    # sns.barplot(y='price',x='bedrooms',data=df)
    # plt.show()
    # ##We can also use countplot to see the counts
    # plt.figure(figsize=(10,5))
    # sns.countplot(df['bedrooms'])
    # plt.show()
    ##We can also use distplot to see the distribution
    # plt.figure(figsize=(10,5))
    # sns.distplot(df['price'])
    # plt.show()
    ##From price distribution it is clear that there are some outliers which needed to be removed
    ##That is most of the price is between 0 to 3*10^6 so we need to remove some outliers >3*10^6
    ##This can be done by sorting prices in descending order and removing top 1% priced samples
    # print(df.sort_values('price',ascending=False).head(20))
    ##Creating a new dataframe with non top 1% samples
    start_index=round(0.01*len(df)) ##no. of samples in 1% of top prices
    non_top_1p_df=df.sort_values('price',ascending=False).iloc[start_index:]

    ##Plot latitude and longitude in scatter plot and price as hue
    # sns.scatterplot(y='lat',x='long',hue='price',data=non_top_1p_df)
    # plt.show()
    # ##set palette for differnet color gradient
    # sns.scatterplot(y='lat',x='long',hue='price',data=non_top_1p_df,palette='RdYlGn')
    # plt.show()
    ##From the above scatter plot it is clear that watfront areas have more price
    ##We can also use boxplot to check the above statement is correct or not
    # plt.figure(figsize=(10,5))
    # sns.boxplot(x='waterfront',y='price',data=df)
    # plt.show()

    ##Step4: Feature engineering// Removing unwanted features
    #print(non_top_1p_df.info())
    ##id looks unwanted so remove that column
    non_top_1p_df=non_top_1p_df.drop('id',axis=1)

    ##from date column we need only month and year as there is some relationship between month,year and price

    non_top_1p_df['date']=pd.to_datetime(non_top_1p_df['date']) ##converted to pandas date time format
    non_top_1p_df['year']=non_top_1p_df['date'].apply(lambda date:date.year) ##Extract year only
    non_top_1p_df['month']=non_top_1p_df['date'].apply(lambda date:date.month) ##Extract month only
    non_top_1p_df=non_top_1p_df.drop('date',axis=1)

    ##Year rennovated column contains 0 for not rennovated and years corresponding to the year of rennovation--make it 0 (not rennovated)
    ##1 for rennovated
    non_top_1p_df['yr_renovated']=non_top_1p_df['yr_renovated'].apply(lambda yr_rennovated:1 if yr_rennovated>0 else 0)
    #non_top_1p_df.to_csv("data_updated.csv",index=False)

    ##zipcode looks like a lot of feature engineering needed --Let's drop it for this time
    non_top_1p_df=non_top_1p_df.drop('zipcode',axis=1)
    #print(non_top_1p_df.head())

    ##Step4: Now the data is ready for separating features from dataframe and train test split
    X=non_top_1p_df.drop('price',axis=1).values
    y=non_top_1p_df['price'].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    ##Step5: Scaling
    scaler=MinMaxScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    ##Step6: Create a model
    model=Sequential()
    model.add(Dense(units=19,activation='relu'))
    model.add(Dense(units=24,activation='relu'))
    model.add(Dense(units=36,activation='relu'))
    model.add(Dense(units=24,activation='relu'))
    model.add(Dense(units=19,activation='relu'))

    model.add(Dense(units=1))

    # model.compile(optimizer='adam',loss='mse')
    # model.fit(x=X_train,y=y_train,batch_size=128,epochs=400,validation_data=(X_test,y_test))
    # train_loss_df=pd.DataFrame(model.history.history)
    # train_loss_df.plot()
    # plt.show()
    # model.save('kaggle_regression_v01.h5')

    ##Since there is no overfitting..evaluate test data
    model=load_model('kaggle_regression_v01.h5')
    test_loss=model.evaluate(x=X_test,y=y_test,verbose=1)
    print(test_loss)

    ##To compare test data predictions with the labels
    output_df=pd.DataFrame(model.predict(x=X_test).reshape(4277,),columns=['Predictions'])
    output_df['Labels']=y_test
    rmse=mean_squared_error(output_df['Labels'],output_df['Predictions'])**0.5
    print(rmse)
    print(non_top_1p_df['price'].describe().T)##rmse Comparing with price mean 518367(+/-)141740 not much big ie  20% tolerance
    ##Use explanied variance score for better analysis 1 is best lower values are worse
    evs=explained_variance_score(output_df['Labels'],output_df['Predictions'])
    print(evs) ##a score  of 0.759 looks good..you can train even more or change model for better score
    ##make a scatter plot of predictions and labels, the plot will be a straight line for 100% correct predcitions
    sns.scatterplot(x=output_df['Labels'],y=output_df['Predictions'])
    plt.plot(output_df['Labels'],output_df['Labels'],'r-') ##plot y_test as straight line for better understanding
    plt.show()

    ##Step6: Test model using a new data, Let's take the first row sample as new data
    print(X_test.shape)
    new_data=non_top_1p_df.drop('price',axis=1).iloc[0].values
    new_data=new_data.reshape(1,19)
    out_new_data=model.predict(scaler.transform(new_data))
    print(out_new_data,y_test[0])
