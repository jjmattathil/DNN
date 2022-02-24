""" This script aims to predict 'price' based on 'feature1' and 'feature2'  of fake.csv
    ---Regression Problem--- """
##Import all Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error,mean_absolute_error

if __name__ == '__main__':

    ##Step1: Import data
    df=pd.read_csv("fake_reg.csv")
    ##Plot data just to see how features are dependent on price
    #sns.pairplot(df)
    #plt.show()

    ##Step2: Convert data to numpy array (Use values for converting df to numpy array)
    X=df[['feature1','feature2']].values ###Capital X because 2D
    y=df['price'].values ###Small y because 1D
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    print(X_train.shape) ###(800,2)
    print(X_test.shape) ###(200,2)
    print(y_train.shape) ###(800,)
    print(y_test.shape) ###(200,)

    ##Step3: Normalize or Scale feature data
    scaler=MinMaxScaler()
    scaler.fit(X_train) ## fit function will calculate std,min, and max of X_train
    X_train=scaler.transform(X_train) ## Aplly scaling to X_train data using calculated std,min and max values
    scaler.fit(X_test) ## fit function will calculate std,min, and max of X_test
    X_test=scaler.transform(X_test) ## Aplly scaling to X_test data using calculated std,min and max values
    print(X_train.max()) ##Now the values in range 0 to 1
    print(X_train.min()) ##Now the values in range 0 to 1

    ##Step4: Create a model
    model=Sequential()
    model.add(Dense(units=4,activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=2,activation='relu'))
    model.add(Dense(units=1))

    ##Choose an optimizer and loss fn.
    model.compile(optimizer='adam',loss='mse')
    ##Train model
    model.fit(x=X_train,y=y_train,epochs=300)
    model.save("mlp_regression_v01.h5")


    ##plot loss ---converting to a dataframe for easy plotting
    loss_df=pd.DataFrame(model.history.history)
    loss_df.plot()
    plt.show()

    ##Step5: Model evaluation returns mean square error// save model if needed
    model = load_model("mlp_regression_v01.h5")
    test_loss=model.evaluate(x=X_test,y=y_test,verbose=0)
    train_loss=model.evaluate(x=X_train,y=y_train,verbose=0)
    print(test_loss)
    print(train_loss)

    ##To comparetest data predictions with the labels
    test_predictions=pd.Series(model.predict(x=X_test).reshape(200,))
    predictions_df=pd.DataFrame(data=test_predictions,columns=["Model predictions"])
    predictions_df["Price Labels"]=y_test
    print(predictions_df)
    ##make a scatter plot of predictions and labels, the plot will be a straight line for 100% correct predcitions
    sns.scatterplot(y=predictions_df["Model predictions"],x=predictions_df["Price Labels"])
    plt.show()
    ##Find mean absolute error,mse and rmse for quantitative analysis
    mabe=mean_absolute_error(predictions_df["Price Labels"],predictions_df["Model predictions"])
    mse=mean_squared_error(predictions_df["Price Labels"],predictions_df["Model predictions"])
    rmse=mean_squared_error(predictions_df["Price Labels"],predictions_df["Model predictions"])**0.5
    print(mabe,mse,rmse)

    ##Step6: Test model using a new data
    new_data=[[1000,1000]]
    new_data=scaler.transform(new_data)
    new_data_price=model.predict(x=new_data)
    print(new_data_price)




