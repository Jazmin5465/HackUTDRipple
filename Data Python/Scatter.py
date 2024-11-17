import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import plotly.express as px


def fill_all_nulls_with_prediction_and_print(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fills null values in all columns (except time) using a regression model.
    Handles missing values in predictors using median imputation.
    Prints the entire DataFrame after filling.
    """
    data = data.copy()
    for target_column in data.columns:
        if target_column == 'Time' or data[target_column].isnull().sum() == 0:
            continue  

        
        train_data = data[data[target_column].notnull()]
        test_data = data[data[target_column].isnull()]

        if test_data.empty:
            continue  

        
        X_train = train_data.drop(columns=[target_column, 'Time'], errors='ignore')
        y_train = train_data[target_column]
        X_test = test_data.drop(columns=[target_column, 'Time'], errors='ignore')

        
        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        
        model = LinearRegression()
        model.fit(X_train, y_train)

        
        predicted_values = model.predict(X_test)

        
        data.loc[data[target_column].isnull(), target_column] = predicted_values

    
    print(data)
    
    # Generate scatter plot
    fig = px.scatter(data, x='Inj Gas Valve Percent Open', 
                     y='Inj Gas Meter Volume Instantaneous', 
                     title='Scatter Plot: Valve Percent Open vs Meter Volume',
                     labels={
                         'Inj Gas Valve Percent Open': 'Valve Percent Open (%)',
                         'Inj Gas Meter Volume Instantaneous': 'Meter Volume (Instantaneous)'
                     },
                     trendline='ols'  # Add line of best fit
                    )
    fig.show()
    
    return data

# Loading data
boldData = pd.read_csv('HackUTD-RippleEffect/data/Bold_744H-10_31-11_07.csv') 
courageousData = pd.read_csv('HackUTD-RippleEffect/data/Courageous_729H-09_25-09_28.csv') 
fearlessData = pd.read_csv('HackUTD-RippleEffect/data/Fearless_709H-10_31-11_07.csv')
gallantData = pd.read_csv('HackUTD-RippleEffect/data/Gallant_102H-10_04-10_11.csv') 
nobleData = pd.read_csv('HackUTD-RippleEffect/data/Noble_4H-10_24-10_29.csv') 
resoluteData = pd.read_csv('HackUTD-RippleEffect/data/Resolute_728H-10_14-10_21.csv') 
ruthlessData = pd.read_csv('HackUTD-RippleEffect/data/Ruthless_745H-10_01-10_08.csv') 
steadfastData = pd.read_csv('HackUTD-RippleEffect/data/Steadfast_505H-10_30-11_07.csv') 
valiantData = pd.read_csv('HackUTD-RippleEffect/data/Valiant_505H-09_22-09_30.csv')

#clog occurs when valve is getting larger but gas volume is getting smaller

# tests
filled_full_dataset_with_print = fill_all_nulls_with_prediction_and_print(boldData)
