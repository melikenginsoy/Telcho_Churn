#############################################
# Feature Engineering of The Telcho_Churn
#############################################

##################
# Business Problem
##################

# Developing a machine learning model that can predict customers leaving the company is requested.
# In this project, necessary data analysis and feature engineering steps will be performed
# before model development.

####################
# About the Data Set
####################

# Telco churn data provided 7043 California customers with home phone and contains information about
# a fictitious telecom company that provides Internet services.
# It shows which customers have left, stayed or signed up for their service.

# CustomerId:
# Gender:
# SeniorCitizen: Whether the customer is old (1, 0)
# Partner: Whether the customer has a partner (Yes, No)
# Dependents: Whether the customer has dependents (Yes, No)
# tenure: Number of months the customer has stayed with the company
# PhoneService: Whether the customer has telephone service (Yes, No)
# MultipleLines: Whether the customer has more than one line (Yes, No, No phone service)
# InternetService: Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity: Whether the customer has online security (Yes, No, No Internet service)
# OnlineBackup: Whether the customer has an online backup (Yes, No, No Internet service)
# DeviceProtection: Whether the customer has device protection (Yes, No, No Internet service)
# TechSupport: Whether the customer received technical support (Yes, No, No Internet service)
# StreamingTV: Whether the customer has a TV broadcast (Yes, No, No Internet service)
# StreamingMovies: Whether the client is streaming movies (Yes, No, No Internet service)
# Contract: Customer's contract term (Month to month, One year, Two years)
# PaperlessBilling: Whether the customer has a paperless bill (Yes, No)
# PaymentMethod: Customer's payment method (Electronic check, Postal check,
#                                           Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges: Amount charged from the customer on a monthly basis
# TotalCharges: Total amount charged from customer
# Churn:  (Yes, No)

############################
# Exploratory Data Analysis
############################

# 1- Necessary Libraries:

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from statsmodels.stats.proportion import proportions_ztest
from sklearn.ensemble import RandomForestClassifier

# 2- Customize DataFrame to display:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# 3- Loading the Dataset:

def load():
    data = pd.read_csv("Telcho_Churn/Telco-Customer-Churn.csv")
    return data


df = load()

# 4- Dataset Overview:

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# The data set consists of 3 numerical, 18 object variables and 7043 observation units.
# While TotalCharges should be a numeric variable, it is an object type variable.
# We must make this transformation.

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')

# The dependent variable:

df["Churn"] = df["Churn"].map({'No':0,'Yes':1})

df["Churn"].value_counts()

"""
0    5163
1    1869
"""
df.info()

# There are no missing observations in the data set.

# 5- Numeric and Categorical variables:

# Although the data set consists of 4 numerical variables, some variables may actually be categorical.
# There may actually be cardinal variables, as there are categorical variables and the number of unique
# classes is high.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of 3 lists with return is equal to the total number of variables:
        cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Variables: 21
# cat_cols: 17
# num_cols: 3
# cat_but_car: 1
# num_but_cat: 2

# num_cols
# ['tenure', 'MonthlyCharges', 'TotalCharges']

# cat_but_car
# ['customerID']

# cat_cols
# ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity''OnlineBackup',
#  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
#  'Churn','SeniorCitizen']

# Analysis of numerical and categorical variables:

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

# Average of numerical variables relative to the dependent variable:

def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({"CHURN_MEAN": dataframe.groupby(categorical_col)[target].mean()}))
    print("###################################")


for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# 6- Outliers Analysis:

# Define an outlier thresholds for variables:

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Check for outliers for variables:

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

# tenure False
# MonthlyCharges False
# TotalCharges False

# 7- The Missing Values Analysis:

# Missing value and ratio analysis for variables:

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df, True)

"""
n_miss  ratio
TotalCharges      11  0.160
"""

# 8- Correlation Analysis:

def correlated_cols(dataframe, plot=False):
    corr_matrix = dataframe.corr()

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr_matrix, cmap="RdBu")
        plt.show()
    return print(corr_matrix)


correlated_cols(df, plot=True)

"""
                SeniorCitizen  tenure  MonthlyCharges  TotalCharges  Churn
SeniorCitizen           1.000   0.017           0.220         0.102  0.151
tenure                  0.017   1.000           0.248         0.826 -0.352
MonthlyCharges          0.220   0.248           1.000         0.651  0.193
TotalCharges            0.102   0.826           0.651         1.000 -0.199
Churn                   0.151  -0.352           0.193        -0.199  1.000

"""
# There is a moderate positive high level of relationship between tenure and TotalCharges.(0.826)
# There is a moderate positive correlation between MonthlyCharges and TotalCharges. (0.651)

############################
# Future Engineering
############################

# 1- Necessary actions for missing and outliers values:

na_cols = missing_values_table(df, True)

"""
              n_miss  ratio
TotalCharges      11  0.160
"""
# Since the number of missing observations is low, we prefer to delete it from the data set.

df.dropna(inplace=True)

# 2- Creating New Features:

df["gender"] = df["gender"].map({'Male':0,'Female':1})

df.loc[((df['gender'] == 0) & (df["PhoneService"]== "No" )), 'PhoneService_gender'] ="no_PhoneService_male"
df.loc[((df['gender'] == 0) & (df["PhoneService"]== "Yes")), 'PhoneService_gender'] ="yes_PhoneService_male"
df.loc[((df['gender'] == 1) & (df["PhoneService"]== "No")), 'PhoneService_gender'] ="no_PhoneService_female"
df.loc[((df['gender'] == 1) & (df["PhoneService"]== "Yes")), 'PhoneService_gender'] ="yes_PhoneService_female"

df.groupby("PhoneService_gender").agg({"Churn": ["mean","count"]})

"""
                         mean count
PhoneService_gender                
no_PhoneService_female  0.243   329
no_PhoneService_male    0.256   351
yes_PhoneService_female 0.272  3154
yes_PhoneService_male   0.263  3198
"""

df.loc[((df['gender'] == 0) & (df["SeniorCitizen"]== 1)), 'Senior_young_gender'] ="senior_male"
df.loc[((df['gender'] == 0) & (df["SeniorCitizen"]== 0)), 'Senior_young_gender'] ="young_male"
df.loc[((df['gender'] == 1) & (df["SeniorCitizen"]== 1)), 'Senior_young_gender'] ="senior_female"
df.loc[((df['gender'] == 1) & (df["SeniorCitizen"]== 0)), 'Senior_young_gender'] ="young_female"

df.groupby("Senior_young_gender").agg({"Churn": ["mean","count"]})

"""
                    Churn      
                     mean count
Senior_young_gender            
senior_female       0.423   568
senior_male         0.411   574
young_female        0.240  2915
young_male          0.233  2975
"""

# 3- label Encoding and One-Hot Encoding:

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

# Create a variable for each observation unit by making the variables we have one hot encoder.

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()

# 4- Standardization for numerical variables:

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()

############
# Modelling
############

# Dependent variable:
y = df["Churn"]

# Independent variables:
X = df.drop(["Churn","customerID"], axis=1)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# 0.7990521327014218
