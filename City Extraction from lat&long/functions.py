#!/usr/bin/env python
# coding: utf-8

# In[1]:


def get_out_liers(df, name):
    d = df[name].describe()
    Q1, Q3 = d['25%'], d['75%']
    iqr = Q3 - Q1
    min_whis, max_whis = (Q1 - 1.5 * iqr), (Q3 + 1.5 * iqr)
    out = list(df[(df[name] < min_whis) | (df[name] > max_whis)].index)
    return out


# In[2]:


def substitute(df, col, places, value):
    df.loc[places, col] = value


# In[3]:


def OneHotE(df, col):
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    encoder = OneHotEncoder(sparse=False, drop='first')
    transformed = pd.DataFrame(encoder.fit_transform(df[[col]]), columns = encoder.get_feature_names_out())
    df2 = pd.concat([df.drop(columns= [col]), transformed],axis =1)
    return df2


# In[4]:


def BinaryE(df, col):
    from category_encoders import BinaryEncoder
    import pandas as pd
    bi = BinaryEncoder()
    encoded = bi.fit_transform(df[[col]])
    df2 = pd.concat([df.drop(columns= [col]), encoded],axis =1)
    return df2


# In[5]:


def train_test(df, split_col):
    from sklearn.model_selection import train_test_split
    x = df.drop(columns= [split_col])
    y = df[split_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=7)
    return x_train, x_test, y_train, y_test


# In[6]:


def simpImputer(df, col, strategy_no):
    """
    strategy_no : 0 for mean, 1 for median, and 2 for mode
    """
    from sklearn.impute import SimpleImputer
    strategy = ["mean", "median", "most_frequent"]
    imputer = SimpleImputer(strategy= strategy[strategy_no])
    df[col] = imputer.fit_transform(df[[col]])


# In[12]:


def knnImputer(df):
    from sklearn.impute import KNNImputer
    import pandas as pd
    imputer = KNNImputer()
    num_df = df.select_dtypes(include= 'number')
    cate_df = df.select_dtypes(exclude= 'number')
    num_df = pd.DataFrame(imputer.fit_transform(num_df), columns= num_df.columns)
    new_df = pd.concat([cate_df.reset_index(), num_df.reset_index()],axis=1)
    return new_df


# In[8]:


def random_value(df, col, no_values, value):
    import numpy as np
    arr = np.array(value)
    for i in range(0,3):
        l = list(df.sample(no_values//3).index)
        df.loc[l, col] = np.random.choice(arr)


# In[4]:


def city_state_country(lat,lon):
    coord = f"{lat}, {lon}"
    geolocator = Nominatim(user_agent="DataExtraction")
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    state = address.get('state', '')
    country = address.get('country', '')
    return city, state, country


# In[6]:


def city(lat,lon):
    from geopy.geocoders import Nominatim
    from time import sleep
    import numpy as np
    coord = f"{lat}, {lon}"
    geolocator = Nominatim(user_agent="user_agent{}".format(np.random.randint(50,1000000)))
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    return city


# In[ ]:




