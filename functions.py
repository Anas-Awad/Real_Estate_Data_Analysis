#!/usr/bin/env python
# coding: utf-8

# In[34]:


def get_out_liers(df, name):
    d = df[name].describe()
    Q1, Q3 = d['25%'], d['75%']
    iqr = Q3 - Q1
    min_whis, max_whis = (Q1 - 1.5 * iqr), (Q3 + 1.5 * iqr)
    out = list(df[(df[name] < min_whis) | (df[name] > max_whis)].index)
    return out


# In[35]:


def substitute(df, col, places, value):
    df.loc[places, col] = value


# In[47]:


def labelE(df,col):
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    le = LabelEncoder()
    le.fit(df[col])
    col = pd.Series(data=le.transform(df[col]), index=df[col].index)
    return col, le


# In[37]:


def OneHotE(df, col):
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    encoder = OneHotEncoder(sparse=False, drop='first')
    transformed = pd.DataFrame(encoder.fit_transform(df[[col]]), columns = encoder.get_feature_names_out())
    df2 = pd.concat([df.drop(columns= [col]), transformed],axis =1)
    return df2,encoder


# In[49]:


def BinaryE(df, col):
    from category_encoders import BinaryEncoder
    import pandas as pd
    bi = BinaryEncoder(drop_invariant=True)
    encoded = bi.fit_transform(df[col])
    df2 = pd.concat([df.drop(columns= [col]), encoded],axis =1)
    return df2,bi


# In[39]:


def train_test(df, split_col):
    from sklearn.model_selection import train_test_split
    x = df.drop(columns= [split_col])
    y = df[split_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    return x_train, x_test, y_train, y_test


# In[40]:


def simpImputer(df, col, strategy_no):
    """
    strategy_no : 0 for mean, 1 for median, and 2 for mode
    """
    from sklearn.impute import SimpleImputer
    strategy = ["mean", "median", "most_frequent"]
    imputer = SimpleImputer(strategy= strategy[strategy_no])
    df[col] = imputer.fit_transform(df[[col]])


# In[41]:


def knnImputer(df):
    from sklearn.impute import KNNImputer
    import pandas as pd
    imputer = KNNImputer()
    num_df = df.select_dtypes(include= 'number')
    cate_df = df.select_dtypes(exclude= 'number')
    num_df = pd.DataFrame(imputer.fit_transform(num_df), columns= num_df.columns)
    new_df = pd.concat([cate_df.reset_index(), num_df.reset_index()],axis=1)
    return new_df


# In[42]:


def random_value(df, col, no_values, value):
    import numpy as np
    arr = np.array(value)
    for i in range(0,3):
        l = list(df.sample(no_values//3).index)
        df.loc[l, col] = np.random.choice(arr)


# In[43]:


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


# In[44]:


def season_of_date(date):
    import pandas as pd
    try:
        year = str(date.year)
        seasons = {'spring': pd.date_range(start=year+'-03-21', end= year+'-06-20'),
                   'summer': pd.date_range(start=year+'-06-21', end=year+'-09-22'),
                   'autumn': pd.date_range(start=year+'-09-23', end=year+'-12-20')}
        if date in seasons['spring']:
            return 'spring'
        if date in seasons['summer']:
            return 'summer'
        if date in seasons['autumn']:
            return 'autumn'
        else:
            return 'winter'
    except:
        return None


# In[45]:


def binning_price(p):
    if p>0 and p<=5:
        return 'Low'
    elif p>5 and p<10:
        return 'Medium'
    else:
        return 'High'


# In[ ]:




