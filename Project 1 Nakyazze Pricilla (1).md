```python
# This system recommends restocking of fruits based on a user-item matrix representing
# fruit ratings given by users (on different days of the week) to items (the fruits). 
# Since not every item has a rating for everyday, we build a recommendation system to
# predict the missing values 
# — and from there, we can recommend items.

```


```python

```


```python
import pandas as pd
import numpy as np

data = {
    "bananas":       [3, 4, 3, np.nan, 5, 9, np.nan],
    "Apples":        [7, 2, np.nan, 5, 4, 8, 4],
    "Watermelon":    [np.nan, 5, 3, np.nan, 5, 8, 3],
    "PassionFruit":  [6, np.nan, 5, np.nan, 9, 9, 2],
    "SugarCane":     [6, 8, np.nan, 5, 7, 4, np.nan]
}
index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df = pd.DataFrame(data, index=index)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bananas</th>
      <th>Apples</th>
      <th>Watermelon</th>
      <th>PassionFruit</th>
      <th>SugarCane</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Monday</th>
      <td>3.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Friday</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Saturday</th>
      <td>9.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Sunday</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Break your ratings into separate training and test datasets.
```


```python
# non-missing values
non_missing_positions = [(i, j) for i in range(df.shape[0]) for j in range(df.shape[1]) if not pd.isna(df.iat[i, j])]

# 20% for test set
np.random.seed(42)
test_sample_size = int(0.2 * len(non_missing_positions))
test_indices = np.random.choice(len(non_missing_positions), size=test_sample_size, replace=False)

#  dataframes
train_df = df.copy()
test_df = df.copy()

# Populate test_df with only the held-out values, and mask them in train_df
for idx in range(len(non_missing_positions)):
    i, j = non_missing_positions[idx]
    if idx in test_indices:
        train_df.iat[i, j] = np.nan  # remove from train
    else:
        test_df.iat[i, j] = np.nan  # remove from test
```


```python
print("Original ratings:", df.count().sum())
print("Train ratings:   ", train_df.count().sum())
print("Test ratings:    ", test_df.count().sum())
```

    Original ratings: 26
    Train ratings:    21
    Test ratings:     5



```python
# Ensure numeric values
baseline_pred = baseline_pred.astype(float)

# Generate top recommendation per user
recommendations = {}

for user in baseline_pred.index:
    # Identify which items were missing in train_df
    missing_items = train_df.loc[user][train_df.loc[user].isna()].index.tolist()
    
    if missing_items:
        # Slice the predicted ratings for missing items only
        predicted_ratings = baseline_pred.loc[user, missing_items]
        
        # Get the item with the highest predicted rating
        recommended_item = predicted_ratings.idxmax()
        predicted_rating = predicted_ratings.max()
        
        recommendations[user] = (recommended_item, predicted_rating)

# Display recommendations
print("Top Recommendation per User:")
for user, (item, rating) in recommendations.items():
    print(f"  {user}: Recommend '{item}' with predicted rating {rating:.2f}")# Ensure numeric values
baseline_pred = baseline_pred.astype(float)

# Generate top recommendation per user
recommendations = {}

for user in baseline_pred.index:
    # Identify which items were missing in train_df
    missing_items = train_df.loc[user][train_df.loc[user].isna()].index.tolist()
    
    if missing_items:
        # Slice the predicted ratings for missing items only
        predicted_ratings = baseline_pred.loc[user, missing_items]
        
        # Get the item with the highest predicted rating
        recommended_item = predicted_ratings.idxmax()
        predicted_rating = predicted_ratings.max()
        
        recommendations[user] = (recommended_item, predicted_rating)

# Display recommendations
print("Top Recommendation per User:")
for user, (item, rating) in recommendations.items():
    print(f"  {user}: Recommend '{item}' with predicted rating {rating:.2f}")
```

    Top Recommendation per User:
      Monday: Recommend 'bananas' with predicted rating 6.47
      Tuesday: Recommend 'PassionFruit' with predicted rating 5.02
      Wednesday: Recommend 'SugarCane' with predicted rating 4.21
      Thursday: Recommend 'PassionFruit' with predicted rating 5.27
      Friday: Recommend 'PassionFruit' with predicted rating 5.52
      Sunday: Recommend 'SugarCane' with predicted rating 3.21
    Top Recommendation per User:
      Monday: Recommend 'bananas' with predicted rating 6.47
      Tuesday: Recommend 'PassionFruit' with predicted rating 5.02
      Wednesday: Recommend 'SugarCane' with predicted rating 4.21
      Thursday: Recommend 'PassionFruit' with predicted rating 5.27
      Friday: Recommend 'PassionFruit' with predicted rating 5.52
      Sunday: Recommend 'SugarCane' with predicted rating 3.21



```python
recommendations = {}

for user in baseline_pred.index:
    missing_items = train_df.loc[user][train_df.loc[user].isna()].index
    if len(missing_items) > 0:
        recommended_item = baseline_pred.loc[user, missing_items].idxmax()
        predicted_rating = baseline_pred.loc[user, recommended_item]
        recommendations[user] = (recommended_item, predicted_rating)

print("Top Recommendation per User:")
for user, (item, rating) in recommendations.items():
    print(f"  {user}: Recommend '{item}' with predicted rating {rating:.2f}")
```

    Top Recommendation per User:
      Monday: Recommend 'bananas' with predicted rating 6.47
      Tuesday: Recommend 'PassionFruit' with predicted rating 5.02
      Wednesday: Recommend 'SugarCane' with predicted rating 4.21
      Thursday: Recommend 'PassionFruit' with predicted rating 5.27
      Friday: Recommend 'PassionFruit' with predicted rating 5.52
      Sunday: Recommend 'SugarCane' with predicted rating 3.21



```python
# Compute the raw average for each user (i.e., each row in train_df)
user_raw_averages = train_df.mean(axis=1, skipna=True)

# Display the results
for user, avg in zip(train_df.index, user_raw_averages):
    print(f"{user}: {avg:.2f}")
```

    Monday: 6.33
    Tuesday: 4.75
    Wednesday: 4.00
    Thursday: 5.00
    Friday: 5.25
    Saturday: 7.60
    Sunday: 3.00



```python
#Break your ratings into separate training and test datasets.
```


```python
print(train_df)
```

               bananas  Apples  Watermelon  PassionFruit  SugarCane
    Monday         NaN     7.0         NaN           6.0        6.0
    Tuesday        4.0     2.0         5.0           NaN        8.0
    Wednesday      NaN     NaN         3.0           5.0        NaN
    Thursday       NaN     NaN         NaN           NaN        5.0
    Friday         5.0     4.0         5.0           NaN        7.0
    Saturday       9.0     8.0         8.0           9.0        4.0
    Sunday         NaN     4.0         NaN           2.0        NaN



```python
print(baseline_pred)
```

                bananas    Apples  Watermelon  PassionFruit  SugarCane
    Monday     6.466667  5.946667    6.183333      6.600000   6.546667
    Tuesday    4.883333  4.363333    4.600000      5.016667   4.963333
    Wednesday  4.133333  3.613333    3.850000      4.266667   4.213333
    Thursday   5.133333  4.613333    4.850000      5.266667   5.213333
    Friday     5.383333  4.863333    5.100000      5.516667   5.463333
    Saturday   7.733333  7.213333    7.450000      7.866667   7.813333
    Sunday     3.133333  2.613333    2.850000      3.266667   3.213333



```python
print(test_df)
```

               bananas  Apples  Watermelon  PassionFruit  SugarCane
    Monday         3.0     NaN         NaN           NaN        NaN
    Tuesday        NaN     NaN         NaN           NaN        NaN
    Wednesday      3.0     NaN         NaN           NaN        NaN
    Thursday       NaN     5.0         NaN           NaN        NaN
    Friday         NaN     NaN         NaN           9.0        NaN
    Saturday       NaN     NaN         NaN           NaN        NaN
    Sunday         NaN     NaN         3.0           NaN        NaN



```python
test_entries = [(index, col, test_df.loc[index, col])
                for index in test_df.index
                for col in test_df.columns
                if not pd.isna(test_df.loc[index, col])]

# Example output
for user, item, rating in test_entries:
    print(f"{user} rated {item} = {rating}")
```

    Monday rated bananas = 3.0
    Wednesday rated bananas = 3.0
    Thursday rated Apples = 5.0
    Friday rated PassionFruit = 9.0
    Sunday rated Watermelon = 3.0



```python
#mean
global_mean = train_df.stack().mean()
print(f"Global mean rating: {global_mean:.2f}")
```

    Global mean rating: 5.52



```python
#Bias

user_bias = train_df.sub(global_mean, axis=0).mean(axis=1, skipna=True)
print(user_bias)
```

    Monday       0.809524
    Tuesday     -0.773810
    Wednesday   -1.523810
    Thursday    -0.523810
    Friday      -0.273810
    Saturday     2.076190
    Sunday      -2.523810
    dtype: float64



```python
# Subtract user bias from ratings to isolate item bias
adjusted_for_user = train_df.copy()
for user in train_df.index:
    adjusted_for_user.loc[user] = train_df.loc[user] - user_bias[user]

item_bias = adjusted_for_user.sub(global_mean, axis=0).mean(skipna=True)
print(item_bias)
```

    bananas         0.133333
    Apples         -0.386667
    Watermelon     -0.150000
    PassionFruit    0.266667
    SugarCane       0.213333
    dtype: float64



```python
baseline_pred = pd.DataFrame(index=train_df.index, columns=train_df.columns)

for user in train_df.index:
    for item in train_df.columns:
        bu = user_bias.get(user, 0)
        bi = item_bias.get(item, 0)
        baseline_pred.loc[user, item] = global_mean + bu + bi
```


```python
# test_df = values that were removed from df to create train_df
test_df = df.copy()

# For all entries that are NOT missing in df but ARE missing in train_df,
# keep the value; otherwise, set to NaN
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if pd.notna(df.iat[i, j]) and pd.isna(train_df.iat[i, j]):
            # Keep it in test_df
            continue
        else:
            # Set to NaN (either was missing originally or still present in train_df)
            test_df.iat[i, j] = np.nan
```


```python
print("Total held-out ratings in test_df:", test_df.count().sum())
```

    Total held-out ratings in test_df: 5



```python
# Ensure numeric values
baseline_pred = baseline_pred.astype(float)

# Generate top recommendation per user
recommendations = {}

for user in baseline_pred.index:
    # Identify which items were missing in train_df
    missing_items = train_df.loc[user][train_df.loc[user].isna()].index.tolist()
    
    if missing_items:
        # Slice the predicted ratings for missing items only
        predicted_ratings = baseline_pred.loc[user, missing_items]
        
        # Get the item with the highest predicted rating
        recommended_item = predicted_ratings.idxmax()
        predicted_rating = predicted_ratings.max()
        
        recommendations[user] = (recommended_item, predicted_rating)

# Display recommendations
print("Top Recommendation per User:")
for user, (item, rating) in recommendations.items():
    print(f"  {user}: Recommend '{item}' with predicted rating {rating:.2f}")
```

    Top Recommendation per User:
      Monday: Recommend 'bananas' with predicted rating 6.47
      Tuesday: Recommend 'PassionFruit' with predicted rating 5.02
      Wednesday: Recommend 'SugarCane' with predicted rating 4.21
      Thursday: Recommend 'PassionFruit' with predicted rating 5.27
      Friday: Recommend 'PassionFruit' with predicted rating 5.52
      Sunday: Recommend 'SugarCane' with predicted rating 3.21



```python
baseline_pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bananas</th>
      <th>Apples</th>
      <th>Watermelon</th>
      <th>PassionFruit</th>
      <th>SugarCane</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Monday</th>
      <td>6.466667</td>
      <td>5.946667</td>
      <td>6.183333</td>
      <td>6.600000</td>
      <td>6.546667</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>4.883333</td>
      <td>4.363333</td>
      <td>4.600000</td>
      <td>5.016667</td>
      <td>4.963333</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>4.133333</td>
      <td>3.613333</td>
      <td>3.850000</td>
      <td>4.266667</td>
      <td>4.213333</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>5.133333</td>
      <td>4.613333</td>
      <td>4.850000</td>
      <td>5.266667</td>
      <td>5.213333</td>
    </tr>
    <tr>
      <th>Friday</th>
      <td>5.383333</td>
      <td>4.863333</td>
      <td>5.100000</td>
      <td>5.516667</td>
      <td>5.463333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Raw Average mean
# Using your training data,
# calculate the raw average (mean) rating for every user-item combination
```


```python
global_mean = train_df.stack().mean()
print(f"Global average rating: {global_mean:.2f}")
```

    Global average rating: 5.52



```python

raw_avg_pred = train_df.copy()
raw_avg_pred = raw_avg_pred.fillna(global_mean)
```


```python
print(raw_avg_pred)
```

               bananas   Apples  Watermelon  PassionFruit  SugarCane
    Monday     5.52381  7.00000     5.52381       6.00000    6.00000
    Tuesday    4.00000  2.00000     5.00000       5.52381    8.00000
    Wednesday  5.52381  5.52381     3.00000       5.00000    5.52381
    Thursday   5.52381  5.52381     5.52381       5.52381    5.00000
    Friday     5.00000  4.00000     5.00000       5.52381    7.00000
    Saturday   9.00000  8.00000     8.00000       9.00000    4.00000
    Sunday     5.52381  4.00000     5.52381       2.00000    5.52381



```python
user_avg = train_df.mean(axis=1, skipna=True)
user_avg = user_avg.fillna(global_mean)  # Replace NaN user averages with global mean

item_avg = train_df.mean(axis=0, skipna=True)
item_avg = item_avg.fillna(global_mean) 
```


```python
print(user_avg)
```

    Monday       6.333333
    Tuesday      4.750000
    Wednesday    4.000000
    Thursday     5.000000
    Friday       5.250000
    Saturday     7.600000
    Sunday       3.000000
    dtype: float64



```python
#RMSE Calculate the RMSE for raw average for both your training data and your test data.
```


```python
raw_avg_pred = raw_avg_pred.astype(float)
```


```python
def compute_rmse(pred_df, actual_df):
    # Replace NaN with 0 in both DataFrames
    pred_df = pred_df.fillna(0)
    actual_df = actual_df.fillna(0)

    pred_vals = pred_df.values.flatten()
    actual_vals = actual_df.values.flatten()

    num_compared = len(actual_vals)
    print(f"Comparing {num_compared} data points (NaNs replaced with 0).")

    return np.sqrt(np.mean((pred_vals - actual_vals) ** 2))
```


```python
rmse_raw_test = compute_rmse(raw_avg_pred, test_df)
print(f"RMSE (Raw Average) on test data: {rmse_raw_test:.3f}")
```

    Comparing 35 data points (NaNs replaced with 0).
    RMSE (Raw Average) on test data: 5.441



```python
pred_df = raw_avg_pred.fillna(0)
actual_df = train_df.fillna(0)
```


```python
import numpy as np

def compute_rmse_no_mask(pred_df, actual_df):
    pred_vals = pred_df.values.flatten()
    actual_vals = actual_df.values.flatten()
    return np.sqrt(np.mean((pred_vals - actual_vals) ** 2))
```


```python
rmse_raw_train = compute_rmse_no_mask(pred_df, actual_df)
print(f" RMSE (Raw Average) on training data [NaNs treated as 0]: {rmse_raw_train:.3f}")
```

     RMSE (Raw Average) on training data [NaNs treated as 0]: 3.494



```python
global_mean = train_df.stack().mean()
print("Global mean:", global_mean)
```

    Global mean: 5.523809523809524



```python
raw_avg_pred = train_df.fillna(global_mean)
```


```python

```


```python
import pandas as pd
import numpy as np

# Step 1: Compute global mean from training data
global_mean = train_df.stack().mean()
print(f"Global Mean Rating: {global_mean:.2f}")

# Step 2: Create raw average prediction matrix
raw_avg_pred = train_df.copy().fillna(global_mean)
```

    Global Mean Rating: 5.52



```python
# Using your training data, calculate the bias for each user and each item.
```


```python

temp_df = train_df.copy()

for user in train_df.index:
    temp_df.loc[user] = train_df.loc[user] - user_bias[user]

# Now compute item bias from the adjusted DataFrame
item_bias = temp_df.sub(global_mean, axis=0).mean(skipna=True)

```


```python
print("Global Mean:", round(global_mean, 3))
print("\nUser Biases:")
print(user_bias.round(3))

print("\nItem Biases:")
print(item_bias.round(3))
```

    Global Mean: 5.524
    
    User Biases:
    Monday       0.810
    Tuesday     -0.774
    Wednesday   -1.524
    Thursday    -0.524
    Friday      -0.274
    Saturday     2.076
    Sunday      -2.524
    dtype: float64
    
    Item Biases:
    bananas         0.133
    Apples         -0.387
    Watermelon     -0.150
    PassionFruit    0.267
    SugarCane       0.213
    dtype: float64



```python
# calculate the baseline predictors
#for every user-item combination.
```


```python
baseline_pred = pd.DataFrame(index=train_df.index, columns=train_df.columns)

# Calculate baseline predictor for each user-item pair
for user in train_df.index:
    for item in train_df.columns:
        bu = user_bias.get(user, 0)
        bi = item_bias.get(item, 0)
        baseline_pred.loc[user, item] = global_mean + bu + bi
```


```python
baseline_pred = baseline_pred.astype(float)
```


```python
print("Baseline predictor matrix (first few rows):")
print(baseline_pred.round(2).head())
```

    Baseline predictor matrix (first few rows):
               bananas  Apples  Watermelon  PassionFruit  SugarCane
    Monday        6.47    5.95        6.18          6.60       6.55
    Tuesday       4.88    4.36        4.60          5.02       4.96
    Wednesday     4.13    3.61        3.85          4.27       4.21
    Thursday      5.13    4.61        4.85          5.27       5.21
    Friday        5.38    4.86        5.10          5.52       5.46



```python
import numpy as np

def compute_rmse(pred_df, actual_df):
    # Only evaluate on positions where actual data exists
    mask = ~pd.isna(actual_df)
    pred_vals = pred_df[mask].values.flatten()
    actual_vals = actual_df[mask].values.flatten()

    if len(pred_vals) == 0:
        print(" No values to evaluate.")
        return np.nan

    return np.sqrt(np.mean((pred_vals - actual_vals) ** 2))
```


```python
print("Known ratings in test set:", test_df.count().sum())
print("Known ratings in train set:", train_df.count().sum())
```

    Known ratings in test set: 5
    Known ratings in train set: 21



```python
print("Number of known ratings in test_df:", test_df.count().sum())
```

    Number of known ratings in test_df: 5



```python
print(baseline_pred.index.equals(test_df.index))   # Should be True
print(baseline_pred.columns.equals(test_df.columns))
```

    True
    True



```python
print(baseline_pred.dtypes) # why am i getting nan values?
```

    bananas         float64
    Apples          float64
    Watermelon      float64
    PassionFruit    float64
    SugarCane       float64
    dtype: object



```python
print(baseline_pred.index.equals(test_df.index))   # Should be True
print(baseline_pred.columns.equals(test_df.columns)) 
```

    True
    True



```python
mask = ~pd.isna(test_df)
print("Number of values to evaluate:", mask.sum().sum())
```

    Number of values to evaluate: 5



```python
baseline_pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bananas</th>
      <th>Apples</th>
      <th>Watermelon</th>
      <th>PassionFruit</th>
      <th>SugarCane</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Monday</th>
      <td>6.466667</td>
      <td>5.946667</td>
      <td>6.183333</td>
      <td>6.600000</td>
      <td>6.546667</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>4.883333</td>
      <td>4.363333</td>
      <td>4.600000</td>
      <td>5.016667</td>
      <td>4.963333</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>4.133333</td>
      <td>3.613333</td>
      <td>3.850000</td>
      <td>4.266667</td>
      <td>4.213333</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>5.133333</td>
      <td>4.613333</td>
      <td>4.850000</td>
      <td>5.266667</td>
      <td>5.213333</td>
    </tr>
    <tr>
      <th>Friday</th>
      <td>5.383333</td>
      <td>4.863333</td>
      <td>5.100000</td>
      <td>5.516667</td>
      <td>5.463333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# RMSE for the baseline predictors for both your training data and your test data
```


```python
import numpy as np

# Collect predicted and actual ratings
predicted_ratings = []
actual_ratings = []

for user, item, actual in test_entries:
    pred = baseline_pred.loc[user, item]
    predicted_ratings.append(pred)
    actual_ratings.append(actual)

# Convert to arrays
predicted_ratings = np.array(predicted_ratings)
actual_ratings = np.array(actual_ratings)

# Compute RMSE
rmse = np.sqrt(np.mean((predicted_ratings - actual_ratings) ** 2))
print(f" RMSE on test data: {rmse:.3f}")
```

     RMSE on test data: 2.263



```python
import numpy as np

# Extract non-NaN training entries
train_entries = [
    (user, item, train_df.loc[user, item])
    for user in train_df.index
    for item in train_df.columns
    if not pd.isna(train_df.loc[user, item])
]

# Compare predictions to actuals
predicted_ratings_train = []
actual_ratings_train = []

for user, item, actual in train_entries:
    pred = baseline_pred.loc[user, item]
    predicted_ratings_train.append(pred)
    actual_ratings_train.append(actual)

# Convert to arrays and compute RMSE
predicted_ratings_train = np.array(predicted_ratings_train)
actual_ratings_train = np.array(actual_ratings_train)

rmse_train = np.sqrt(np.mean((predicted_ratings_train - actual_ratings_train) ** 2))
print(f" RMSE on training data: {rmse_train:.3f}")
```

     RMSE on training data: 1.447



```python
# summarize
#Dataset	RMSE (Root Mean Square Error)
#Training Set	 1.447
#Test Set	     2.263
```


```python
# bias
##User	  Bias	    Interpretation
#Monday	   +0.810	   Rates higher than average
##Tuesday	–0.774	    Rates lower than average
#Wednesday	–1.524	    Rates much lower than average
##Thursday	–0.524	     Slightly lower ratings
#Friday	    –0.274	     Neutral/slightly low
#Saturday	+2.076	     Very generous rater
#Sunday	    –2.524	      Very strict/low rating
```


```python
#item bias
#Item	    Bias	 Interpretation
#bananas	    +0.133	 Slightly better than average
#Apples	    –0.387	 Rated lower than average
#Watermelon	–0.150	   Slightly below average
#PassionFruit  +0.267	Well-liked
#SugarCane	 +0.213	     Also well-liked
```


```python
This baseline model captures trends in how users and items deviate from the average, 
but doesn’t capture interaction effects or personalized preferences well 
which is why RMSE is higher on test data.
```


      Cell In[86], line 2
        but doesn’t capture interaction effects or personalized preferences well
                 ^
    SyntaxError: invalid character '’' (U+2019)




```python

```


```python

```
