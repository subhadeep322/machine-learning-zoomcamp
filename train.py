import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor

# 1. Load Data
print("Loading data...")
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

# 2. Clean Data
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# 3. Validation Framework (Train/Test split)
print("Splitting data...")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_full_train = df_full_train.reset_index(drop=True)

y_full_train = np.log1p(df_full_train.charges.values)
del df_full_train['charges']

# 4. Feature Engineering
print("Vectorizing data...")
full_train_dicts = df_full_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(full_train_dicts)

# 5. Train Final Model (Using the best depth: 5)
print("Training the Random Forest model...")
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_full_train, y_full_train)

# 6. Export Model
output_file = 'model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"Success! Model saved to {output_file}")