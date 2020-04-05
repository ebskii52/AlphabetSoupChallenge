#%% [markdown]
## Import, analyze, clean, and preprocess a “real-world” classification dataset.

# %%
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf

# Import our input dataset
charityDF = pd.read_csv('charity_data.csv')
charityDF.head(n=2)

#%%
## What variable(s) are considered the target for your model?
charityDF.IS_SUCCESSFUL

#%%
## What variable(s) are considered to be the features for your model?
## What variable(s) are neither and should be removed from the input data?
charityDF.drop(columns=["NAME","IS_SUCCESSFUL"])

#%%
## Generate our categorical variable list
charityCat = charityDF.dtypes[charityDF.dtypes == "object"].index.tolist()

# Check the number of unique values in each column
charityDF[charityCat].nunique()

#%%
# Print out the Calcification Counts value counts
CLASSIFICATION_counts = charityDF.CLASSIFICATION.value_counts()
replace_Classification = list(CLASSIFICATION_counts[CLASSIFICATION_counts < 100].index)

# Replace in DataFrame
for classify in replace_Classification:
    charityDF.CLASSIFICATION = charityDF.CLASSIFICATION.replace(classify,"Other")

# Check to make sure binning was successful
charityDF.CLASSIFICATION.value_counts()

#%%
# Encode categorical variables using one-hot encoding.
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(charityDF.CLASSIFICATION.values.reshape(-1,1)))

# Rename encoded columns
encode_df.columns = enc.get_feature_names(['CLASSIFICATION'])
encode_df.head()

#%%
# Merge the two DataFrames together and drop the Country column
charityDF = charityDF.merge(encode_df,left_index=True,right_index=True).drop("CLASSIFICATION",1)

#%%
# Print out the Application Counts value counts
APPLICATION_counts = charityDF.APPLICATION_TYPE.value_counts()

## Combine rare categorical values via bucketing.
## Determine which values to replace
replace_APPLICATION= list(APPLICATION_counts[APPLICATION_counts < 100].index)

# Replace in DataFrame
for applications in replace_APPLICATION:
    charityDF.APPLICATION_TYPE = charityDF.APPLICATION_TYPE.replace(applications,"Other")

# Check to make sure binning was successful
charityDF.APPLICATION_TYPE.value_counts()

#%% [markdown]
# Encode categorical variables using one-hot encoding.
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(charityDF.APPLICATION_TYPE.values.reshape(-1,1)))

# Rename encoded columns
encode_df.columns = enc.get_feature_names(['APPLICATION_TYPE'])
encode_df.head()

#%%
# Merge the two DataFrames together and drop the Country column
charityDF = charityDF.merge(encode_df,left_index=True,right_index=True).drop("APPLICATION_TYPE",1)

# %%
## Store the names of all cryptocurrencies on a DataFramed named coins_name, 
## and use the crypto_df.index as the index for this new DataFrame.
charityDF.head()

#%%
# Encode categorical variables using one-hot encoding.
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(charityDF.AFFILIATION.values.reshape(-1,1)))
# Rename encoded columns
encode_df.columns = enc.get_feature_names(['AFFILIATION'])
# Merge the two DataFrames together and drop the Country column
charityDF = charityDF.merge(encode_df,left_index=True,right_index=True).drop("AFFILIATION",1)

#%%
# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(charityDF.ORGANIZATION.values.reshape(-1,1)))
# Rename encoded columns
encode_df.columns = enc.get_feature_names(['ORGANIZATION'])
# Merge the two DataFrames together and drop the Country column
charityDF = charityDF.merge(encode_df,left_index=True,right_index=True).drop("ORGANIZATION",1)

#%%
# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(charityDF.USE_CASE.values.reshape(-1,1)))
# Rename encoded columns
encode_df.columns = enc.get_feature_names(['USE_CASE'])
# Merge the two DataFrames together and drop the Country column
charityDF = charityDF.merge(encode_df,left_index=True,right_index=True).drop("USE_CASE",1)

#%%
# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(charityDF.INCOME_AMT.values.reshape(-1,1)))
# Rename encoded columns
encode_df.columns = enc.get_feature_names(['INCOME_AMT'])
# Merge the two DataFrames together and drop the Country column
charityDF = charityDF.merge(encode_df,left_index=True,right_index=True).drop("INCOME_AMT",1)

#%%
# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(charityDF.SPECIAL_CONSIDERATIONS.values.reshape(-1,1)))
# Rename encoded columns
encode_df.columns = enc.get_feature_names(['SPECIAL_CONSIDERATIONS'])
# Merge the two DataFrames together and drop the Country column
charityDF = charityDF.merge(encode_df,left_index=True,right_index=True).drop("SPECIAL_CONSIDERATIONS",1)


# %%
# Split our preprocessed data into our features and target arrays
y = charityDF["IS_SUCCESSFUL"].values
X = charityDF.drop(["IS_SUCCESSFUL"],1).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

#%%
len(X_train[0])

# %%
# Define the model - deep neural net
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  12
hidden_nodes_layer2 = 8

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()

# %%
# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# %%
# Train the model
fit_model = nn.fit(X_train,y_train,epochs=1000)

#%%
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %%
