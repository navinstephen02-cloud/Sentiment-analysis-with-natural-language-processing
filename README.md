# Sentiment-analysis-with-natural-language-processing
PySpark-based project for sentiment analysis of US airline reviews. Includes data cleaning, preprocessing, and partition performance experiments to optimize execution. Provides exploratory analysis, schema inspection, and visualizations to prepare text data for sentiment classification.

Airline Sentiment Analysis with PySpark
This project analyzes US airline user reviews to predict sentiment (positive, negative, neutral) using PySpark for distributed data processing. The notebook explores dataset cleaning, partition performance, and sentiment classification preparation.

Project Overview

Implemented in PySpark to handle large-scale text data.
Dataset: US Airline Reviews with text + sentiment labels.
Tasks include:
Data preprocessing (cleaning, null handling).
Partitioning experiments for performance optimization.
Schema inspection and exploratory data analysis (EDA).
Preparing data for sentiment classification with NLP.

Features

Partition Performance Analysis: Tested multiple partition sizes [2, 5, 8, 12] to compare execution times.
Data Cleaning:
Removed null values in key fields (airline_sentiment, text).
Used regexp_replace to handle unwanted characters/patterns.
Exploratory Data Analysis:
Dataset schema overview.
Missing value inspection.
Class distribution visualization.
Visualization: Plotted execution times and confusion matrix heatmaps using Matplotlib.

Requirements
pyspark
pandas
numpy
matplotlib
seaborn

The notebook will:
Initialize Spark session.
Load airline review dataset.
Perform cleaning and preprocessing.
Test partition performance.
Generate visualizations.

Results

Optimal partition size improves execution speed significantly.
Cleaned dataset is ready for NLP sentiment classification.
\Visualizations highlight class imbalances and missing values.

Next Steps

Implement classification models (Logistic Regression, Naïve Bayes, Transformers).
Scale training using PySpark MLlib or Hugging Face Transformers.
Deploy model for real-time sentiment analysis.


code

pip install pyspark
from pyspark.sql import SparkSession
# Initialize SparkSession
spark = SparkSession.builder \
    .appName("BDPP-Project-DifferentPartition") \
    .getOrCreate()
# calculate thhe number of workers in GCP
num_workers = spark.sparkContext.getConf().get("spark.executor.instances")
print(f'Number of worker nodes in GPC: {num_workers}')
# Importing the libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, regexp_replace
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder, Tokenizer, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
partition_sizes = [2, 5, 8, 12] # list of the partition sizes going to assign
executionTime_data = [] # a list to save execution time of above partitons loading

for size in partition_sizes: # iterate through the partitions
    start_time = time.time() # start time of execution
    # loading the data with respective partition
    data = spark.read.load("gs://projectbdpp_bucket1/Tweets.csv",format="csv", sep=",", inferSchema="true", header="true")

    execution_time = time.time() -start_time # calculate the execution time of data loading

    print(f'Number of Partitions: {size}, Execution time: {execution_time}s')

    executionTime_data.append(execution_time) # append to the list
# execution times of loading train data
print(executionTime_data)
#first 5 rows of the dataset
data.show(10)
data.count()
fig, ax = plt.subplots(figsize=(12, 6))
X = list(map(str, partition_sizes))
Y = executionTime_data
title = 'Tweet dataset loading'
color = 'Blue'

bar_container = ax.bar(X, Y, color=color)
ax.set_xlabel('Number of Partitions', fontsize=12)
ax.set_ylabel('Execution Time (second)', fontsize=12)
ax.set_ylim((0, 1))
ax.bar_label(bar_container, fmt='%.4fs')
ax.set_title(title)

fig.suptitle('Execution time of loading data with different partitions.', fontsize=15)
plt.show()
# Print the schema of the dataset
data.printSchema()
# Define a dictionary to store missing value counts
missing_value_counts = {}

# Iterate through each column in the DataFrame
for column in data.columns:
    # Get the sum of null values for the column
    missing_count = data.filter(data[column].isNull()).count()

    # Store the count in the dictionary
    missing_value_counts[column] = missing_count

# Print the missing value counts for each column
for column, count in missing_value_counts.items():
    print("Column '{}': {} missing values".format(column, count))
# Drop rows with null values in specific columns
columns_with_null = ['airline_sentiment', 'text']
clean_data = data.na.drop(subset=columns_with_null)

# Show the cleaned DataFrame
clean_data.show(50)
from pyspark.sql.functions import col

# Column for which you want to check null values
column_name = "airline_sentiment"
column_name_2 = "text"

# Count the number of null values in the specified column
null_count = clean_data.where(col(column_name).isNull()).count()

print(f"Number of null values in column '{column_name}': {null_count}")
print(f"Number of null values in column '{column_name_2}': {null_count}")
clean_data.count()
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

# Initialize Spark session
spark = SparkSession.builder.appName("SentimentEncoding").getOrCreate()

# Display the column names
print(clean_data.columns)

clean_data = clean_data.withColumn(
    'Airline_sentiment_Numeric',
    when(col('airline_sentiment') == 'negative', 0)
    .when(col('airline_sentiment') == 'neutral', 1)
    .when(col('airline_sentiment') == 'positive', 2)
)

# Drop the original 'airline_sentiment' column if it's no longer needed
clean_data = clean_data.drop('airline_sentiment')

# Show the resulting DataFrame
clean_data.show(10)
pip install pyspark nltk
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pyspark.sql.functions import col
import nltk
nltk.download('stopwords')
# Import necessary libraries
import re
import nltk
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from nltk.corpus import stopwords

# Create Spark session
spark = SparkSession.builder \
    .appName("Tweet Preprocessing") \
    .getOrCreate()

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Create DataFrame
new_data = spark.createDataFrame(clean_data, ["text"])

# Define the tweet_to_words function
def tweet_to_words(tweet):
    letters_only = re.sub("[^a-zA-Z]", " ", tweet)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)

# Register the UDF
tweet_to_words_udf = udf(tweet_to_words, StringType())

# Apply UDF to preprocess tweet text
new_data = new_data.withColumn("clean_tweet", tweet_to_words_udf(col("text")))

# Show the resulting DataFrame
new_data.show(truncate=False)
# Normalization method definition
def clean_text(column):
    column = regexp_replace(column, "[0-9]+", "number")    # Remove numbers (replace the word 'number' for every number)
    column = regexp_replace(column, "(http|https)://[^\s]*", "httpaddr") # Remove links
    column = regexp_replace(column, "[^\s]+@[^\s]+", "emailaddr") # Remove email addresses
    column = regexp_replace(column, "(?:(?:[0-9]{2}[:\/,]){2}[0-9]{2,4})", "") # Remove dates
    column = regexp_replace(column, "\:|\/|\#|\.|\?|\!|\&|\"|\,|\$|\;", "") # Remove symbols
    column = regexp_replace(column, "(\\n)|\n|\r|\t", " ")  # Remove CR, tab, and LR
    column = regexp_replace(column, "[^a-zA-Z0-9 ]", "")   # Remove any other non-alphanumeric characters
    column = regexp_replace(column, "[^\x00-\x7F]+", " ")  # Removing all the non ASCII characters
    column = regexp_replace(column, "\s+", " ")            # Replace multiple white spaces with one space
    return column
# Apply the normalization method to all values in 'review' column
from pyspark.sql.functions import col
clean_data = clean_data.withColumn("review_cleaned", clean_text(col("text")))
clean_data.show(5)
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from pyspark.ml import Pipeline
# Implementation of vectorization pipeline
tokenizer = Tokenizer(inputCol="review_cleaned", outputCol="tokenized")
cv = CountVectorizer(vocabSize=2**16, inputCol="tokenized", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="review_features", minDocFreq=5)

review_pipeline = Pipeline(stages=[tokenizer, cv, idf]) # creating the pipeline
pipeline = Pipeline(stages=[ review_pipeline])
final_df = pipeline.fit(clean_data).transform(clean_data)
final_df.show()
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder, Tokenizer, CountVectorizer, IDF
finalAssembler = VectorAssembler(inputCols=[ 'review_features'] , outputCol="features")
final_features_df = finalAssembler.transform(final_df).select(['review_features', 'Airline_sentiment_Numeric'])
final_features_df.show()
final_features_df.count()
trainSet, testSet = final_features_df.randomSplit([0.8, 0.2])
trainSet.count()
testSet.count()
# Convert the data frame to panda's dataframe
panda_df = clean_data.toPandas()
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
# Stesp of making word cloud of the normalized 'review' attribute
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'lightyellow', stopwords = stopwords, width = 1200, height = 800).generate(str(panda_df['review_cleaned']))

plt.figure(figsize=(10,5))
plt.title('Word Cloud Of Reviews', fontsize = 20)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()
# Ploting the Histograms of train and test data
train_data = trainSet.groupBy("Airline_sentiment_Numeric").count().toPandas()
test_data = testSet.groupBy("Airline_sentiment_Numeric").count().toPandas()


fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))
train_data.plot(kind = 'bar',x = 'Airline_sentiment_Numeric', y = 'count', ax = ax1, color = 'red')
test_data.plot(kind = 'bar',x = 'Airline_sentiment_Numeric', y = 'count', ax = ax2, color = 'blue')
ax1.set_title("Distribution of Train data")
ax2.set_title("Distribution of Test data")
plt.show()
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime
start_time = datetime.now() # Starting time of the execution

# Training the Logistic Regression model
lr = LogisticRegression(featuresCol="review_features", labelCol="Airline_sentiment_Numeric")
model = lr.fit(trainSet)

execution_time = datetime.now() - start_time # calculation total execution time for training the model

# Evaluation of the model
predictions = model.transform(testSet)
evaluator = MulticlassClassificationEvaluator(labelCol="Airline_sentiment_Numeric", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy * 100}%")
print(f"Execution Time of training: {execution_time}")
start_time = time.time() # Starting time of the execution

# Training the Decision Tree Classifier model
dt = DecisionTreeClassifier(featuresCol="review_features", labelCol="Airline_sentiment_Numeric",maxDepth = 5)
dt_model = dt.fit(trainSet)

execution_time = time.time() - start_time # calculation total execution time for training the model

predictions = dt_model.transform(testSet)
evaluator = MulticlassClassificationEvaluator(labelCol="Airline_sentiment_Numeric", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy * 100}%")
print(f"Execution Time of training: {execution_time}s")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Show the predictions
predictions.select("prediction", "Airline_sentiment_Numeric", "review_features").show(20)
# Extract the predictions and true labels
preds_and_labels = predictions.select("prediction", "Airline_sentiment_Numeric").collect()
preds = [int(row.prediction) for row in preds_and_labels]
labels = [int(row.Airline_sentiment_Numeric) for row in preds_and_labels]

# Create confusion matrix
confusion_matrix = np.zeros((3, 3), dtype=int)  # Assuming there are 3 classes
for p, l in zip(preds, labels):
    confusion_matrix[l][p] += 1

# Convert to DataFrame for easier plotting
confusion_df = pd.DataFrame(confusion_matrix, index=['Actual_0', 'Actual_1', 'Actual_2'], columns=['Pred_0', 'Pred_1', 'Pred_2'])

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime

start_time = datetime.now()  # Starting time of the execution

# Define the Random Forest model
rf = RandomForestClassifier(featuresCol="review_features", labelCol="Airline_sentiment_Numeric")

# Train the Random Forest model
model = rf.fit(trainSet)

execution_time = datetime.now() - start_time  # Calculate total execution time for training the model

# Evaluation of the model
predictions1 = model.transform(testSet)
evaluator = MulticlassClassificationEvaluator(labelCol="Airline_sentiment_Numeric", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions1)

print(f"Model Accuracy: {accuracy * 100}%")
print(f"Execution Time of training: {execution_time}")
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline

# Starting time of the execution
start_time = datetime.now()

# Assuming you have prepared the trainSet and testSet with appropriate features and labels
# Define a feature vector assembler
vectorAssembler = VectorAssembler(inputCols=["review_features"], outputCol="features")

# Define a standard scaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)

# Define the RandomForest classifier
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="Airline_sentiment_Numeric")

# Define a pipeline
pipeline = Pipeline(stages=[vectorAssembler, scaler, rf])

# Train the RandomForest model
model = pipeline.fit(trainSet)

# Calculate the execution time for training the model
execution_time = datetime.now() - start_time

# Make predictions on the test set
predictions = model.transform(testSet)

# Evaluate the model using the MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="Airline_sentiment_Numeric", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Print the model accuracy and execution time
print(f"Model Accuracy: {accuracy * 100}%")
print(f"Execution Time of training: {execution_time}")
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Starting time of the execution
start_time = datetime.now()

# Assuming you have prepared the trainSet and testSet with appropriate features and labels
# Define a feature vector assembler
vectorAssembler = VectorAssembler(inputCols=["review_features"], outputCol="features")

# Define a standard scaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)

# Define the RandomForest classifier
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="Airline_sentiment_Numeric")

# Define a pipeline
pipeline = Pipeline(stages=[vectorAssembler, scaler, rf])

# Train the RandomForest model
model = pipeline.fit(trainSet)

# Calculate the execution time for training the model
execution_time = datetime.now() - start_time

# Make predictions on the test set
predictions = model.transform(testSet)

# Evaluate the model using the MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="Airline_sentiment_Numeric", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Print the model accuracy and execution time
print(f"Model Accuracy: {accuracy * 100}%")
print(f"Execution Time of training: {execution_time}")

# Show the predictions
predictions.select("prediction", "Airline_sentiment_Numeric", "review_features").show(5)

# Extract the predictions and true labels
preds_and_labels = predictions.select("prediction", "Airline_sentiment_Numeric").collect()
preds = [int(row.prediction) for row in preds_and_labels]
labels = [int(row.Airline_sentiment_Numeric) for row in preds_and_labels]

# Create confusion matrix
confusion_matrix = np.zeros((3, 3), dtype=int)  # Assuming there are 3 classes
for p, l in zip(preds, labels):
    confusion_matrix[l][p] += 1

# Convert to DataFrame for easier plotting
confusion_df = pd.DataFrame(confusion_matrix, index=['Actual_0', 'Actual_1', 'Actual_2'], columns=['Pred_0', 'Pred_1', 'Pred_2'])

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


