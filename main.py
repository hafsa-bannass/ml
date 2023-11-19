from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import yagmail

### Create a spark session
spark = SparkSession.builder.appName("SetimentAnalysis").getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 5 * 1024 * 1024)  # 5 MiB

### Import the dataset
dataset = spark.read.csv("Twitter_Data.csv")
dataset = dataset.withColumnRenamed("_c0", "tweet").withColumnRenamed("_c1", "label").na.drop()

#Show the dataset
dataset.show(10)

### Features preparations

# Tokenize the tweets
tokenize = Tokenizer(inputCol="tweet", outputCol="words")
tokenized = tokenize.transform(dataset)
tokenized.select("tweet","words").show(10)


# Remove the unimportant words
stopRemove = StopWordsRemover(inputCol="words", outputCol="filtered")
stopRemoved = stopRemove.transform(tokenized)
stopRemoved.select("tweet","filtered").show(10)

# Vectorize words
vectorize = CountVectorizer(inputCol="filtered", outputCol="countVec")
counted_vec = vectorize.fit(stopRemoved).transform(stopRemoved)
counted_vec.select("tweet","countVec").show(10)



idf = IDF(inputCol="countVec", outputCol="features")
rescale = idf.fit(counted_vec).transform(counted_vec)

# Index the label column
indexer = StringIndexer(inputCol="label", outputCol="Index")
indexed = indexer.fit(rescale).transform(rescale)
indexed.show()

### Prediction & evaluation

# Train and test split
(training, testing) = indexed.randomSplit([0.8, 0.2], seed=42)

# Fit the model
LoReg = LogisticRegression(featuresCol="features", labelCol="Index")
LoRegModel = LoReg.fit(training)

# test the model
test_pred = LoRegModel.transform(testing)
test_pred.select("features", "label", "Index", "prediction").show(100)

# Evaluate accuracy
predictions = LoRegModel.transform(testing)
evaluator = MulticlassClassificationEvaluator(labelCol="Index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Display accuracy
print("Accuracy: {:.2%}".format(accuracy))

sentiment_counts = predictions.groupBy("prediction").count().toPandas()

# Plot the bar chart
plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts["prediction"], sentiment_counts["count"], color=['green', 'red', 'orange'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(sentiment_counts["prediction"], ["Positive", "Negative", "Neutral"])
plt.show()

# Define the threshold for triggering alerts
threshold = 0.7 
# Utilize les variables d'environnement pour stocker vos informations d'identification
sender_email = "hafsa.bannass@gmail.com"
receiver_email = "hafsa.bannass@gmail.com"
app_password = "abcde12"  
# Verify if the accuracy is below the threshold 
if accuracy < threshold:   
    subject = "Sentiment Analysis Alert"
    body = f"The sentiment accuracy has fallen below the threshold. Current accuracy: {accuracy:.2%}"

    try:
        # Connect to yagmail SMTP server
        yag = yagmail.SMTP(sender_email, app_password)

        # Send email
        yag.send(to=receiver_email, subject=subject, contents=body)

        print("Alert sent!")
    except Exception as e:
        print(f"Error sending alert: {e}")
else:
    print("No alert triggered. Current accuracy:", accuracy)