# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions as fn 
from pyspark.ml.feature import RegexTokenizer
from hdfs import InsecureClient

#importing the beautiful soal and regular expression libraries for data cleaning.
from bs4 import BeautifulSoup
import re
import pandas as pd

# ML Method
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF

#importing an ntlk library function 'WordPunctTokenizer' which divides a string into substrings by splitting.
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

# 提高转换的效率
def _map_to_pandas(rdds):
    return [pd.DataFrame(list(rdds))]
    
def topas(df, n_partitions=None):
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand


# sc.setLogLevel('WARN')
# spark = SparkSession.builder.getOrCreate()
# # local[*] spark://master:7077
# sparkConf = SparkConf().setAppName('sentiment_analysis').set('spark.ui.showConsoleProgress', 'false').setMaster('yarn')
spark = SparkSession.builder \
    .master('yarn') \
    .config("spark.driver.memory", "8g") \
    .appName('my-cool-app') \
    .getOrCreate()

# sc = SparkContext('yarn', 'spark_file_conversion')
# sc.setLogLevel('WARN')
# 设置打印warn及以上级别的日志
spark.sparkContext.setLogLevel("WARN")
# spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('hdfs://172.24.71.22:22/test/IMDB.csv')
df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('hdfs://172.24.71.22:9000/test/IMDB1.csv')  # , delimiter="\n"

df = df.toPandas()
# df = topas(df)  
reg1 = r'@[A-Za-z0-9]+' #removing numbers and special characters
reg2 = r'<[^<]+?>'  #removing special characters. 
comb = r'|'.join((reg1, reg2)) #joining both the reg regression variables and will use it below in our function.

#created a function 'clean' for cleaning the review text using beautiful soap and reg exp as defined above.
def clean(review):
    s = BeautifulSoup(review, 'lxml')
    s_soup = s.get_text()
    strip = re.sub(comb, '', s_soup)
    letter = re.sub("[^a-zA-Z]", " ", strip)
    lower = letter.lower()
    word = tok.tokenize(lower)
    return (" ".join(word)).strip()

clean_reviews = []
for i in range(0, len(df['review'])):
    if df['review'][i] is None:
        continue
    tmp = clean(df['review'][i])
    if tmp is None or len(tmp) == 0 or tmp in ['\n', '\r\n'] or tmp.strip() == '':
        continue
    clean_reviews.append(tmp)
    
new_df = pd.DataFrame(clean_reviews,columns=['review'])
new_df['id'] = df.id
new_df['sentiment'] = df.score
new_df.dropna(axis=0, how='any', inplace=True)
# new_df = pd.concat([new_df, new_df])


pysparkDF = spark.createDataFrame(new_df) 

pysparkDF.write.mode('append').format("csv").option("encoding","utf-8").option("header",True).save("/test/imdb_clean_path")
# # 解决文件夹已存在，导致无法进行新的保存的操作
client = InsecureClient('http://172.24.71.22:9870', user='hadoop')
fnames = client.list('/test/imdb_clean_path')
# print(fnames)

#creating an sql context for reading the cleaned data in spark sql data frame.
# conf = SparkConf().setMaster("spark://master:9000").setAppName("Big_Data_Project")  #we are running on the local cluster with 4 nodes, but its scalable for multicluster enivironment as well.
# sc = SparkContext.getOrCreate(conf = conf)
sqlContext = SQLContext(spark)

imdb_clean_df = sqlContext.read.format('com.databricks.spark.csv').options(header = 'true', inferschema='true').load('hdfs://172.24.71.22:9000/test/imdb_clean_path/'+fnames[1])


pr = imdb_clean_df.select(fn.length('review').alias('Positive Review Length')).where(fn.col('sentiment') == 1)
nr = imdb_clean_df.select(fn.length('review').alias('Negative Review Length')).where(fn.col('sentiment') == 0)
#reading the file in a dataframe.
bucket_df = sqlContext.read.format('com.databricks.spark.csv').options(header = 'true', inferschema='true').load('hdfs://172.24.71.22:9000/test/sentiments.csv')
#creating a tokenizer object for spliting the input review column and creating a new output column 'split' 
token = RegexTokenizer().setGaps(False).setPattern("\\p{L}+").setInputCol("review").setOutputCol("split")  # 
#used the token transformer for applying the changes.
df_word = token.transform(imdb_clean_df)

#We will compare with the bucket words by exploding the split words into different rows using f.explode function and join the dataframes.
joined_df = df_word.select('id', fn.explode('split').alias('word')).join(bucket_df, 'word')
predict_df = joined_df.groupBy('id').agg(fn.avg('sentiment').alias('avg_sent')).withColumn('pred_sent', fn.when(fn.col('avg_sent') > 0, 1).otherwise(0))

print(imdb_clean_df.join(predict_df, 'id').select(fn.expr('float(sentiment = pred_sent)').alias('accuracy')).select(fn.avg('accuracy')).show(5))

#removing the stop words.
englishStopWords = StopWordsRemover.loadDefaultStopWords("english")
filter = StopWordsRemover().setStopWords(englishStopWords).setCaseSensitive(False).setInputCol("split").setOutputCol("filter_words")

# we will remove the words that appear in 8 docs or less.
count_vector = CountVectorizer(minTF=1, minDF=8, vocabSize=2**17).setInputCol("filter_words").setOutputCol("TF")

# we now create a pipelined estimator.
CVP = Pipeline(stages=[token, filter, count_vector]).fit(imdb_clean_df)

#applying the transformation to the dataframe.
CVP.transform(imdb_clean_df)


IDF = IDF().setInputCol('TF').setOutputCol('TF_IDF')
#creating an IDF pipeline
IDF_pip = Pipeline(stages=[CVP, IDF]).fit(imdb_clean_df)#applying the transformation.
TF_IDF = IDF_pip.transform(imdb_clean_df)
TF_IDF.select('id', 'filter_words', 'TF', "TF_IDF").show(4)

train_df, val_df, test_df = imdb_clean_df.randomSplit([0.80, 0.1, 0.1], seed=0)
print(train_df.count(), val_df.count(), test_df.count())

# Logistic Regression Model
#using the TF_IDF features from IDF pipeline and feeding it to the Logistic regression model.

LR = LogisticRegression().setLabelCol('sentiment').setFeaturesCol('TF_IDF').setMaxIter(100)
#creating a pipeline transformation for logistic regression and running on the training data set.
LR_pip = Pipeline(stages=[IDF_pip, LR]).fit(train_df)#applying the transformation on the validation data frame for predicting the sentiments.
LR_pip.transform(val_df).select('id', 'sentiment', 'prediction').show(5)
#applying the transformation on the test data frame
pred_test = LR_pip.transform(test_df).select('id', 'sentiment', 'prediction')
# pred_test.show(5)

# Calculating the Accuracy score on the testing data set for our ML model.
accuracy = pred_test.filter(pred_test.sentiment == pred_test.prediction).count() / float(test_df.count())
print(accuracy)

