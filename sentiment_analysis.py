# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions as fn 
from pyspark.ml.feature import RegexTokenizer
from hdfs import InsecureClient
# import findspark
# findspark.init()

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

# def _map_to_pandas(rdds):
#     return [pd.DataFrame(list(rdds))]

# 获取spark的上下文
# sc = SparkContext('local', 'spark_file_conversion')
# sc.setLogLevel('WARN')
# spark = SparkSession.builder.getOrCreate()
# 调整spark.driver.memory 大小设置根据实际环境调整local[*]
spark = SparkSession.builder \
    .master('yarn') \
    .config("spark.driver.memory", "8g") \
    .appName('my-cool-app') \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
# 用于Pandas和PySpark的DataFrame进行转换
# spark.conf.set("spark.sql.execution.arrow.enabled", "true")  # pyspark 2.0的写法，已废弃
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# spark.conf.set('spark.yarn.dist.files','file:/usr/spark-3.2.1/python/lib/pyspark.zip,file:/usr/spark-3.2.1/python/lib/py4j-0.10.9.3-src.zip')
# spark.conf.setExecutorEnv('PYTHONPATH','pyspark.zip:py4j-0.10.9.3-src.zip')

# 读取本地或HDFS上的文件【.load('hdfs://192.168.3.9:8020/input/movies.csv')】
# 因为DataFrame默认以逗号进行分割，当某一列的值包含逗号时，会造成读取错误，所以需事先去除csv文件的逗号

df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('hdfs://172.24.71.22:9000/test/IMDB1.csv')  # , delimiter="\n"# 将spark.dataFrame转为pandas.DataFrame
# df = df.toPandas()
# df = topas(df)  # 导致下面的score列名属性消失
# print(df.score)
reg1 = r'@[A-Za-z0-9]+' #removing numbers and special characters
reg2 = r'<[^<]+?>'  #removing special characters. 
comb = r'|'.join((reg1, reg2)) #joining both the reg regression variables and will use it below in our function.

# #created a function 'clean' for cleaning the review text using beautiful soap and reg exp as defined above.
# def clean(review):
#     s = BeautifulSoup(review, 'lxml')
#     s_soup = s.get_text()
#     strip = re.sub(comb, '', s_soup)
#     letter = re.sub("[^a-zA-Z]", " ", strip)
#     lower = letter.lower()
#     word = tok.tokenize(lower)
#     return (" ".join(word)).strip()


# # 这个for循环怎么弄成分布式？
# clean_reviews = []
# for i in range(0, len(df['review'])):
#     if df['review'][i] is None:
#         continue
#     tmp = clean(df['review'][i])
#     if tmp is None:
#         continue
#     if len(tmp) == 0:
#         continue
#     if tmp in ['\n', '\r\n']:  # 清楚空行
#         continue
#     if tmp.strip() == '':
#         continue
#     clean_reviews.append(tmp)
    
# new_df = pd.DataFrame(clean_reviews,columns=['review'])
# new_df['id'] = df.id
# new_df['sentiment'] = df.score
# new_df.dropna(axis=0, how='any', inplace=True)
# print(df.score)
# print(new_df.head(5))
# new_df.to_csv('imdb_clean.csv',encoding='utf-8')
# csv = 'imdb_clean.csv'

# 下面是使用pyspark的写法
from pyspark.sql import Row
def clean(item):
    review = item['review']
    s = BeautifulSoup(review, 'lxml')
    s_soup = s.get_text()
    strip = re.sub(comb, '', s_soup)
    letter = re.sub("[^a-zA-Z]", " ", strip)
    lower = letter.lower()
    word = tok.tokenize(lower)
    res = (" ".join(word)).strip()
    # 由于item是PySpark的Row对象，与tuple类似，无法进行修改，所以创建一个新的Row对象
    ans = Row(id=item['id'],sentiment=item['score'], review=res) 
    # item['review'] = res  # 无法进行修改
    return ans

# new_df = df.select("review").collect()  # 用SQL语句进行选择，返回是list
# print(type(new_df))
# 并行化，调用处理函数
df.dropna(how='any')
new_df = spark.sparkContext.parallelize(df.collect()).map(clean)  # 类型是RDD.PipelineRDD
print(type(new_df))

# 将RDD数据转化为DataFrame，需要定义列和类型
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType
schema = StructType([
    StructField('id', StringType(), True),
    StructField('sentiment', IntegerType(), True),
    StructField('review', StringType(), True)
])
# 转换为csv
# pysparkDF = new_df.toDF()  # 缺乏类型
pysparkDF = spark.createDataFrame(new_df,schema)
pysparkDF.dropna(how='any')
print(type(pysparkDF))

# pysparkDF = spark.createDataFrame(new_df) 
# 下面两行代码是解决文件目录已存在的问题

pysparkDF.write.mode('overwrite').format("csv").option("encoding","utf-8").option("header",True).save("/test/imdb_clean_path")
client = InsecureClient('http://172.24.71.22:9870', user='hadoop')
fnames = client.list('/test/imdb_clean_path')
# print(fnames)
#creating an sql context for reading the cleaned data in spark sql data frame.
# conf = SparkConf().setMaster("spark://master:9000").setAppName("Big_Data_Project")  #we are running on the local cluster with 4 nodes, but its scalable for multicluster enivironment as well.
# sc = SparkContext.getOrCreate(conf = conf)
# sqlContext = SQLContext(sc)
sqlContext = SQLContext(spark)

imdb_clean_df = sqlContext.read.format('com.databricks.spark.csv').options(header = 'true', inferschema='true').load('hdfs://172.24.71.22:9000/test/imdb_clean_path/'+fnames[1])
# pysparkDF.write.mode('append').format("csv").repartition(1).option("encoding","utf-8").option("header",True).save("/test/imdb_clean.csv") 
# pysparkDF.write.format("csv").options(header='true', inferschema='true').save('hdfs://172.24.71.22:9000/test/imdb_clean_test26')
# pysparkDF.coalesce(1).write.csv('hdfs://172.24.71.22:9000/test/imdb_clean_test4')

# imdb_clean_df = sqlContext.read.format('com.databricks.spark.csv').options(header = 'true', inferschema='true').load('hdfs://172.24.71.22:9000/test/imdb_clean_test26')
# print(type(imdb_clean_df))
# print(imdb_clean_df.show(5))
# pr = imdb_clean_df.select(fn.length('review').alias('Positive Review Length')).where(fn.col('sentiment') == 1)
# nr = imdb_clean_df.select(fn.length('review').alias('Negative Review Length')).where(fn.col('sentiment') == 0)
#reading the file in a dataframe.
bucket_df = sqlContext.read.format('com.databricks.spark.csv').options(header = 'true', inferschema='true').load('hdfs://172.24.71.22:9000/test/sentiments.csv')
#creating a tokenizer object for spliting the input review column and creating a new output column 'split' 
token = RegexTokenizer().setGaps(False).setPattern("\\p{L}+").setInputCol("review").setOutputCol("split")  # 
#used the token transformer for applying the changes.
df_word = token.transform(imdb_clean_df)
# print(df_word.show(6))
#We will compare with the bucket words by exploding the split words into different rows using f.explode function and join the dataframes.
joined_df = df_word.select('id', fn.explode('split').alias('word')).join(bucket_df, 'word')
predict_df = joined_df.groupBy('id').agg(fn.avg('sentiment').alias('avg_sent')).withColumn('pred_sent', fn.when(fn.col('avg_sent') > 0, 1).otherwise(0))

print(imdb_clean_df.join(predict_df, 'id').select(fn.expr('float(sentiment = pred_sent)').alias('accuracy')).select(fn.avg('accuracy')).show(5))

# 加载stop会造成类型不对
# stop = sc.textFile('hdfs://172.24.71.22:22/test/stop_words.txt').map(lambda line: line.split())

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

# Logistic Regression Model
#using the TF_IDF features from IDF pipeline and feeding it to the Logistic regression model.

LR = LogisticRegression().setLabelCol('sentiment').setFeaturesCol('TF_IDF').setMaxIter(100)
#creating a pipeline transformation for logistic regression and running on the training data set.
LR_pip = Pipeline(stages=[IDF_pip, LR]).fit(train_df)#applying the transformation on the validation data frame for predicting the sentiments.
LR_pip.transform(val_df).select('id', 'sentiment', 'prediction').show(5)
#applying the transformation on the test data frame
pred_test = LR_pip.transform(test_df).select('id', 'sentiment', 'prediction')

# Calculating the Accuracy score on the testing data set for our ML model.
accuracy = pred_test.filter(pred_test.sentiment == pred_test.prediction).count() / float(test_df.count())
print(accuracy)

