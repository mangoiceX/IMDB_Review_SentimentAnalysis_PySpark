# IMDB_Review_SentimentAnalysis_PySpark
Project for IMDB review sentiment analysis using PySpark and Hadoop cluster with 3 nodes.


Input the following command to run the code:
bin/spark-submit --master spark://yourIP /usr/hadoop-3.3.1/sparkdata/sa_final.py sentiment_analysis.py


A statement for sentiment_analysis_for_version.py:
There is a for-circulation in the code, which is not distributed. We recommend to use sentiment_analysis.py instead.
We use sentiment_analysis.py as our final version.
