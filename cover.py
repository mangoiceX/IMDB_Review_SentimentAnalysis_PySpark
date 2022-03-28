import re
import pandas as pd

file = 'archive/IMDB1.csv'
df = pd.read_csv(file)

review = df['review']
replace = []

for item in review:
    review.append(re.sub(',', '', item))

df.drop('review',axis=1, inplace=True)

df['review'] = replace
print(df.shape[0])


df.head(220000).to_csv('IMDB.csv', index=False, header=True)