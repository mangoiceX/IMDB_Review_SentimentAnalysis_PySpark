import json
import pandas as pd

with open('archive/part-01.json', 'r') as f:
    part1 = json.load(f)

id = []
review = []
score = []

for data in part1:
    print(data)
    exit(0)
    id.append(data['review_id'])
    review.append(data['review_detail'])
    if data['rating'] in ['6', '7', '8', '9', '10']:
        score.append('1')
    else:
        score.append('0')


dataset = list(zip(id, review, score))
df = pd.DataFrame(data=dataset, columns=['id', 'review', 'score'])
df.to_csv('IMDB1.csv', index=False, header=True)


