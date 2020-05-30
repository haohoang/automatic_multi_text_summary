import fasttext
import json
import utils
from Summary import Summary

with open("data.json", 'r') as f:
    data = json.load(f)

model = fasttext.load_model("wiki.en.bin")

# Kmeans -> Relative position
for topic in data.keys():
    tom_tat = Summary(data=data[topic], model=model)
    tom_tat.summary("Summay/Summary_1/" + topic + ".txt")

# Kmeans -> Absolute position
for topic in data.keys():
    tom_tat = Summary(data=data[topic], model=model, rel_position=False)
    tom_tat.summary("Summay/Summary_2/" + topic + ".txt")

# Kmeans -> MMR -> Absolute position
for topic in data.keys():
    tom_tat = Summary(data=data[topic], model=model, rel_position=False, mmr=True)
    tom_tat.summary("Summay/Summary_3/" + topic + ".txt")

#Kmeans -> LSA -> MMR -> Absolute position
for topic in data.keys():
    tom_tat = Summary(data=data[topic], model=model, rel_position=False, lsa=True, mmr=True)
    tom_tat.summary("Summay/Summary_1/" + topic + ".txt")

