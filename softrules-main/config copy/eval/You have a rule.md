You have a rule
Apply rule expander module to expand it into multiple rules
    - using word similarity (glove)
embed each resulting rule using glove (average of the words in the constraint)
embed sentence (average of glove for each word)
cosine similarity between them

for prediction
if maximum cosine similarity > threshold, predict that, otherwise 'no_relation'

