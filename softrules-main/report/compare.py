import pandas as pd 
import numpy as np

#df.to_csv('x1_with95')
# file1 = open("x1_with_95","a")
# file2 = open("x1_without_95","a")


# A=set(pd.read_csv("c1.csv", index_col=False, header=None)[0]) #reads the csv, takes only the first column and creates a set out of it.
#     B=set(pd.read_csv("c2.csv", index_col=False, header=None)[0]) #same here
#     print(A-B) #set A - set B gives back everything thats only in A.
#     print(B-A) # same here, other way around.


df1=pd.read_excel('diff1.xlsx')
df2=pd.read_excel('diff2.xlsx')
df1.equals(df2)
comparison_values = df1.values == df2.values
print (comparison_values)

rows,cols=np.where(comparison_values==False)
for item in zip(rows,cols):
    df1.iloc[item[0], item[1]] = '{} --> {}'.format(df1.iloc[item[0], item[1]],df2.iloc[item[0], item[1]])

#df1 = pd.read_csv('x1_with_95.txt')
#df2 = pd.read_csv('x1_without_95.txt')
#print(df1-df2)
#df1.apply(tuple,1)

#result = df1[~df1.apply(tuple,1).isin(df2.apply(tuple,1))]
#print(result)