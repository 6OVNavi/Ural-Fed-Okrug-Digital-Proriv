import pandas as pd
emotions={'angry': 0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6}
file=pd.read_csv('FSD.csv')
file['emotion']=file['emotion'].apply(lambda x: emotions[x])
file.to_csv('FSD.csv', index=False)