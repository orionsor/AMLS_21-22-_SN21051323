import pandas as pd

FileList = pd.read_csv("./dataset/label.csv")
    FileList['binary_label'] = ''
    for index, row in FileList.iterrows():
        if row['label'][0] == 'n':
            FileList['binary_label'][index] = 'negative'
        else:
            FileList['binary_label'][index] = 'positive'

FileList.to_csv('./dataset/binary_label.csv', index=False, header=True)