import glob, os
import argparse
import re
import pandas as pd

class processIMDB:

    def __init__(self,DataPath:str):
        # training set
        sent_pos = self.read_files(DataPath+'train/pos')
        sent_pos = self.preprocess_reviews(sent_pos)
        sent_neg = self.read_files(DataPath+'train/neg')
        sent_neg = self.preprocess_reviews(sent_neg)
        labels = [1]*len(sent_pos)
        labels.extend([0]*len(sent_neg))
        sent_pos.extend(sent_neg)
        df_train = pd.DataFrame({"label":labels,"review":sent_pos})
        # dev test
        sent_pos = self.read_files(DataPath+'test/pos')
        sent_pos = self.preprocess_reviews(sent_pos)
        sent_neg = self.read_files(DataPath+'test/neg')
        sent_neg = self.preprocess_reviews(sent_neg)
        labels = [1]*len(sent_pos)
        labels.extend([0]*len(sent_neg))
        sent_pos.extend(sent_neg)
        df_test = pd.DataFrame({"label":labels,"review":sent_pos})

        # saving dataframe as tsv
        if os.path.exists("imdb_data"):
            df_train.to_csv("imdb_data/train.tsv",sep='\t', index=False)
            df_test.to_csv("imdb_data/dev.tsv",sep='\t', index=False)
        else:
            os.mkdir("imdb_data")
            df_train.to_csv("imdb_data/train.tsv",sep='\t', index=False)
            df_test.to_csv("imdb_data/dev.tsv",sep='\t', index=False)
        print(" Process Data saved  @imdb_data/")


    def preprocess_reviews(self, reviews):
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
        return reviews

    def read_files(self, dirPath : str):
        sent = []
        for file in os.listdir(dirPath):
            if file.endswith(".txt"):
                with open(os.path.join(dirPath, file),'r') as fp:
                    sent.append(fp.readlines()[0])
        return sent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f",dest='path',help="raw imdb data dir",metavar='path')
    result = parser.parse_args()
    pr = processIMDB(result.path)
