import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import csv
import sys
class keyboardTypos:

    def __init__(self):
        keyboard_cartesian = {'q': {'x': 0, 'y': 1}, 'w': {'x': 1, 'y': 1}, 'e': {'x': 2, 'y': 1},
                              'r': {'x': 3, 'y': 1}, 't': {'x': 4, 'y': 1}, 'y': {'x': 5, 'y': 1},
                              'u': {'x': 6, 'y': 1}, 'i': {'x': 7, 'y': 1}, 'o': {'x': 8, 'y': 1},
                              'p': {'x': 9, 'y': 1}, 'a': {'x': 0, 'y': 2}, 'z': {'x': 0, 'y': 3},
                               's': {'x': 1, 'y': 2}, 'x': {'x': 1, 'y': 3}, 'd': {'x': 2, 'y': 2},
                               'c': {'x': 2, 'y': 3}, 'f': {'x': 3, 'y': 2}, 'b': {'x': 4, 'y': 3},
                               'm': {'x': 6, 'y': 3}, 'g': {'x': 4, 'y': 2}, 'h': {'x': 5, 'y': 2},
                               'j': {'x': 6, 'y': 2}, 'k': {'x': 7, 'y': 2},'l': {'x': 8, 'y': 2},
                               'v': {'x': 3, 'y': 3}, 'n': {'x': 5, 'y': 3}, '1': {'x': 0, 'y': 0}, '2': {'x': 1,'y': 0},
                               '3': {'x': 2, 'y': 0}, '4': {'x': 3, 'y': 0}, '5': {'x': 4, 'y': 0}, '6': {'x': 5, 'y': 0},
                               '7':{'x': 6, 'y': 0}, '8': {'x': 7, 'y': 0}, '9': {'x': 8, 'y': 0}, '0': {'x': 10, 'y': 0}}
        keys = self.compute_distance(keyboard_cartesian)
        self.probs = self.compute_prob(keys)

    def compute_distance(self,keyboard_cartesian):
        keys = {k: {} for k in keyboard_cartesian.keys()}
        for i in keyboard_cartesian.keys():
            for j in keyboard_cartesian.keys():
                dist = self.euclidean_distance(keyboard_cartesian,i, j)
                keys[i][j] = dist
                keys[j][i] = dist
        return keys

    def compute_prob(self,keys):
        keys = {k: {kp: vp for kp, vp in v.items() if 0.0 < vp <= 2.0} for k, v in keys.items()}
        probs = {k: {kp: ((1 / vp**2) / (sum([1 / x**2 for _, x in v.items()]))) for kp, vp in v.items()} for k, v in keys.items()}
        probs = {k: ([kp for kp in v.keys()], [vp for _, vp in v.items()]) for k, v in probs.items()}
        return probs

    def euclidean_distance(self,keyboard_cartesian,a,b):
        X = (keyboard_cartesian[a]['x'] - keyboard_cartesian[b]['x'])**2
        Y = (keyboard_cartesian[a]['y'] - keyboard_cartesian[b]['y'])**2
        return math.sqrt(X+Y)

    def alt_char(self,char):
        try:
            return np.random.choice(self.probs[char][0], 1, p=self.probs[char][1])[0]
        except:
            return char

    def pick_idx(self,text_len, prob):
        if text_len < 5:
            return []
        sample_space = list(range(text_len))
        return np.random.choice(sample_space, int(text_len * prob) if text_len >= 10 else 1)

    def change_char(self,string, idx, char):
        return string[: idx] + char + string[idx + 1: ]

    # adding typo to imdb dataset
    def add_typos(self,texts: list,prob):
        # input : a list of strings
        # output : a list of string after adding typos <same length>
        ret_texts= []
        print("Adding keyboard errors")
        for text in tqdm(texts):
            idxs = self.pick_idx(len(text), prob=prob)
            for i, idx in enumerate(idxs):
                try:
                    while text[idx] == ' ' or not text[idx].isalnum():
                        idx += 1
                        idx %= len(text)
                except Exception:
                    print(idx, text)
                text = self.change_char(text, idx, char=self.alt_char(text[idx].lower()))
            ret_texts.append(text)
        return ret_texts
    
    def generate_data_imdb(self,dataDir):
        df_train,df_dev = self.read_tsv(dataDir)
        prob = [0,0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.20,0.225]
        print("Generating data")
        for p in tqdm(prob):
            ret_text = self.add_typos(list(df_train['review']),p)
            temp_df = pd.DataFrame({"label":list(df_train['label']),"review":ret_text})
            ret_text_dev = self.add_typos(list(df_dev['review']),p)
            temp_df_dev = pd.DataFrame({"label":list(df_dev['label']),"review":ret_text_dev})
            dir_name = dataDir+'/imdb_'+str(p)
            if os.path.exists(dir_name):
                temp_df.to_csv(dir_name+"/train.tsv",sep="\t",index=False)
                temp_df_dev.to_csv(dir_name+"/dev.tsv",sep="\t",index=False)
            else:
                os.mkdir(dir_name)
                temp_df.to_csv(dir_name+"/train.tsv",sep="\t",index=False)
                temp_df_dev.to_csv(dir_name+"/dev.tsv",sep="\t",index=False)
            del temp_df
            del temp_df_dev
    def generate_data_sst2(self,dataDir):
        df_train,df_dev= self.read_tsv(dataDir,False)
        prob = [0,0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.20,0.225]
        print("Generating data")
        for p in tqdm(prob):
            ret_text = self.add_typos(list(df_train['sentence']),p)
            temp_df = pd.DataFrame({"label":list(df_train['label']),"sentence":ret_text})
            ret_text_dev = self.add_typos(list(df_dev['sentence']),p)
            temp_df_dev = pd.DataFrame({"label":list(df_dev['label']),"sentence":ret_text_dev})

            dir_name = dataDir+'/sst2_'+str(p)
            if os.path.exists(dir_name):
                temp_df.to_csv(dir_name+"/train.tsv",sep="\t",index=False)
                temp_df_dev.to_csv(dir_name+"/dev.tsv",sep="\t",index=False)
            else:
                os.mkdir(dir_name)
                temp_df.to_csv(dir_name+"/train.tsv",sep="\t",index=False)
                temp_df_dev.to_csv(dir_name+"/dev.tsv",sep="\t",index=False)
            del temp_df
            del temp_df_dev
    def generate_data_sts(self,dataDir):
        train_lines= self._read_tsv_sts(dataDir+"/train.tsv")
        train_sent1 = [line[7] for line in train_lines]
        train_sent2 = [line[8] for line in train_lines]
        train_label = [line[-1] for line in train_lines]
        dev_lines= self._read_tsv_sts(dataDir+"/dev.tsv")
        dev_sent1 = [line[7] for line in dev_lines]
        dev_sent2 = [line[8] for line in dev_lines]
        dev_label = [line[-1] for line in dev_lines]
        prob = [0,0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.20,0.225]
        print("Generating data")
        for p in tqdm(prob):
            ret_text_sent1 = self.add_typos(train_sent1,p)
            ret_text_sent2 = self.add_typos(train_sent2,p)
            assert len(ret_text_sent1) == len(train_sent1)
            df_train = pd.DataFrame({"sentence1":ret_text_sent1,"sentence2": ret_text_sent2,"label":train_label})

            ret_text_sent1 = self.add_typos(dev_sent1,p)
            ret_text_sent2 = self.add_typos(dev_sent2,p)
            assert len(ret_text_sent2) == len(dev_sent2)
            df_dev = pd.DataFrame({"sentence1":ret_text_sent1,"sentence2": ret_text_sent2,"label":dev_label})

            dir_name = dataDir+'/sts_'+str(p)
            if os.path.exists(dir_name):
                df_train.to_csv(dir_name+"/train.tsv",sep="\t",index=False)
                df_dev.to_csv(dir_name+"/dev.tsv",sep="\t",index=False)
            else:
                os.mkdir(dir_name)
                df_train.to_csv(dir_name+"/train.tsv",sep="\t",index=False)
                df_dev.to_csv(dir_name+"/dev.tsv",sep="\t",index=False)


    def _read_tsv_sts(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            skip_first = True
            for line in reader:
                if skip_first:
                    skip_first = False
                    continue
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def read_tsv(self,dataDir,test=False):
        df_train = pd.read_csv(dataDir+"/train.tsv",sep="\t",error_bad_lines=False)
        df_dev = pd.read_csv(dataDir+"/dev.tsv",sep="\t",error_bad_lines=False)
        if test:
            df_test = pd.read_csv(dataDir+"/test.tsv",sep="\t")
            return df_train,df_dev,df_test
        else:
            return df_train,df_dev

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",dest='dataset',help="imdb or sst2 or sts",metavar='str')
    parser.add_argument("--f",dest='path',help="processed imdb data dir",metavar='path')
    result = parser.parse_args()
    obj = keyboardTypos()
    if result.dataset =="imdb":
        obj.generate_data_imdb(result.path)
    elif result.dataset == "sst2":
        obj.generate_data_sst2(result.path)
    elif result.dataset == "sts":
        obj.generate_data_sts(result.path)



