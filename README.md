### Steps to run this experiment with IMDB dataset
1. Download IMDB dataset from [imdb](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) & Extract.
2. Preprocess IMDB dataset by using following command.
```
 python preprocesImdb.py --f Path To Imdb Dataset
```
3. Now add noise to dataset to create noisy dataset
```
python keyboardTypos.py --d imdb --f Path To Preprocessed Imdb dataset
```
4. Now start experiment by using following command.
```
python run.py --d imdb --f path to noisy imdb dataset --out classification --task binary
```
