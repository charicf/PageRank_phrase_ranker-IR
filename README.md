# PageRank phrase ranker-IR

Script that uses PageRank to detect top ranked key-phrases in text documents

## Description

The script receives three main arguments that define the path to the documents, the path to the ground truth key-phrases, and the window size. An example of the format of folders and files is provided. The collection consist of titles and abstracts of research articles, where each document has to be POS tagged. The first step is to find which files can be analyzed. Those files will correspond to files that exist in both paths (documents and ground truth). Then, for each resulting file, the following steps are applied.

First, the text is tokenized from the input text. After that, the pre-process step selects the relevant words (nouns and adjectives corresponding  to {NN,  NNS,  NNP,  NNPS,  JJ}). Then, words are stemmed and the stop words are removed. Finally, a vocabulary is constructed with all the relevant terms.
Second, the vocabulary and the original text are used to compute the adjacency matrix. The adjacency matrix represents the graph formed by the connections between w (window-size) words. This matrix contains the weights between the terms (number of repetitions). 
Third, the adjacency matrix is used to implement the PageRank algorithm. The goal is to find the scores for each word based on the weights registered in the matrix. This is done iteratively until convergence or until a maximum number of steps is achieved.
Four, the n-grams are obtained and the score is computed. The score of each n-gram is equal to the sum of scores of each word. 
Finally, the n-grams are sorted by score and the first k elements are taken from it. 

After the key-phrases for each file have been obtained and ranked, the MMR score is computed. This is done by taking the ground truth key-phrases. 

## Packages and versions

Python 3.7.5

The following packages are required

```bash
preprocessing_assg1.py. Must be in the same folder than the main script 
re, argparse, os, numpy, collections, itertools, nltk==3.4.5
```
If packages are not installed: 

```
pip install <package_name>
```
## Arguments

```
argument("-p", "--files_path", required = True, default = "cranfieldDocs", help="path to input documents")
argument("-g", "--gold_path", required = True, default = "queries.txt", help="path to the query file")
argument("-w", "--window", required = False, default = 3, help="window of w words in the original text. Default = 3")
argument("-k", "--top_k", required = False, default = 10, help="top-k ranked n-grams or phrases. Default = 10")
argument("-a", "--alpha", required = False, default = 0.85, help="damping factor. Default = 0.85")
argument("-s", "--steps", required = False, default = 50, help="top-k ranked n-grams or phrases. Default = 50")

```

## Run the script

```python
python text_rank.py -p <folder_path> -g <folder_path> -w <window_size> -k <top_k_ranking> -a <alpha> -s <steps>
```

##### Example:

```python
python text_rank.py -p www\abstracts -g www\gold -w 6 -k 10 -a 0.85 -s 50
```

## Results

```
For w 6, MRR values for each k ranging from 1 to 10 for the WWW collection

MRR for k 1 is 0.05488721804511278
MRR for k 2 is 0.08383458646616541
MRR for k 3 is 0.10714285714285708
MRR for k 4 is 0.12706766917293238
MRR for k 5 is 0.13804511278195494
MRR for k 6 is 0.1468170426065163
MRR for k 7 is 0.1535839598997493
MRR for k 8 is 0.15828320802005003
MRR for k 9 is 0.16229323308270674
MRR for k 10 is 0.16635338345864642

```

## Author
#### Charic Daniel Farinango Cuervo

## Contributing
Pull requests are welcome.

## License
[MIT](https://choosealicense.com/licenses/mit/)