
#By: Charic Daniel Farinango Cuervo 

# SMART stopwords: http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
from nltk.tokenize import word_tokenize,  RegexpTokenizer
from nltk.corpus import stopwords 
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer 
import argparse
import os
import pdb


def get_postag_tokens_by_space(text):

	input_tokens = []
	tokenizer = RegexpTokenizer(r'[a-zA-Z0-9_-]+[_]+[a-zA-Z]+')

	input_tokens.extend(tokenizer.tokenize(text.lower()))

	return input_tokens

def get_tokens_by_space(text):

	input_tokens = []
	tokenizer = RegexpTokenizer(r'[A-Za-z-]+')

	input_tokens.extend(tokenizer.tokenize(text.lower()))

	return input_tokens

def tokenize_and_stem_lines(lines):
	
	processed_doc = []
	for text in lines:
		processed_line = get_tokens_by_space(text)
		stemmed_text = get_porter_stemmer(processed_line) 
		keyphrase = " ".join(stemmed_text)
		processed_doc.append(keyphrase)

	return processed_doc

def tokenize_white_space(text_files_dir):

	#text_files_dir = "test/"
	input_tokens = []
	tokenizer = RegexpTokenizer(r'[A-Za-z]+')

	if os.path.isfile(text_files_dir):

		input_file = open(text_files_dir, "r")
		input_tokens.extend(tokenizer.tokenize(input_file.read().lower()))

	else:
	
		if text_files_dir[-1] != "\\": text_files_dir = text_files_dir + '\\'
		for filename in os.listdir(text_files_dir):
			#if filename.endswith(".txt"):
			with open(text_files_dir+filename, "r") as input_file:
				input_tokens.extend(tokenizer.tokenize(input_file.read().lower()))

	return input_tokens

# def words_freq(tockens):

# 	freq_words = FreqDist(tockens)

# 	return freq_words

def find_stopwords(words):

	stop_words = set(stopwords.words('english')) 

	detected_stopwords = [] 
  
	for w in words: 
		if w in stop_words: 
			detected_stopwords.append(w) 

	return detected_stopwords

def remove_stopwords(words):

	stop_words = set(stopwords.words('english')) 



	removed_stopwords = [] 
  
	for w in words: 
		if w not in stop_words: 
			removed_stopwords.append(w) 

	return removed_stopwords

def get_porter_stemmer(words):

	porter_stemmer = PorterStemmer()
	steamm_words = []

	for x in words:
		steamm_words.append(porter_stemmer.stem(x))

	#pdb.set_trace()
	#steamm_words = [porter_stemmer.stem(word) for word in words]

	#pdb.set_trace()

	steamm_tokens = remove_stopwords(steamm_words)

	return steamm_tokens

def words_accounting_fifteen(words_freq, total_words_collection):

	fifteen = 0.15 * total_words_collection
	summ = 0
	x = 0
	sorted_words = sorted(words_freq, key=words_freq.get, reverse=True)

	for word in sorted_words:

		x += 1
		summ = summ + words_freq[word]
		if summ >= fifteen:
			return x

def find_answers(tokens):

	words_freq = FreqDist(tokens)
	total_words_collection = len(tokens)

	top_words = [x[0] for x in words_freq.most_common(20)]
	top_words_freq = words_freq.most_common(20)

	detected_stopwords = find_stopwords(top_words)

	number_words = words_accounting_fifteen(words_freq, total_words_collection)

	print("\nTotal number of words in collection: ", total_words_collection)
	print("Vocabulary size: ", len(words_freq))
	print("Top 20 words in the ranking: ", top_words_freq)
	print("Stopwords: ", detected_stopwords)
	print("Minimum number of unique words accounting for 15% of the total numberof words in the collection: ", number_words)


# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--files_path", required = True, default = "citeseer/", help="path to input documents")

# args = vars(ap.parse_args())
# text_files_dir = args["files_path"]

# tokens = tokenize_white_space(text_files_dir)

# find_answers(tokens)

# print ("\nResults with Porter stemmer and stopword eliminator: ")
# steamm_tokens = get_porter_stemmer(tokens)

# find_answers(steamm_tokens)

# print("\n By: Charic Daniel Farinango Cuervo ")

#print("Last 20 words in the ranking: ", words_freq.most_common()[-20:])





