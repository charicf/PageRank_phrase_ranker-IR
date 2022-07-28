import argparse
import os
import pdb
from preprocessing import *
import re
from itertools import combinations 
import numpy as np
from nltk.util import ngrams
import collections

def preprocessing(tokens):

	stopwords_rem = remove_stopwords(tokens)
	stemmed_words = get_porter_stemmer(stopwords_rem) # This method performs stemming and then stopword removing

	final_words = [word for word in stemmed_words if len(word) > 2]

	return final_words

def filter_ngrams_by_POS_tagg(n_grams):

	cleaned_ngrams = []
	#for word in vocabulary:
	for ngram in n_grams:
		cleaned_ngram = []
		for i, word in enumerate(ngram):
			valid = True
			idx = word.index("_")
			pos_tag = word[idx+1:]
			if pos_tag not in ['nn', 'nns', 'nnp', 'nnps', 'jj']:
				valid = False
				break
			cleaned_ngram.append(word[:idx])
		cleaned_ngram = get_porter_stemmer(cleaned_ngram)
		
		if valid: cleaned_ngrams.append(tuple(cleaned_ngram))
	 
	return cleaned_ngrams


def filter_by_POS_tagg(vocabulary):

	cleaned_voc = []
	for i, word in enumerate(vocabulary):
		idx = word.index("_")
		pos_tag = word[idx+1:]
		if pos_tag in ['nn', 'nns', 'nnp', 'nnps', 'jj']:
			
			cleaned_voc.append(word[:idx])
		else:
			cleaned_voc.append("/")

	return cleaned_voc

# Returns the adjacency matrix with the weights from the edges
def get_adjaceny_matrix(cleaned_doc, graph_list, w=3):

	prev_window = []

	list_len = len(graph_list)
	adjaceny_m = np.zeros((list_len, list_len))

	for idx in range(len(cleaned_doc) - (w-1)):

		window = cleaned_doc[idx: idx + w]
		comb = combinations(window, 2)
		comb_list =  list(comb)

		for i in comb_list:
			if '/' not in i:
				edge = (graph_list.index(i[0]), graph_list.index(i[1]))
				# Checks if the edge has already been counted in the previous window (avoid to count same edge between same vertices in adjacent windows)
				# w1, w2, w3, -, w1, w3, -, w1, w3: 
				# wd1 (w1,w2,w3): (w1, w3) |  wd2(w2,w3,-) | wd3(w3,-,w1): (w3,w1) | wd4(-,w1,w3):(w1,w3) | wd5(w1,w3,-): (w1,w3)
				# wd1, wd3, wd4 ara valid: weight = 3. wd5 is not valid because the same pair was counted in wd4
				if i not in prev_window: #If different pairs, then add to the weight
					adjaceny_m[edge[0]][edge[1]] += 1
					adjaceny_m[edge[1]][edge[0]] += 1
				
				# if it is the first appearance of the pair (new edge) then set it to one. If it is different than 0 the wight stays the same
				elif adjaceny_m[edge[0]][edge[1]] == 0:
					adjaceny_m[edge[0]][edge[1]] = 1
					adjaceny_m[edge[1]][edge[0]] = 1

		prev_window = comb_list

	return adjaceny_m

def run_pageRank(adj_mtx, a=0.85, steps=50):

	min_diff = 1e-5 # convergence threshold
	norm = np.sum(adj_mtx, axis=0)
	norm_adj_mtx = np.divide(adj_mtx, norm, out=adj_mtx, where=norm!=0) # this is ignore the 0 element in norm

	pr = np.divide(np.ones(adj_mtx.shape[0]), adj_mtx.shape[0], where=adj_mtx.shape[0]!=0) 
	pi = np.copy(pr)
	previous_pr = np.zeros(adj_mtx.shape[0])
		
	for epoch in range(steps):
		pr = (a*(np.dot(norm_adj_mtx, pr))) + ((1-a)*(pi))
		#pr = pr / np.nanmax(pr)

		if  np.mean(pr != previous_pr) < min_diff:
			print("PageRank converged at ", epoch, "steps, before specified number of steps")
			break
		else:
			if pr.sum() == 0 or (True in np.isnan(pr)) or (True in np.isinf(pr)): 
				print("Infinite or nan values present in Pr")
			previous_pr = pr

	return pr

def get_ngrams(tokenized_text, max_n):

	tokenized_text
	n_grams = []

	for i in range(max_n):
		n_grams.extend(list(ngrams(tokenized_text, i+1)))

	cleaned_ngrams = filter_ngrams_by_POS_tagg(n_grams)

	return cleaned_ngrams

def score_ngrams(n_grams, page_rank, vocabulary):

	scored_ngrams = {}

	for ngram in n_grams:
		score = 0
		for word in ngram:
			score +=  page_rank[vocabulary.index(word)]

		ngram = " ".join(ngram)
		scored_ngrams[ngram] = score

	return scored_ngrams


def filter_files(text_files_dir, gold_path):

	text_files_list = os.listdir(text_files_dir)
	gold_files_list = os.listdir(gold_path)
	text_files = list(set(gold_files_list).intersection(text_files_list))
	return text_files

def preprocess_text(tokenized_text):

	vocabulary = [] 
	tagged_text = filter_by_POS_tagg(tokenized_text)
	stemmed_words = get_porter_stemmer(tagged_text) 
	cleaned_text = list(filter(lambda a: a != '/', stemmed_words))
	
	# Remove repeated vertices (words)
	[vocabulary.append(x) for x in cleaned_text if x not in vocabulary] 

	return stemmed_words, vocabulary

def get_gold_docs(gold_path, text_files):

	gold_docs = []
	if gold_path[-1] != "\\": gold_path = gold_path + '\\'
	for g_filename in text_files:
		with open(gold_path+g_filename, "r") as gold_file:
			processed_file = tokenize_and_stem_lines(gold_file.readlines())
			gold_docs.append(processed_file)

	return gold_docs

def get_MMR(ranked_key_phrases, gold_key_phrases, k):

	D = len(gold_key_phrases)
		#pdb.set_trace()
	rd_sum = 0
	for doc_id, document in enumerate(ranked_key_phrases):
		for kp_id, candidate_keyphrase in enumerate(document[:k]):
			#pdb.set_trace()
			if candidate_keyphrase[0] in gold_key_phrases[doc_id]:
				rd_sum += 1/(kp_id+1) 
				break
	MRR = (1/D)*rd_sum

	return MRR

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--files_path", required = True, default = "cranfieldDocs", help="path to input documents")
ap.add_argument("-g", "--gold_path", required = True, default = "queries.txt", help="path to the query file")
ap.add_argument("-w", "--window", required = False, default = 3, help="window of w words in the original text. Default = 3")
ap.add_argument("-k", "--top_k", required = False, default = 10, help="top-k ranked n-grams or phrases. Default = 10")
ap.add_argument("-a", "--alpha", required = False, default = 0.85, help="damping factor. Default = 0.85")
ap.add_argument("-s", "--steps", required = False, default = 50, help="top-k ranked n-grams or phrases. Default = 50")

args = vars(ap.parse_args())
text_files_dir = args["files_path"]
gold_path = args["gold_path"]
window = int(args["window"])
k_max = int(args["top_k"])
a = float(args["alpha"])
s = int(args["steps"])

ranked_key_phrases = []
MMRs = []

text_files = filter_files(text_files_dir, gold_path)

if text_files_dir[-1] != "\\": text_files_dir = text_files_dir + '\\'

# Apply TextRank to find scores for keyphrases
for filename in text_files:
	vocabulary = []
	with open(text_files_dir+filename, "r") as input_file:
		#vocabulary.extend(get_postag_tokens_by_space(input_file.read()))
		tokenized_text = get_postag_tokens_by_space(input_file.read())

		cleaned_doc, vocabulary = preprocess_text(tokenized_text)

		adj_mtx = get_adjaceny_matrix(cleaned_doc, vocabulary, window)

		page_rank = run_pageRank(adj_mtx, a, s)

		n_grams = get_ngrams(tokenized_text, 3)

		scored_ngrams = score_ngrams(n_grams, page_rank, vocabulary)

		ranked_key_phrases.append(collections.Counter(scored_ngrams).most_common(k_max))

gold_key_phrases = get_gold_docs(gold_path, text_files)

# Get MMRs for top-k ranked n-grams
for k in range(k_max):

	MMRs.append(get_MMR(ranked_key_phrases, gold_key_phrases, k+1))

# Print MMRs
for k, result in enumerate(MMRs):
	print ("MRR for k {} is {}".format(k+1,result ))
		