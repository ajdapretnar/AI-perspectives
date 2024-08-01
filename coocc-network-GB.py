import itertools
from collections import defaultdict
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist
from lemmagen3 import Lemmatizer
import networkx as nx


df = pd.read_csv("data/parliamentGB.csv")
df["Date"] = pd.to_datetime(df["Date"])

window = 5
lemmatizer = Lemmatizer("en")
pattern = r'\w+'
tokenizer = RegexpTokenizer(pattern)

# Tokenize and preprocess text data
stop_words = set(stopwords.words('english'))
seed_words = ["artificial", "intelligence", "AI"]

with open("data/parlamint-stopwords-GB.txt") as f:
    parlastop = set(f.read().splitlines())
    stop_words.update(parlastop)

def preprocess_text(text):
    unigrams = tokenizer.tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in unigrams]
    filtered_unigrams = [token for token in lemmatized if (token.isalnum()
                                                           and token not in stop_words)]
    return filtered_unigrams

# Tokenize and preprocess the text column
df['ProcessedText'] = df['Sentences'].apply(preprocess_text)

tokens = [token for l in df['ProcessedText'] for token in l]
fdist = FreqDist(tokens)

# filter on less than 10 occurrences
filtered_tokens = [token for token in tokens if fdist[token] > 50]

# Function to find co-occurring words in a window of text
def find_cooccurring_words(texts, window_size):
    word_mapping = defaultdict(set)
    for row_id, document in enumerate(texts):
        for i in range(len(document)):
            start = max(0, i - window_size)
            end = min(len(document), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    word_mapping[tuple(sorted([document[i], document[j]]))].add(row_id)
    return word_mapping

cooccurring_words = find_cooccurring_words(df['ProcessedText'],
                                                   window)

# Count occurrences of word tuple connections in documents
# Keep only tokens that appear in at least 2 documents
word_freq = {word: len(docs) for word, docs in cooccurring_words.items() if len(
    docs) > 2 and (word[0] in filtered_tokens and word[1] in filtered_tokens)
               and (word[0] in seed_words or word[1] in seed_words)}

candidate_words = set([w for words in word_freq for w in words])

# Each combination must also appear more than 2 times
for combo in itertools.combinations(candidate_words, 2):
    if combo not in word_freq:
        weight = len(cooccurring_words[combo])
        if weight > 2:
            word_freq[combo] = weight

# Construct the graph
G = nx.Graph()
for w1, w2 in word_freq:
    G.add_edge(f"{w1}", f"{w2}", weight=word_freq[tuple([w1, w2])])
attrs = {word: str(fdist[word]) for word in candidate_words}
df2 = pd.DataFrame({"Words": attrs.keys(), "Counts": attrs.values()})
nx.write_pajek(G, path=f"ParlamintGB.net",
               encoding='utf-8')
df2.to_csv(f"ParlamintGB.tab", sep="\t",
          index=False)
