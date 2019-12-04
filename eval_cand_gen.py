from collections import defaultdict, OrderedDict
import faiss
import json
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from tqdm import tqdm

from IPython import embed


REPS_FILE='tmp/cand_gen_reps/train.pkl'
MENTIONS_FILE='data/mentions/train.json'
TFIDF_CANDIDATES='data/tfidf_candidates/train.json'

with open(REPS_FILE, 'rb') as f:
  all_reps = pickle.load(f)

mention_reps = all_reps['mention_reps']
entity_reps = all_reps['entity_reps']

mention2entity = {}
with open(MENTIONS_FILE, 'r') as f:
  for line in f:
    m = json.loads(line)
    uid = m['mention_id']
    mention2entity[uid] = m['label_document_id']

tfidf_candidates = {}
with open(TFIDF_CANDIDATES, 'r') as f:
  for line in f:
    cand_dict = json.loads(line)
    tfidf_candidates[cand_dict['mention_id']] = cand_dict['tfidf_candidates']

mention_uids = list(mention_reps.keys())
entity_uids = list(entity_reps.keys())

X = np.vstack([x for _, x in entity_reps.items()])
Q = np.vstack([x for _, x in mention_reps.items()])

entity_index = faiss.IndexFlatL2(X.shape[1])
entity_index.add(X)

# query the index
print('Querying the index...')
k = 64
D, I = entity_index.search(Q, k)
print('Done.')

knn_candidates = {}
for i in range(Q.shape[0]):
  knn_candidates[mention_uids[i]] = [entity_uids[j] for j in I[i]]

total = float(len(mention2entity.keys()))

# compute hits
normal_hits = 0.0
for m1, e in mention2entity.items():
  if e in tfidf_candidates[m1]:
    normal_hits += 1

knn_hits = 0.0
for m1, e in mention2entity.items():
  if e in knn_candidates[m1]:
    knn_hits += 1

print('BM25 Hits: {}, ({}/{})'.format(normal_hits/total, normal_hits, total))
print('kNN Hits: {}, ({}/{})'.format(knn_hits/total, knn_hits, total))
