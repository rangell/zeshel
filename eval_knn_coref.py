from collections import defaultdict, OrderedDict
import faiss
import json
import numpy as np
import pickle
from tqdm import tqdm

from IPython import embed


REPS_FILE='tmp/cereal/train.pkl'
MENTIONS_FILE='data/mentions/train.json'
TFIDF_CANDIDATES='data/tfidf_candidates/train.json'


with open(REPS_FILE, 'rb') as f:
  reps = pickle.load(f)
  reps = OrderedDict(reps)

mention2entity = {}
with open(MENTIONS_FILE, 'r') as f:
  for line in f:
    m = json.loads(line)
    mention2entity[m['mention_id']] = m['label_document_id']

tfidf_candidates = {}
with open(TFIDF_CANDIDATES, 'r') as f:
  for line in f:
    cand_dict = json.loads(line)
    tfidf_candidates[cand_dict['mention_id']] = cand_dict['tfidf_candidates']


uids = list(reps.keys())
X = np.vstack([x for _, x in reps.items()])

# build the kNN index
index = faiss.IndexFlatL2(X.shape[1])
index.add(X)

# query the index
print('Querying the index...')
k = 25
D, I = index.search(X, k)
print('Done.')

embed()
exit()

# convert indices back to uids
coref_candidates = {uids[i] : [uids[j] for j in I[i]] for i in range(X.shape[0])}

# compute hits
coref_hits = 0
for m1, e in mention2entity.items():
  for m2 in [m for m in coref_candidates[m1]]:
    if e in tfidf_candidates[m2] or e in tfidf_candidates[m1]:
      coref_hits += 1
      break


embed()
exit()


