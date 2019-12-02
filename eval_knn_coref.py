from collections import defaultdict, OrderedDict
import faiss
import json
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from tqdm import tqdm

from IPython import embed


REPS_FILE='tmp/coref_mention_reps/test.pkl'
MENTIONS_FILE='data/mentions/test.json'
TFIDF_CANDIDATES='data/tfidf_candidates/test.json'
OUTPUT_CANDIDATES='tmp/coref_knn_candidates/test.k_3.pkl'


with open(REPS_FILE, 'rb') as f:
  all_reps = pickle.load(f)

mention2entity = {}
entity2mention = defaultdict(list)
corpus_reps = {}
with open(MENTIONS_FILE, 'r') as f:
  for line in f:
    m = json.loads(line)

    uid = m['mention_id']
    mention2entity[uid] = m['label_document_id']
    entity2mention[m['label_document_id']].append(uid)

    corpus = m['corpus']
    if corpus not in corpus_reps.keys():
      corpus_reps[corpus] = {}
    corpus_reps[corpus][uid] = all_reps[uid]

tfidf_candidates = {}
with open(TFIDF_CANDIDATES, 'r') as f:
  for line in f:
    cand_dict = json.loads(line)
    tfidf_candidates[cand_dict['mention_id']] = cand_dict['tfidf_candidates']


coref_candidates = {}
for corpus, reps in corpus_reps.items():
  print('Computing coref candidates for corpus {}...'.format(corpus))
  uids = list(reps.keys())
  X = np.vstack([x for _, x in reps.items()])

  # build the kNN index
  index = faiss.IndexFlatL2(X.shape[1])
  index.add(X)

  # query the index
  print('Querying the index...')
  k = 3
  D, I = index.search(X, k)
  print('Done.')

  # convert indices back to uids
  for i in range(X.shape[0]):
    coref_candidates[uids[i]] = [uids[j] for j in I[i]] 

total = float(len(mention2entity.keys()))

# compute hits
normal_hits = 0.0
for m1, e in mention2entity.items():
  if e in tfidf_candidates[m1]:
    normal_hits += 1

gt_hits = 0.0
for m1, e in mention2entity.items():
  for m2 in entity2mention[e]:
    if e in tfidf_candidates[m2]:
      gt_hits += 1
      break

coref_hits = 0.0
for m1, e in mention2entity.items():
  for m2 in [m for m in coref_candidates[m1]]:
    if e in tfidf_candidates[m2] or e in tfidf_candidates[m1]:
      coref_hits += 1
      break

print('Normal Hits: {}, ({}/{})'.format(normal_hits/total, normal_hits, total))
print('Coref Hits: {}, ({}/{})'.format(coref_hits/total, coref_hits, total))
print('Ground Truth Coref Hits: {}, ({}/{})'.format(gt_hits/total, gt_hits, total))

print('\nDumping candidates...')
with open(OUTPUT_CANDIDATES, 'wb') as f:
  pickle.dump(coref_candidates, f, pickle.HIGHEST_PROTOCOL)
