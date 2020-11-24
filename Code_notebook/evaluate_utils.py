import numpy as np

def predict(X, W, tgt_emb, K):
    """
    returns a list of most (K) probable vector translations of a given word X.
    X: word embedding
    W : rotation matrix
    tgt_embed : embeddings of words in target language
    K : nearest-neighbors parameter
    """
    word_emb_tgt = X.reshape(1,-1).dot(W).reshape(-1)
    res = get_nn_vec(word_emb_tgt,tgt_emb,K)
    return res

def get_nn_vec(word_emb, tgt_emb, K):
  """
  return a list of K-nearest neighbors of a word word_emb
  """
  scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
  k_best = scores.argsort()[-K:][::-1]
  return [tgt_emb[k_best[i]] for i in range(K)]

def accuracy(X,Z,W, tgt_emb, K=1):
  """
  Returns percentage of accuracy on given X,Z
    X : embeddings of words in source language
    Z : embeddings of translated word corresponding to X (Z is translation of X line by line)
    W : rotation matrix
    tgt_embed : embeddings of words in target language
    K : nearest-neighbors parameter

  """
  acc = 0
  #For every word in X
  for i in range(X.shape[0]):
    # We look for the K-probable translations vectors
    z_pred = predict(X[i], W, tgt_emb, K)
    for cand in range(K):
      # If one our candidates is equal to the true translation
      if (Z[i] == z_pred[cand]).all():
        # We increment our counter
        acc += 1
        continue
  return (acc/X.shape[0])*100


def get_word_nn(word, W, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]].reshape(1,-1).dot(W).reshape(-1)
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        print('    %.4f - %s' % (scores[idx], tgt_id2word[idx]))


