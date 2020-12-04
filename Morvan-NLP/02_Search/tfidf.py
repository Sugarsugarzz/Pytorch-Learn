"""
TF-IDF 实现
    计算好所有文档向量，每次来一个搜索问题，将这个问题转换成同样的向量，找到与问题向量最相似的文档。
"""
import numpy as np
from collections import Counter
import itertools

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

docs_words = [d.replace(',', '').split(" ") for d in docs]
vocab = set(itertools.chain(*docs_words))
v2i = {v: i for i, v in enumerate(vocab)}
i2v = {i: v for v, i in v2i.items()}

# idf
idf_methods = {
    "log": lambda x: 1 + np.log(len(docs) / (x+1)),
    "prob": lambda x: np.maximum(0, np.log((len(docs) - x) / (x+1))),
    "len_norm": lambda x: x / (np.sum(np.square((x)) + 1))
}

def get_idf(method="log"):
    # low idf for a word appears in more docs, mean less important
    df = np.zeros((len(i2v), 1))
    # 统计每个词在全局文档中出现的数量
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i, 0] = d_count
    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    # log方法： IDF = 1 + log(文档数 / word出现数 + 1)
    return idf_fn(df)  # [n_vocab, 1]

# tf
tf_methods = {
    "log": lambda x: np.log(1+x),
    "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
    "boolean": lambda x: np.minimum(x, 1)
}

def get_tf(method="log"):
    # how frequent a word appears in a doc
    _tf = np.zeros((len(vocab), len(docs)), dtype=np.float64)  # [n_vocab, n_docs]
    # 统计每篇文档中各个词频
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        for v in counter.keys():
            _tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]  # 归一化

    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    return weighted_tf(_tf)  # [n_vocab, n_doc]

# 将搜索问题向量化，把搜索向量在文档向量上进行举例计算，算出哪些文档贴近搜索文档
def cosine_similarity(q, _tf_idf):
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity

def docs_score(q, len_norm=False):
    q_words = q.replace(",", "").split(" ")

    # add unknown words
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i) - 1] = v
            unknown_v += 1

    if unknown_v > 0:
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype=np.float)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=np.float)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf

    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float)
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]

    q_vec = q_tf * _idf

    q_scores = cosine_similarity(q_vec, _tf_idf)
    if len_norm:
        len_docs = [len(d) for d in docs_words]
        q_scores = q_scores / np.array(len_docs)
    return q_scores

def get_keywords(n=2):
    for c in range(3):
        col = tf_idf[:, c]
        idx = np.argsort(col)[-n:]
        print("doc{}, top{} keywords {}".format(c, n, [i2v[i] for i in idx]))

# TF-IDF
# 一个词语和文章的矩阵，代表用词语向量表示的文章
tf = get_tf()
idf = get_idf()
tf_idf = tf * idf
print("tf shape(vecb in each docs): ", tf.shape)
print("\ntf samples:\n", tf[:2])
print("\nidf shape(vecb in all docs): ", idf.shape)
print("\nidf samples:\n", idf[:2])
print("\ntf_idf shape: ", tf_idf.shape)
print("\ntf_idf sample:\n", tf_idf[:2])


# test
get_keywords()
q = "I get a coffee cup"
scores = docs_score(q)
d_ids = scores.argsort()[-3:][::-1]
print("\ntop 3 docs for '{}': \n{}".format(q, [docs[i] for i in d_ids]))