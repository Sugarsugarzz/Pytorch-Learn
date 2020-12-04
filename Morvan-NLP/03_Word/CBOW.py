"""
Continuous Bag of Words (CBOW)
    挑一个要预测的词，来学习这个词前后文中词语的意思。
    如给定一句话，将这句话拆成输入和输出，用前后文的词向量来预测句中的某个词。
    滑动窗口，例如窗口大小为5，给定前两个和后两个，预测中间的字

词向量的应用
"""
import tensorflow as tf
from tensorflow import keras


corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]

class CBOW(keras.Model):
    def __init__(self, v_dim, emb_dim):
        super().__init__()
        self.embeddings = keras.layers.Embedding(
            input_dim=v_dim, output_dim=emb_dim,  # [n_vocab, emb_dim]
            embeddings_initializer=keras.initializers.RandomNormal(0., 0.1)
        )

    def call(self, x, training=None, mask=None):
        # x.shape  [n, skip_window*2]
        o = self.embeddings(x)  # [n, skip_window*2, emb_dim]
        o = tf.reduce_mean(o, axis=1)  # [n, emb_dim]
        return o



