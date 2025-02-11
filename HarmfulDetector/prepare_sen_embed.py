from collections import Counter
import numpy as np
import math
import fasttext
import fasttext.util
import json
import os
from py_vncorenlp import VnCoreNLP
from collections import defaultdict
import re
import numpy as np
from tqdm import tqdm

count_full = Counter()

def getLog(frequency, a=10 ** -3):
    """
    Tính trọng số dựa trên tần suất từ.
    """
    return a / (a + math.log(1 + frequency))

def get_embed(token):
    """
    Lấy vector embedding cho một token.
    
    Args:
    - model: Mô hình embedding (fastText, GloVe, Word2Vec, etc.)
    - token: Từ cần lấy vector.
    
    Returns:
    - vector: Vector embedding của từ hoặc None nếu không có.
    """
    try:
        return model_embed[token]
    except KeyError:
        return None

def get_vector_avg_weighted_full(sent):
    """
    Tính vector trung bình có trọng số cho một câu dựa trên embedding.
    
    Args:
    - model: Mô hình embedding (fastText, GloVe, Word2Vec, etc.)
    - sent: Câu đầu vào (list các từ hoặc chuỗi).
    
    Returns:
    - doc_vector: Vector đại diện cho câu.
    """
    # Nếu sent là chuỗi, chuyển thành danh sách từ
    if isinstance(sent, str):
        sent = sent.split()
    
    vectors = []
    weights = []

    for token in sent:
        # Lấy vector embedding cho token
        vector = get_embed(token)
        if vector is not None:
            # Lấy tần suất của từ
            frequency = count_full.get(token, 10)
            weight = getLog(frequency)
            
            vectors.append(vector)
            weights.append(weight)
    
    if vectors:
        # Tính vector trung bình có trọng số
        doc_vector = np.average(vectors, weights=weights, axis=0)
    else:
        # Nếu không có vector hợp lệ, trả về vector 0
        doc_vector = np.zeros(model.vector_size)
    
    return doc_vector


model = VnCoreNLP(save_dir='/kaggle/working', annotators=["wseg","ner"], max_heap_size='-Xmx4g')
model_embed = fasttext.load_model('/kaggle/working/cc.vi.300.bin')  # Load mô hình
sent = "một con vịt xòe ra 2 cái cánh bàm bàm bàm"
doc_vector = get_vector_avg_weighted_full(sent)
print(doc_vector.shape)

