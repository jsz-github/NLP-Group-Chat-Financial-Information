from . import bilsm_crf_model
from . import process_data
import numpy as np
import pickle
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.layers.crf import crf_loss, crf_viterbi_accuracy


def load_vocab_chunks():
    with open('model/11config.pkl', 'rb') as inp:
        (vocab, chunk_tags) = pickle.load(inp)
    return vocab, chunk_tags


# _model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
# model.load_weights('model/BiLSTM-CRF.h5') # 定义好模型后，可以通过load_weights加载模型
vocab, chunk_tags = load_vocab_chunks()
print('vocab len:', len(vocab))
print('chunk_tags:', chunk_tags)
predict_text = '专业电销卡加入企业白名单耐用日呼500+稳定不封号包售后！包售后！包售后支持挑选归属地!地区可选需要的老板联系电话18715102986   微信cqy132520  谢谢！另换 金融 教育 贷款群，现诚招代理，欢迎各位前来咨询！！！！！'
str, length = process_data.process_data(predict_text, vocab)
# 自定义CRF层加载时，需要用custom_objects指定目标层
model = load_model("./model/11BiLSTM-CRF.h5", custom_objects={'CRF': CRF, 'crf_loss': crf_loss,
                                                            'crf_viterbi_accuracy': crf_viterbi_accuracy})
print(model.summary())
predict = model.predict(str)
# print(predict)
raw = predict[0][-length:]
result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]

per, loc, org = '', '', ''

# 输出每个实体类别下的名词
for s, t in zip(predict_text, result_tags):
    if t in ('B-PER', 'I-PER'):
        per += ' ' + s if (t == 'B-PER') else s
    if t in ('B-ORG', 'I-ORG'):
        org += ' ' + s if (t == 'B-ORG') else s
    if t in ('B-LOC', 'I-LOC'):
        loc += ' ' + s if (t == 'B-LOC') else s

print(['person:' + per, 'location:' + loc, 'organzation:' + org])
