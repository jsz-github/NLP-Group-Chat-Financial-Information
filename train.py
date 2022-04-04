from . import bilsm_crf_model
from time import time
import os
# from my_model_utils import save_training_history, save_training_log


EPOCHS = 3
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
print('train len:', len(train_x))
print('test len:', len(test_x))
# train model
start = time()
history = model.fit(train_x, train_y, batch_size=16, epochs=EPOCHS, validation_data=(test_x, test_y), verbose=1)
print('training finished in %.3f' % (time() - start))
model.save('model/11BiLSTM-CRF.h5')
path = os.path.abspath('model/img/')
# save_training_history(history, path, 'BiLSTM-CRF_10epochs')
# save_training_log(history, path, 'BiLSTM-CRF_10epochs', '10')
