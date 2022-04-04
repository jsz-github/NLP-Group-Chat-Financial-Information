'''
Created on 2020年12月28日 上午9:31:21
@Description: keras 模型工具
@Author: j
'''
import os
import time
import matplotlib.pyplot as plt


def save_training_history(history, save_path, model_name):
    """
    @param : history: 模型训练记录 
    @param : pic_name: 图片名称 
    @param : save_path: 保存路径 
    """
    fig = plt.figure()  # 新建一张图
    plt.plot(history.history['acc'], label='training acc')
    plt.plot(history.history['val_acc'], label='val acc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(save_path, model_name + '_acc.png'))
    fig = plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig(os.path.join(save_path, model_name + '_loss.png'))
    print('SAVE IMAGE FILE AT ', save_path)


def save_training_log(history, save_path, model_name, epochs=None, train_shape=None, test_shape=None, score=None):
    logFilePath = os.path.join(save_path, model_name + '_log.txt')
    fobj = open(logFilePath, 'a')
    fobj.write('---------------------------------------------------------------------------\n')
    fobj.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
    fobj.write('model name: ' + str(model_name) + '\n')
    if epochs is not None:
        fobj.write('epoch: ' + str(epochs) + '\n')
    if train_shape is not None:
        fobj.write('x_train shape: ' + str(train_shape) + '\n')
    if test_shape is not None:
        fobj.write('x_test shape: ' + str(test_shape) + '\n')
    fobj.write('training accuracy: ' + str(history.history['acc'][-1]) + '\n')
    fobj.write('training accuracy list: ' + str(history.history['acc']) + '\n')
    fobj.write('training loss list: ' + str(history.history['loss']) + '\n')
    fobj.write('testing accuracy list: ' + str(history.history['val_acc']) + '\n')
    fobj.write('testing loss list: ' + str(history.history['val_loss']) + '\n')
    if score is not None:
        fobj.write('model evaluation results: loss:' + str(score[0]) + ' acc:' + str(score[-1]) + '\n')
    fobj.write('---------------------------------------------------------------------------\n')
    fobj.write('\n')
    fobj.close()
