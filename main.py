from model import *
from data import *
from datasets import *
from model2 import *
from matplotlib import pyplot as plt

import splitfolders
import shutil
import pickle
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ## KEYPOINT1

log = open("training.log", "w")
sys.stdout = log

mode = 'training'   ## KEYPOINT2

if mode == 'training' or mode == 'multiresunet_training':
    print('good')
    # shutil.rmtree('data/microplastics/train/')
    # shutil.rmtree('data/microplastics/val/')
    # shutil.rmtree('data/microplastics/test/')
    splitfolders.ratio('data/microplastics/datasets_full_v2', output="data/microplastics/", seed=2, ratio=(0.2, 0.2, 0.6))
    shutil.move('data/microplastics/train/', 'data/microplastics/1/')
    shutil.move('data/microplastics/val/', 'data/microplastics/2/')
    shutil.move('data/microplastics/test/', 'data/microplastics/6/')
    splitfolders.ratio('data/microplastics/6/', output="data/microplastics/", seed=2, ratio=(0.33, 0.33, 0.34))
    shutil.move('data/microplastics/train/', 'data/microplastics/3/')
    shutil.move('data/microplastics/val/', 'data/microplastics/4/')
    shutil.move('data/microplastics/test/', 'data/microplastics/5/')




for crossval_i in range(5):    ## KEYPOINT3
    if os.path.isdir('data/microplastics/train/') == True:
        shutil.rmtree('data/microplastics/train/')
        shutil.rmtree('data/microplastics/test/')
    os.mkdir('data/microplastics/train/')

    list_dir = ['data/microplastics/1/', 'data/microplastics/2/', 'data/microplastics/3/','data/microplastics/4/','data/microplastics/5/']
    copyTree(list_dir[crossval_i], 'data/microplastics/test/')
    del list_dir[crossval_i]
    for path in list_dir:
        copyTree(path, 'data/microplastics/train/')
    print('training-testing spliting finished for {}/5 cross validation'.format(crossval_i+1))
    print(mode)
    path_list_actual = os.listdir('data/microplastics/test/image/')
    testBatchSize = len(path_list_actual)
#
# if mode == 'training' or mode == 'multiresunet_training':
#     print('good')
#     shutil.rmtree('data/microplastics/train/')
#     shutil.rmtree('data/microplastics/test/')
#     splitfolders.ratio('data/microplastics/datasets2', output="data/microplastics/", seed=2, ratio=(.7, 0.0, 0.3))

    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest',
                        validation_split=0.1)
    trainGene = trainGenerator(2,'data/microplastics/train','image','label',data_gen_args,save_to_dir = None)
    validGene = validGenerator(2,'data/microplastics/train','image','label',data_gen_args,save_to_dir = None)

    if mode == 'training':
        model = unet()
        # model_checkpoint = ModelCheckpoint('unet_microplastics.hdf5', monitor='loss', verbose=1, save_best_only=True)
        # history = model.fit_generator(trainGene,steps_per_epoch=3,epochs=10,callbacks=[model_checkpoint])
        ## KEYPOINT4,5
        total_history_dict = trainStep(model, trainGene, validGene, epochs=100, batchSize=testBatchSize, mode=mode, pretrained_weights='unet_microplastics_{}.hdf5'.format(crossval_i))
        plt.plot(total_history_dict['accuracy'], 'b-')
        plt.plot(total_history_dict['iou'], 'g-')
        plt.plot(total_history_dict['dice_coef'], 'r-')
        plt.plot(total_history_dict['test_iou'], 'g--')
        plt.plot(total_history_dict['test_dice_coef'], 'r--')
        plt.title('Model Accuracy')
        plt.ylabel('Metrics')
        plt.xlabel('Epoch')
        plt.legend(['accuracy', 'iou', 'dice_coef','test_iou', 'test_dice_coef'], loc='upper left')
        # plt.show()
        plt.grid()
        plt.savefig('training_unet_{}.png'.format(crossval_i))
        plt.close()
        a_file = open("data_unet_{}.pkl".format(crossval_i), "wb")
        pickle.dump(total_history_dict, a_file)
        a_file.close()
        # a_file = open("data{}.pkl".format(crossval_i), "rb")
        # output = pickle.load(a_file)
        # print(output)

    elif mode == 'testing':
        model = unet('unet_microplastics.hdf5')
    elif mode == 'multiresunet_training':
        model = MultiResUnet()
        # model_checkpoint = ModelCheckpoint('unet_microplastics.hdf5', monitor='loss', verbose=1, save_best_only=True)
        # history = model.fit_generator(trainGene,steps_per_epoch=3,epochs=10,callbacks=[model_checkpoint])
        total_history_dict = trainStep(model, trainGene, validGene, epochs=100, batchSize=testBatchSize, mode=mode, pretrained_weights='multiresunet_microplastics_{}.hdf5'.format(crossval_i))
        plt.plot(total_history_dict['accuracy'], 'b-')
        plt.plot(total_history_dict['iou'], 'g-')
        plt.plot(total_history_dict['dice_coef'], 'r-')
        plt.plot(total_history_dict['test_iou'], 'g--')
        plt.plot(total_history_dict['test_dice_coef'], 'r--')
        plt.title('Model Accuracy')
        plt.ylabel('Metrics')
        plt.xlabel('Epoch')
        plt.legend(['accuracy', 'iou', 'dice_coef','test_iou', 'test_dice_coef'], loc='upper left')
        # plt.show()
        plt.grid()
        plt.savefig('training_multiresunet_{}.png'.format(crossval_i))
        plt.close()
        a_file = open("data_multiresunet_{}.pkl".format(crossval_i), "wb")
        pickle.dump(total_history_dict, a_file)
        a_file.close()
    elif mode == 'multiresunet_testing':
        model = MultiResUnet('multiresunet_microplastics.hdf5')


# results = model.predict(testGene,batch_size=30, verbose=1)
# results = testResize("data/microplastics/test/image", results, 30, flag_resize=True)
# results = (results>0.5)
#
# if mode == 'training' or mode == 'testing':
#     saveResult("data/microplastics/test/image", "data/microplastics/result", results)
#     evaluateResult("data/microplastics/result", "data/microplastics/test/label", 30)
#     print('*************')
# elif mode == 'multiresunet_training' or mode  == 'multiresunet_testing':
#     saveResult("data/microplastics/test/image", "data/microplastics/result2", results)
#     evaluateResult("data/microplastics/result2", "data/microplastics/test/label", 30)
#     print('-------------')






