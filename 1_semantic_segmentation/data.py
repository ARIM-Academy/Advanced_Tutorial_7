from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 修正箇所
from keras.callbacks import ModelCheckpoint
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import sys

from metrics import *
from metrics_v2 import *
from miou import MeanIoU
from model import *
from model2 import *


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255.
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255.
        if (np.max(mask) > 1):
            mask = mask /255.
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, 
                   image_color_mode="grayscale", mask_color_mode="grayscale", 
                   image_save_prefix="image", mask_save_prefix="mask", 
                   flag_multi_class=False, num_class=2, save_to_dir=None, 
                   target_size=(256, 256), seed=1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        subset='training'
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        subset='training'
    )

    while True:  # Ensure continuous data generation
        image_batch, mask_batch = next(zip(image_generator, mask_generator)) 
        img, mask = adjustData(image_batch, mask_batch, flag_multi_class, num_class)
        yield (img, mask)


def trainGenerator_2(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        subset = 'training')
    valid_image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        subset = 'validation')
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        subset = 'training')
    valid_mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        subset = 'validation')
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def validGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        subset = 'training')
    valid_image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        subset = 'validation')
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        subset = 'training')
    valid_mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        subset = 'validation')
    valid_generator = zip(valid_image_generator, valid_mask_generator)
    for (img,mask) in valid_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    path_list = os.listdir(test_path)
    # for i in range(num_image):
    #     img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
    for file in path_list:
        img = io.imread(os.path.join(test_path, file), as_gray = as_gray)
        if (np.max(img) > 1):
            img = img / 255.
            # mask = mask / 255
            # mask[mask > 0.5] = 1
            # mask[mask <= 0.5] = 0
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255




# def saveResult(test_path,save_path,npyfile,flag_multi_class = False,num_class = 2):
#     path_list = os.listdir(test_path)
#     for i,item in enumerate(npyfile):
#         img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
#         io.imsave(os.path.join(save_path, path_list[i]),img)

def saveResult(test_path,save_path,npyfile, threshold=0.5, flag_multi_class = False,num_class = 2):
    path_list = os.listdir(test_path)
    for i,item_i in enumerate(npyfile):
        item = np.array(item_i, dtype=np.float)
        item = item[:, : , np.newaxis]
        item = (item > threshold)
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path, path_list[i]),img)


def testResize(test_path, results, num_image = 30, flag_resize = True, flag_multi_class = False, as_gray = True):
    # results_out = np.full((num_image, 512, 512, 1), 0, dtype=np.float)
    # path_list = os.listdir(test_path)
    # for i in range(num_image):
    #     img_ori = io.imread(os.path.join(test_path, path_list[i]),as_gray = as_gray)
    #     if (np.max(img_ori) > 1):
    #         img_ori = img_ori / 255.
    #     target_size = img_ori.shape
    #     img = results[i,:,:,0]
    #     img_out = trans.resize(img, target_size) if flag_resize else img
    #     results_out[i,:,:,0]  = img_out
    # return results_out
    results_out = []
    path_list = os.listdir(test_path)
    for i in range(num_image):
        img_ori = io.imread(os.path.join(test_path, path_list[i]),as_gray = as_gray)
        if (np.max(img_ori) > 1):
            img_ori = img_ori / 255.
        target_size = img_ori.shape
        img = results[i,:,:,0]
        img_out = trans.resize(img, target_size) if flag_resize else img
        results_out.append(img_out)
    # print(results_out, type(results_out))
    # results_out_array = np.array(results_out, dtype=np.float)
    return results_out


def evaluateResult(predicted_path, actual_path, num_image = 30, as_gray = True):
    # for i in range(num_image):
    #     predicted = io.imread(os.path.join(predicted_path,"%d_predict.png"%i),as_gray = as_gray)
    #     predicted = np.reshape(predicted, predicted.shape + (1,)).astype(float)/255.
    #     actual = io.imread(os.path.join(actual_path,"%d.png"%i),as_gray = as_gray)/255.
    #     actual = np.reshape(actual, actual.shape + (1,)).astype(float)
    #     tf.print('image number:', i, 'iou:', iou(actual, predicted), 'dice_coef:', dice_coef(actual, predicted) )
    path_list_predicted = os.listdir(predicted_path)
    path_list_actual = os.listdir(actual_path)
    iou_sum = 0
    dice_coef_sum = 0
    for i in range(len(path_list_actual)):
        if i < num_image:
            predicted = io.imread(os.path.join(predicted_path, path_list_actual[i]),as_gray = as_gray)
            predicted = np.reshape(predicted, predicted.shape + (1,)).astype(float)/255.
            actual = io.imread(os.path.join(actual_path, path_list_actual[i]),as_gray = as_gray)
            actual = np.reshape(actual, actual.shape + (1,)).astype(float)
            if (np.max(actual) > 1):
                actual = actual / 255.
            iou_sum += iou(actual, predicted)
            dice_coef_sum += dice_coef(actual, predicted)
            print('image name:', path_list_actual[i], 'iou:', iou(actual, predicted), 'dice_coef:', dice_coef(actual, predicted))
            # print('image name:', path_list_predicted[i], 'iou:', iou(actual, predicted), 'dice_coef:', dice_coef(actual, predicted))
    avg_iou = iou_sum/num_image
    avg_dice_coef = dice_coef_sum/num_image
    tf.print('overall_iou:', avg_iou, 'overall_dice_coef:', avg_dice_coef, output_stream="file://training.log")
    return avg_iou, avg_dice_coef


#def trainStep(model, Train_generator, Valid_generator, epochs, batchSize, mode, pretrained_weights = None):
#    # the training phase
#    total_history_dict = dict()
#    for epoch in range(epochs):
#        print('Epoch : {}'.format(epoch + 1))
#        model_checkpoint = ModelCheckpoint(pretrained_weights, monitor='loss', verbose=1, save_best_only=True)
#        history = model.fit(Train_generator, validation_data=Valid_generator, validation_steps=30, steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])  # the only para changed
#        avgiou, avgdice_coef = evaluateStep(model, batchSize=batchSize, mode=mode)
#        avg_iou = [avgiou.numpy().tolist()]
#        avg_dice_coef = [avgdice_coef.numpy().tolist()]
#        for some_key in history.history.keys():  # save and append the results from each epoch
#            current_values = []
#            current_values += history.history[some_key]
#            if some_key in total_history_dict:
#                total_history_dict[some_key].append(current_values)
#            else:
#                total_history_dict[some_key] = [current_values]


#        if {'test_iou', 'test_dice_coef'} <= total_history_dict.keys():
#            total_history_dict['test_iou'].append(avg_iou)
#            total_history_dict['test_dice_coef'].append(avg_dice_coef)
#        else:
#            total_history_dict['test_iou'] = [avg_iou]
#            total_history_dict['test_dice_coef'] = [avg_dice_coef]
#    return total_history_dict




def trainStep(model, Train_generator, Valid_generator, epochs, batchSize, mode, pretrained_weights=None):
    """
    Trains the model and tracks the metrics across epochs.
    
    Args:
        model: The model to train.
        Train_generator: Generator for training data.
        Valid_generator: Generator for validation data.
        epochs: Number of epochs to train.
        batchSize: Batch size for evaluation.
        mode: Training mode (e.g., 'unet', 'multiresunet').
        pretrained_weights: Filepath for saving model weights.
    
    Returns:
        A dictionary with metrics history.
    """
    # Initialize a dictionary to store metrics history
    total_history_dict = {}

    # Ensure the pretrained_weights filename ends with .keras
    if pretrained_weights and not pretrained_weights.endswith('.keras'):
        pretrained_weights = pretrained_weights.replace('.hdf5', '.keras')

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        
        # Configure ModelCheckpoint
        model_checkpoint = ModelCheckpoint(
            filepath=pretrained_weights,
            monitor='loss',
            verbose=1,
            save_best_only=True
        )
        
        # Train for one epoch
        history = model.fit(
            Train_generator,
            validation_data=Valid_generator,
            validation_steps=30,
            steps_per_epoch=300,
            epochs=1,
            callbacks=[model_checkpoint]
        )
        
        # Collect metrics from the training history
        for key, values in history.history.items():
            if key in total_history_dict:
                total_history_dict[key].extend(values)
            else:
                total_history_dict[key] = list(values)  # Ensure it is a list
        
        # Evaluate the model after each epoch
        avgiou, avgdice_coef = evaluateStep(model, batchSize=batchSize, mode=mode)
        
        # Add evaluation metrics
        avg_iou = float(avgiou.numpy())  # Ensure it's a Python float
        avg_dice_coef = float(avgdice_coef.numpy())
        
        if 'test_iou' in total_history_dict:
            total_history_dict['test_iou'].append(avg_iou)
        else:
            total_history_dict['test_iou'] = [avg_iou]
        
        if 'test_dice_coef' in total_history_dict:
            total_history_dict['test_dice_coef'].append(avg_dice_coef)
        else:
            total_history_dict['test_dice_coef'] = [avg_dice_coef]
    
    return total_history_dict



def evaluateStep(model, batchSize, mode, binarized_output=True, threshold=0.5):
    # testGene = testGenerator("data/microplastics/test/image")
    # results = model.predict_generator(testGene, batchSize, verbose=1)
    # results = testResize("data/microplastics/test/image", results, batchSize, flag_resize=True)
    # results = (results > threshold) if binarized_output else results
    # if mode == 'training' or mode == 'testing':
    #     saveResult("data/microplastics/test/image", "data/microplastics/result", results)
    #     avg_iou, avg_dice_coef = evaluateResult("data/microplastics/result", "data/microplastics/test/label", batchSize)
    #     print('*************')
    # elif mode == 'multiresunet_training' or mode == 'multiresunet_testing':
    #     saveResult("data/microplastics/test/image", "data/microplastics/result2", results)
    #     avg_iou, avg_dice_coef = evaluateResult("data/microplastics/result2", "data/microplastics/test/label", batchSize)
    #     print('-------------')
    # return avg_iou, avg_dice_coef
    
    testGene = testGenerator("data/microplastics/test/image")
    results = model.predict_generator(testGene, batchSize, verbose=1)
    results = testResize("data/microplastics/test/image", results, batchSize, flag_resize=True)
    results = (results) if binarized_output else results
    if mode == 'training' or mode == 'testing':
        saveResult("data/microplastics/test/image", "data/microplastics/result", results, threshold=0.5)
        avg_iou, avg_dice_coef = evaluateResult("data/microplastics/result", "data/microplastics/test/label", batchSize)
        print('*************')
    elif mode == 'multiresunet_training' or mode == 'multiresunet_testing':
        saveResult("data/microplastics/test/image", "data/microplastics/result2", results, threshold=0.5)
        avg_iou, avg_dice_coef = evaluateResult("data/microplastics/result2", "data/microplastics/test/label", batchSize)
        print('-------------')
    return avg_iou, avg_dice_coef
