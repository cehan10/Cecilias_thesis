# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:51:43 2023

@author: Cecilia H. Hansen
"""
import pandas as pd
g = pd.read_pickle(r"C:\Users\Cecilia H. Holm\Documents\Speciale\final_df.pkl")


# importing 
import os
from pathlib import Path
import shutil
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import splitfolders #https://pypi.org/project/split-folders/
from skimage.transform import rescale, resize
import glob
from size_function_boundingbox import polyp_features
from dense_optic_flow import opticflow
import seaborn as sb
#%%LOADING MODEL
model_location= 'Segmentation/pre-trained-models/2023-03-02_17.09.27_-_full_run_-_60_epochs/model-AID_Unet_5-4-1-vgg19-OC_model.full_model.hdf5'

#If you load model only for prediction (without training), you need to set compile flag to False.
#and you don`t need to define your custom_loss, because the loss function is not necessary for prediction only.
#model_not_trainable = load_model(model_location, compile=False)


#If you load model to continue training, you need to define your custom loss function:
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)



def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return abs(dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred))


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


model_trainable = load_model(model_location, custom_objects = {
        'bce_dice_loss': bce_dice_loss,
        'dice_coef': dice_coef,
        'iou': iou,
    }
)

#model_not_trainable.summary()
#model_trainable.summary()

#model_weights=model_trainable.load_weights('weights-best.hdf5')


#%%MAKING A MASK TO REMOVE THE BORDER AROUND THE POLYP IMAGES
##This code is made together with Louise Thomsen using https://pyimagesearch.com/2021/01/19/opencv-bitwise-and-or-xor-and-not/ as inspiration
#All images have the same border, so we just choose one to make the mask
img = cv2.imread(r"C:\Users\Cecilia H. Holm\Documents\Speciale\Finished Annotations\11088\P1\Raw Image\CO___01-08-36___2011184.jpg")

 # Create a circular mask
h, w = img.shape[:2] #Height and width
mask = np.zeros((h, w), dtype=np.uint8) #define the mask
cx, cy = w//2, h//2 
r = min(cx, cy) #find radius
cv2.circle(mask,(cx, cy), r, 255, -1) #location on image, color
mask1 = cv2.bitwise_not(mask) #reverse the colors so the circle is white and surroundings are black

#Define new mask
h, w = img.shape[:2] #Height and width
mask = np.zeros((h, w), dtype=np.uint8) #define the mask

#create a rectangle mask
cv2.rectangle(mask, (32, 32), (w-32, h-32), 255, -1) #Define where in the image the box should be, color
mask2 = cv2.bitwise_not(mask)

#Overlay the two masks
combined_masks = cv2.bitwise_or(mask1, mask2)


remove_white = cv2.threshold(combined_masks, 0, 255, cv2.THRESH_OTSU)[1] 
final_mask = cv2.bitwise_not(remove_white)

# =============================================================================
#TESTTING THE MASK ON THE IMAGE
#overlay = cv2.bitwise_or(img, img, mask=final_mask)
#cv2.imshow("test", overlay)
 
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# =============================================================================



#%% Making dataframes and a list from the files in the train and test set

#The videos has been trimmed manually to only contain frames with the polyp in it. These are stored in the folder InputVideosSplit1 and this is what we are working with from here on

input_folder1='C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/'
def make_datasets(root, dic, list1):  
    #count number of patient directories
    for folders in os.listdir(root): 
        for file in os.listdir(root+'/'+folders): 
            folder_list[file]={}
            folder_list[file]=folders
            list1.append(folders+'/'+file)



folder_list={}
train_list=[]
make_datasets(input_folder1+'train', folder_list, train_list)
train_df = pd.DataFrame([folder_list]).T

folder_list={}
val_list=[]
make_datasets(input_folder1+'val', folder_list, val_list)
val_df = pd.DataFrame([folder_list]).T

folder_list={}
test_list=[]
make_datasets(input_folder1+'test', folder_list, test_list)
test_df = pd.DataFrame([folder_list]).T
#%%count number of videoframes in each video

def load_video1(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame) #all frames in videos

    finally:
        cap.release()
    return np.array(frames)
    
gg={} #empty dictionary
def prepare_all_videos2(df, root_dir, videolist, gg):
    
    # For each video.
    for idx, path in enumerate(videolist):
        # Gather all its frames and add a batch dimension.
        frames = load_video1(os.path.join(root_dir, path))
        gg[path]={}
        gg[path]['n_frames']=len(frames) #count number of frames and insert into dictionary
    return gg
prepare_all_videos2(train_df, "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/train/", train_list, gg)
prepare_all_videos2(val_df, "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/val/", val_list,gg)
prepare_all_videos2(test_df, "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/test/", test_list,gg)
#Make it into a dataframe
gg=pd.DataFrame(gg).T
gg.index.name='Polyp' #Making names for the index
gg.to_csv('n_frames_per_video.csv',index=True)
gg.describe() 
#%% PART 1 - THE BELOW CODE IS A CLASSIFIER CONSISTING OF A CNN AND AN RNN. CODE IS FROM https://keras.io/examples/vision/video_classification/
# CODE IS MODIFIED TO FIT THE CURRENT DATASET AND THE CNN WHICH IS THE AID-U-NET TRAINED MODEL.

files = glob.glob('files/*.csv')


#Making a function to remove the frame of images
def remove_border(img, final_mask):
    image_no_border=cv2.bitwise_or(img, img, mask=final_mask) #Final mask we made earlier
    return image_no_border

#Making the labels into unique values

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df[0])
)
print(label_processor.get_vocabulary())

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(val_df[0])
)
print(label_processor.get_vocabulary())

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(test_df[0])
)
print(label_processor.get_vocabulary())

#Defining some parameters

IMG_SIZE = 576
EPOCHS = 60
MAX_SEQ_LENGTH = 13
NUM_FEATURES = 65536 

#Function to load the videos into image frames and to remove the border
    
def load_video(path, final_mask, max_frames=0):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            #next we remove the border by masking the image with the mask we made earlier
            frame = cv2.bitwise_or(frame, frame, mask=final_mask)
            #Next lines we make the inputs fit the input format of the aid-u-net
            frame = np.uint8(resize(frame,
                                (256, 256,3),
                                mode='constant',
                                anti_aliasing=True,
                                preserve_range=True))
            frame= (lambda x: x/(255))(frame)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)
#load_video(r"C:\Users\Cecilia H. Holm\Documents\Speciale\InputVideosSplit1\train\PolypsBetween6and10mm\13403-P1.mpg", final_mask)



feature_extractor = load_model(model_location, custom_objects = {
        'bce_dice_loss': bce_dice_loss,
        'dice_coef': dice_coef,
        'iou': iou,
    }
)

print(feature_extractor.summary())

def prepare_all_videos(df, root_dir, videolist, final_mask):
    num_samples = len(df)
    labels = df[0].values #find the category in the dataframe
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool") #make an empty array of zeros with the given shape. Inputs later should be boolean
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"  #make an empty array of zeros with the given shape. Later inputs should be float
    )

    # For each video.
    for idx, path in enumerate(videolist):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path), final_mask)
        print(os.path.join(root_dir, path))
        #print(frames)
        frames = frames[None, ...] #Adds one more dimension
        #print(frames)
        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool") #make an empty array of zeros with the given shape. Inputs later should be boolean
        #print(temp_frame_mask)
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32" #make an empty array of zeros with the given shape. Later inputs should be float
        )
        #print(temp_frame_features)
        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames): #for each number of video, and each batch of frames for that video
            video_length = batch.shape[0] #first coloumn in the batch element
            length = min(MAX_SEQ_LENGTH, video_length) #maybe leave out this and just say length = video_lenght? But some are very many frames long
            #print(length)
            
            for j in range(length): #for each frame in the defined length
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                ).flatten() #predict the frame with aid-u-net and flatten the output dimensions
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked. Set all to true

        frame_features[idx,] = temp_frame_features.squeeze() #remove all 1's from array shape
        
        frame_masks[idx,] = temp_frame_mask.squeeze()  #remove all 1's from array shape

    return (frame_features, frame_masks), labels


#THESE CAN BE LOADED BELOW
train_data, train_labels = prepare_all_videos(train_df, "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/train/", train_list, final_mask)
val_data, val_labels = prepare_all_videos(val_df, "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/val/", val_list, final_mask)
test_data, test_labels = prepare_all_videos(test_df, "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/test/", test_list, final_mask)



#print(f"Frame features in train set: {train_data[0].shape}")
#print(f"Frame masks in train set: {train_data[1].shape}")

path1=r'C:/Users/Cecilia H. Holm/Documents/Speciale/'

##SAVE THE DATA

with open('train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    
with open('train_labels.pkl', 'wb') as f:
    pickle.dump(train_labels, f)

with open('val_data.pkl', 'wb') as f:
    pickle.dump(val_data, f)
    
with open('val_labels.pkl', 'wb') as f:
    pickle.dump(val_labels, f)
    
with open('test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)
    
with open('test_labels.pkl', 'wb') as f:
    pickle.dump(test_labels, f)
    
##LOAD THE DATA AGAIN

with open('train_data.pkl', 'rb') as f:
     train_data =  pickle.load(f)
     
with open('train_labels.pkl', 'rb') as f:
     train_labels =  pickle.load(f)
     
with open('val_data.pkl', 'rb') as f:
     val_data =  pickle.load(f)
     
with open('val_labels.pkl', 'rb') as f:
     val_labels =  pickle.load(f)
     
with open('test_data.pkl', 'rb') as f:
     test_data =  pickle.load(f)
     
with open('test_labels.pkl', 'rb') as f:
     test_labels =  pickle.load(f)


# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    x = keras.layers.LSTM(16, recurrent_dropout=0.5, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.LSTM(8, recurrent_dropout=0.5)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

print(get_sequence_model().summary())

# Utility for running experiments.
def run_experiment():
    filepath = "./tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        #validation_split=0.2,
        validation_data=([val_data[0], val_data[1]],
        val_labels),
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


hist, sequence_model = run_experiment()


##Accuracy of classification - graph

plt.ylim(0, 1) #defining the y-axis
plt.title('Accuracy of classification')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#defining what to plot
plt.plot(hist.history['accuracy'], color = 'orange', label = 'Training')
plt.plot(hist.history['val_accuracy'], color = 'green', label = 'Validation')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.show()

##Traning and validation loss plot - graph

plt.plot(range(len(hist.history['loss'])), hist.history['loss'], '-', color='r', label='Training loss')
plt.plot(range(len(hist.history['val_loss'])), hist.history['val_loss'], '--', color='r', label='Validation loss')
plt.plot(range(len(hist.history['accuracy'])), hist.history['accuracy'], '-', color='b', label='Training accuracy')
plt.plot(range(len(hist.history['val_accuracy'])), hist.history['val_accuracy'], '--', color='b', label='Validation accuracy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

#%% Predicting on the test images + confusion matrix

preds = get_sequence_model().predict([test_data[0], test_data[1]]) #make predictions on the test data
predsclass=np.argmax(get_sequence_model().predict([test_data[0], test_data[1]]), axis=-1)

con_mat=tf.math.confusion_matrix(
    test_labels,
    predsclass
)
con_mat_df= pd.DataFrame(con_mat,
                     index = ['PolypsBetween10and20mm', 'PolypsBetween6and10mm', 'PolypsOver20mm', 'PolypsUnder6mm'], 
                     columns = ['PolypsBetween10and20mm', 'PolypsBetween6and10mm', 'PolypsOver20mm', 'PolypsUnder6mm'])

figure = plt.figure(figsize=(8, 8))
sb.heatmap(con_mat_df, annot=True, cmap='mako')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(test_labels, predsclass)))
print('Micro Precision: {:.2f}'.format(precision_score(test_labels, predsclass, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(test_labels, predsclass, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(test_labels, predsclass, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(test_labels, predsclass, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(test_labels, predsclass, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(test_labels, predsclass, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(test_labels, predsclass, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(test_labels, predsclass, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(test_labels, predsclass, average='weighted')))
#%%Predicting on a single image

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :]).flatten()
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()
    frames = load_video(os.path.join("test", path), final_mask)
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames


# This utility is for visualization.


test_video = np.random.choice(test_df.index.values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)

print('True label of video is: ' + test_df.loc[test_video])


#%% PART TWO - MAKING A MODEL WITH NEW FEATURES



IMG_SIZE = 576
BATCH_SIZE = 64
EPOCHS = 60

MAX_SEQ_LENGTH = 13
NUM_FEATURES = 8

#New function to prepare all videos

def prepare_all_videos1(df, root_dir, videolist, final_mask):
    num_samples = len(df)
    labels = df[0].values #find the category in the dataframe df
    labels = label_processor(labels[..., None]).numpy() #make the labels readable for the model

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool") #array of zeros
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32" #array of zeros
    )

    # For each video.
    for idx, path in enumerate(videolist):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path), final_mask)
        print(os.path.join(root_dir, path))
        #print(frames)
        frames = frames[None, ...] #Adds one more dimension
        #print(frames)
        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool") #make array of zeros (temporarily)
        #print(temp_frame_mask)
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32" #make array of zeros (temporarily)
        )
        #print(temp_frame_features)
        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames): #i is just a zero that we have put around the dimensions
           
            video_length = batch.shape[0] #length of the frames from the video. Batch shape (no_frames, 256,256,3)
            length = min(MAX_SEQ_LENGTH, video_length) # Whichever has the least amount of frames
            movementparamvid=opticflow(batch[0:length]) #find optic flow magnitude parameters for the first 13 frames
            #predict on the videoframes with aid-u-net
            #print(movementparamvid)
            aidunetparam=feature_extractor.predict(batch[0:length]) #predict on the 13 videoframes with aid-u-net
            #aidunetparam=feature_extractor.predict(batch) #predict on the videoframes with aid-u-net
            for j in range(length): #for each of the 13 frames, put them in to a numpy array and concatenate the two arrays (so we get all 8 of the features stored together with the frame id )
                sizeparam=np.array(polyp_features(aidunetparam[j]), dtype=np.float32) 
                [movementparam]=np.array(movementparamvid[j], dtype=np.float32)
                movementparam=np.array(movementparam, dtype=np.float32)
                temp_frame_features[i, j, :] = np.concatenate([movementparam, sizeparam])
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
            
        frame_features[idx,] = temp_frame_features.squeeze() #put all of the information next to the id of the video
        
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


#THESE CAN BE LOADED BELOW
train_data1, train_labels1 = prepare_all_videos1(train_df, "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/train/", train_list, final_mask)
val_data1, val_labels1 = prepare_all_videos1(val_df, "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/val/", val_list, final_mask)
test_data1, test_labels1 = prepare_all_videos1(test_df, "C:/Users/Cecilia H. Holm/Documents/Speciale/InputVideosSplit1/test/", test_list, final_mask)



print(f"Frame features in train set: {train_data1[0].shape}")
print(f"Frame masks in train set: {train_data1[1].shape}")

path1=r'C:/Users/Cecilia H. Holm/Documents/Speciale/'

##SAVE THE DATA

with open('train_data1.pkl', 'wb') as f:
    pickle.dump(train_data1, f)
    
with open('train_labels1.pkl', 'wb') as f:
    pickle.dump(train_labels1, f)
    
with open('test_data1.pkl', 'wb') as f:
    pickle.dump(test_data1, f)
    
with open('test_labels1.pkl', 'wb') as f:
    pickle.dump(test_labels1, f)
    

with open('val_data1.pkl', 'wb') as f:
    pickle.dump(val_data1, f)
    
with open('val_labels1.pkl', 'wb') as f:
    pickle.dump(val_labels1, f)
    
##LOAD THE DATA AGAIN

with open('train_data1.pkl', 'rb') as f:
     train_data1 =  pickle.load(f)
     
with open('train_labels1.pkl', 'rb') as f:
     train_labels1 =  pickle.load(f)
with open('val_data1.pkl', 'rb') as f:
     val_data1 =  pickle.load(f)
     
with open('val_labels.pkl', 'rb') as f:
     val_labels1 =  pickle.load(f)
     
with open('test_data1.pkl', 'rb') as f:
     test_data1 =  pickle.load(f)
     
with open('test_labels1.pkl', 'rb') as f:
     test_labels1 =  pickle.load(f)


# Utility for running experiments.
def run_experiment2():
    filepath = "./tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history2 = seq_model.fit(
        [train_data1[0], train_data1[1]],
        train_labels1,
        #validation_split=0.2,
        validation_data=([val_data1[0], val_data1[1]],
        val_labels1),
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data1[0], test_data1[1]], test_labels1)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history2, seq_model


hist2, sequence_model = run_experiment2()


##Accuracy of classification - graph

plt.ylim(0, 1) #defining the y-axis
plt.title('Accuracy of classification for model with selected features')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#defining what to plot
plt.plot(hist2.history['accuracy'], color = 'orange', label = 'Training')
plt.plot(hist2.history['val_accuracy'], color = 'green', label = 'Validation')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.show()

##Traning and validation loss plot - graph

plt.plot(range(len(hist2.history['loss'])), hist2.history['loss'], '-', color='r', label='Training loss')
plt.plot(range(len(hist2.history['val_loss'])), hist2.history['val_loss'], '--', color='r', label='Validation loss')
plt.plot(range(len(hist2.history['accuracy'])), hist2.history['accuracy'], '-', color='b', label='Training accuracy')
plt.plot(range(len(hist2.history['val_accuracy'])), hist2.history['val_accuracy'], '--', color='b', label='Validation accuracy')
plt.title('Training and validation loss for model with selected features')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()


#%% Predicting on the test images + confusion matrix

preds = get_sequence_model().predict([test_data1[0], test_data1[1]]) #make predictions on the test data
predsclass1=np.argmax(get_sequence_model().predict([test_data1[0], test_data1[1]]), axis=-1)

con_mat1=tf.math.confusion_matrix(
    test_labels1,
    predsclass1
)
con_mat_df1= pd.DataFrame(con_mat1,
                     index = ['PolypsBetween10and20mm', 'PolypsBetween6and10mm', 'PolypsOver20mm', 'PolypsUnder6mm'], 
                     columns = ['PolypsBetween10and20mm', 'PolypsBetween6and10mm', 'PolypsOver20mm', 'PolypsUnder6mm'])

figure = plt.figure(figsize=(8, 8))
sb.heatmap(con_mat_df1, annot=True, cmap='mako')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#metrcics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(test_labels1, predsclass1)))
print('Micro Precision: {:.2f}'.format(precision_score(test_labels1, predsclass1, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(test_labels1, predsclass1, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(test_labels1, predsclass1, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(test_labels1, predsclass1, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(test_labels1, predsclass1, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(test_labels1, predsclass1, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(test_labels1, predsclass1, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(test_labels1, predsclass1, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(test_labels1, predsclass1, average='weighted')))
