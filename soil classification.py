#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt


# In[64]:


IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCHS=10


# In[33]:


dataset=tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Nihala/OneDrive/Desktop/Soil Project/training1/Soil types',
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[34]:


class_names=dataset.class_names
class_names


# In[35]:


len(dataset)


# In[36]:


for image_batch,label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())


# In[38]:


for image_batch,label_batch in dataset.take(1):
    print(image_batch[0].shape)
    print(image_batch[0].numpy())


# In[40]:


for image_batch,label_batch in dataset.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))
    plt.title(class_names[label_batch[0]])
    plt.axis("off")


# In[47]:


train_split=0.8
train_size=int(len(dataset)*train_split)


# In[49]:


train_ds=dataset.take(train_size)
len(train_ds)


# In[52]:


def get_dataset_partitions_tf(ds,train_split=0.8,test_split=0.1,val_split=0.1,shuffle=True,shuffle_size=10000):
    ds_size=len(ds)
    if shuffle:
        ds.shuffle(shuffle_size,seed=12)
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    train_ds=ds.take(train_size)
    val_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)
    
    return train_ds,test_ds,val_ds


# In[53]:


train_ds,test_ds,val_ds=get_dataset_partitions_tf(dataset)


# In[56]:


resize_and_rescale=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# In[57]:


data_augmentation=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomContrast(factor=0.2),
    layers.experimental.preprocessing.RandomZoom(0.2)])


# In[60]:


input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes=5
model=models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu',input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu',input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax')            
])
model.build(input_shape=input_shape)


# In[61]:


model.summary()


# In[62]:


model.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
metrics=['acc']
)


# In[65]:


history=model.fit(
train_ds,
epochs=EPOCHS,
batch_size=BATCH_SIZE,
verbose=1,
validation_data=val_ds
)


# In[66]:


scores=model.evaluate(test_ds)
scores


# In[77]:


def predict(model,image):
    img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array=tf.expand_dims(img_array,0)
    
    predictions=model.predict(img_array)
    print(predictions)
    
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0])),2)
    return predicted_class,confidence


# In[80]:


plt.figure(figsize=(20,20))
for images,labels in test_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class,confidence=predict(model,images[i].numpy())
        actual_class=class_names[labels[i]]
        
        plt.title(f"Actual:{actual_class},\n Predicted:{predicted_class},\n Confidence:{confidence}")
        plt.axis='off'
        


# In[81]:


model.save("soilmodel.h5")


# In[ ]:


from numpy import loadtxt
from tensorflow.keras.models import load_model
 
# load model
model = load_model('simplemodel.h5')
# summarize model.
model.summary()


# In[82]:


import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming you have a trained model 'model' and datasets 'train_ds', 'val_ds', 'test_ds'
true_labels = []
predicted_labels = []

for image_batch, label_batch in test_ds:
    predictions = model.predict(image_batch)
    predicted_labels.extend(np.argmax(predictions, axis=-1))
    true_labels.extend(label_batch.numpy().astype(int))

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

print("Unique true labels:", np.unique(true_labels))
print("Unique predicted labels:", np.unique(predicted_labels))

cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heat Map')
plt.show()


# In[83]:


from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(true_labels, predicted_labels))


# In[ ]:




