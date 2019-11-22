#!/usr/bin/env python
# coding: utf-8

# ## Import required packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


# ## Make a path for the directory

# In[2]:


dir_data  = 'C:/Users/money/Downloads/PetImages/Training'
categories = ['Dog', 'Cat']


# ## Extract One image and convert into Greyscale

# In[3]:


for i in categories: # Here i represents no. of folders in directory 
  our_path = os.path.join(dir_data, i) # Here join fn. joins the path of our category to the existing path
  for j in os.listdir(our_path): # Here j represents no. of files in a directory
    img_arr = cv2.imread(os.path.join(our_path, j), cv2.IMREAD_GRAYSCALE) # read each image and convert it into greyscale in RGB a 3x3 matrix is formed when we convert it into the greyscale we get only 2x2 matrix
    plt.imshow(img_arr, cmap='gray')

    break 
  break


# In[4]:


print(img_arr)


# In[5]:


print(img_arr.shape)


# ## Resahpe that image

# In[6]:


img_size = 100
new_arr = cv2.resize(img_arr, (img_size, img_size))
plt.imshow(new_arr, cmap='gray')


# ## Create Training Data

# In[7]:


training_data = []

def create_training_data():

    for i in categories: # Here i represents no. of folders in directory 
        our_path = os.path.join(dir_data, i) # Here join fn. joins the path of our category to the existing path
        cat_num = categories.index(i) # Convert labels from string to no. i.e. Dog = 0, Cat = 1

        for j in tqdm(os.listdir(our_path)): # Here j represents no. of files in a directory. Tqdm represents the progress of the loop
            try:
                img_arr = cv2.imread(os.path.join(our_path, j), cv2.IMREAD_GRAYSCALE) # Read each image and convert it into greyscale in RGB a 3x3 matrix is formed when we convert it into the greyscale we get only 2x2 matrix
                new_arr = cv2.resize(img_arr, (img_size, img_size)) # Resize the image
                training_data.append([new_arr, cat_num])
            except Exception as e:
                pass


create_training_data()


# ## Balance The data

# In[8]:


count_dog = 0
count_cat = 0
for a,b in training_data:
    if b == 0:
        count_dog += 1
    if b == 1:
        count_cat += 1

while True:
    if (count_dog > count_cat):
        training_data.pop(0)
        count_dog -= 1
        
    if (count_dog < count_cat):
        training_data.pop()
        count_cat -=1
        
    if(count_dog == count_cat):
        break


# ## Shuffle The Data

# In[9]:


import random
random.shuffle(training_data)


# In[10]:


for a,b in training_data[:10]:
    print(b) # b represents the category number


# ## Create the features and labels

# In[11]:


x = [] # Features(images dataset) : training_data[0]
y = [] # Labels(category dataset) : training_data[1]


# In[12]:


for features, labels in training_data:
    x.append(features)
    y.append(labels)


# ## Reshape the data in x 

# In[13]:


x[0] # we can't pass the list of features to the NN so we have to reshape the data 


# In[14]:


x = np.array(x).reshape(-1, img_size, img_size, 1)


# In[15]:


y = np.array(y)


# In[16]:


x[0]


# ## Save the Dataset

# In[17]:


"""import pickle
# To save
pickle_out =  open("x.pickle", 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out =  open("y.pickle", 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

# To recall
pickle_in = open("x.pickle", 'rb')
x = pickle.load(pickle_in)
"""


# In[18]:


# No need if you are going in one flow 


# ## Make a CNN

# ### Import required libraries

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# ### Normalize

# In[ ]:


x = x/255.0   # To normalize


# In[ ]:


print(x[0])


# ### Model

# In[ ]:


print(x.shape)
print(x.shape[1:])


# In[ ]:


model = Sequential() #we are using a sequential model here.
# 3 convolution 0 dense 32 nodes
model.add(Conv2D(32, (3,3), input_shape = x.shape[1:]))
#Conv2D layer with 256 nodes and kernel size of (3,3).You can also experiment with different values here like 32, 64, 128 but powers of 2.
#Also we have to specify input shape so we take x shape[1:] because x.shape[0] represents the no. of sets.
model.add(Activation('relu')) # Activation = Rectified Linear Unit(relu)
model.add(MaxPooling2D(pool_size = (2,2))) # We do max pooling here. It is done only after convolutional layer is completed.

# Add another layer
model.add(Conv2D(32, (3,3), activation = 'relu'))# We can activate our layer in the same line also.
model.add(MaxPooling2D(pool_size = (2,2)))

# Add another layer
model.add(Conv2D(32, (3,3), activation = 'relu'))# We can activate our layer in the same line also.
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten()) # This converts our 3D feature maps to 1D feature vectors so that we can feed our data to Dense Layer.

model.add(Dense(1, activation = 'sigmoid')) # We add a final output layer. Here activation can be sigmoid or softmax.

# Compile the Model 
model.compile(optimizer = 'adam', loss  = 'binary_crossentropy', metrics = ['accuracy'])
# To minimize cost function we use different methods For ex :- like gradient descent, stochastic gradient descent. So these are call optimizers. We are using a adam.
# To make our model better we either minimize loss or maximize accuracy. NN always minimize loss. To measure it we can use different formulas like 'categorical_crossentropy' or 'binary_crossentropy'. Here I have used binary_crossentropy because the dataset is binary (Dog, Cat)  
# Metrics is to denote the measure of your model. Our focus is on accuracy. 


# In[1]:


model.fit(x, y, epochs=10, validation_split=0.2, batch_size=32, verbose = 2)


# In[ ]:
model.save('Cat_vs_Dog_Model')




