import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
Categories=['cats','dogs'] 
flat_data_arr=[] #input array 
target_arr=[] #output array 
datadir='IMAGES/'
#path which contains all the categories of images 
for i in Categories: 
	
	print(f'loading... category : {i}') 
	path=os.path.join(datadir,i) 
	for img in os.listdir(path): 
		img_array=imread(os.path.join(path,img)) 
		img_resized=resize(img_array,(150,150,3)) 
		flat_data_arr.append(img_resized.flatten()) 
		target_arr.append(Categories.index(i)) 
	print(f'loaded category:{i} successfully') 
flat_data=np.array(flat_data_arr) 
target=np.array(target_arr)

#dataframe 
df=pd.DataFrame(flat_data) 
df['Target']=target 
df.shape

#input data 
x=df.iloc[:,:-1] 
#output data 
y=df.iloc[:,-1]

# Splitting the data into training and testing sets 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=77,stratify=y) 



# Update the parameters grid (if needed)
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.0001, 0.001, 0.1, 1],
              'kernel': ['rbf', 'poly']}

# Create a StratifiedKFold object with fewer splits (2 splits to avoid the error)
cv = StratifiedKFold(n_splits=2)  # Using 2 splits instead of 5

# Create the support vector classifier
svc = svm.SVC(probability=True)

# Create the model using GridSearchCV with the new number of splits
model = GridSearchCV(svc, param_grid, cv=cv)

# Training the model using the training data
model.fit(x_train, y_train)



# Testing the model using the testing data 
y_pred = model.predict(x_test) 

# Calculating the accuracy of the model 
accuracy = accuracy_score(y_pred, y_test) 

# Print the accuracy of the model 
print(f"The model is {accuracy*100}% accurate")


print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

def predict_image(path, model, categories):
    # Read and display the image
    img = imread(path)
    plt.imshow(img)
    plt.show()

    # Resize the image to 150x150
    img_resize = resize(img, (150, 150, 3))
    
    # Flatten the image and reshape it into a 2D array
    l = [img_resize.flatten()]

    # Get the probability predictions
    probability = model.predict_proba(l)

    # Display the probability for each category
    for ind, val in enumerate(categories):
        print(f'{val} = {probability[0][ind] * 100}%')

    # Print the predicted category
    print("The predicted image is: " + categories[model.predict(l)[0]])

# Now you can use this function for any image
path1 = 'nicolas-falgetelli-ihHcWBnLPtE-unsplash.jpg'
path2 = 'sofia-guaico-xqqjZznrar0-unsplash.jpg'

predict_image(path1, model, Categories)
predict_image(path2, model, Categories)

