from PIL import Image
import numpy, os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import RidgeCV

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation, binary_opening
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import LabelEncoder


def importimage(filename):
    image = imread(filename, as_gray=True)
    # crop top and bottom 100 pixels
    image = image[100:620, 0:405]

    gray_image_scaled = image * 255

    threshold_value = threshold_otsu(gray_image_scaled)
    binary_image = gray_image_scaled > threshold_value
 
    binary_image = numpy.invert(binary_image)

    # Dilate image to ensure that small parts of the usb are recognized as one object
    binary_image = binary_dilation(binary_image)
    binary_image = binary_dilation(binary_image)
    binary_image = binary_dilation(binary_image)

    label_image = measure.label(binary_image)

    minx = 9999
    miny = 9999
    maxx = 0 
    maxy = 0
    for region in regionprops(label_image):
        if region.area < 500:
            # remove too small areas
            continue
        
        # the bounding box coordinates
        minRow, minCol, maxRow, maxCol = region.bbox
        if (minx > minCol):
            minx = minCol
        if (miny > minRow):
            miny = minRow
        if (maxx < maxCol):
            maxx = maxCol
        if (maxy < maxRow):
            maxy = maxRow

    width = maxx - minx
    height = maxy - miny    
    # fig2, (ax3) = plt.subplots(1)
    # ax3.imshow(gray_image_scaled, cmap="gray")
    # rectBorder = patches.Rectangle((minx, miny), width, height, edgecolor="red", linewidth=2, fill=False)
    # ax3.add_patch(rectBorder)
    # plt.show()

    resize_width = 64
    resize_height = 64

    if (width > height):
        miny -= (width - height) / 2
        maxy += (width - height) / 2
    if (height > width):
        minx -= (height - width) / 2
        maxx += (height - width) / 2

    cropped_image = gray_image_scaled[int(miny):int(maxy), int(minx):int(maxx)]
    try:
        scaled_cropped_image = resize(cropped_image, (resize_width, resize_height), anti_aliasing=False, mode='constant')
    except:
        return None
    # fig3, (ax4) = plt.subplots(1)
    # ax4.imshow(scaled_cropped_image , cmap="gray")
    # plt.show()

    return scaled_cropped_image

path="frames/"
labels = []
imgs = []
for directory in os.listdir(path):
    for directory2 in os.listdir(path+directory):
        for file in os.listdir(path+directory+"/"+directory2):
#            print(path+directory+"/"+directory2+"/"+file)
            img = importimage(path+directory+"/"+directory2+"/"+file)
            if img is not None:
                labels.append(directory)
                imgs.append(img.reshape(-1))
            # exit(0)

print("Imported {} images.".format(len(imgs)))

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import time

def cross_validation(model, num_of_folds, training_data, training_labels):
    print("Starting cross_validation")
    accuracy_result = cross_val_score(model, training_data, training_labels, cv=num_of_folds)
    print(accuracy_result)
    print("End cross_validation")

t0 = time.time()
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# svc_model = SVC(kernel='linear', probability=True)
# svc_model = SVC(kernel='poly', probability=True, gamma='scale', coef0=0.5)
# cross_validation(svc_model, 4, imgs, labels)
# cross_validation(svc_model, 10, imgs, labels) # good one!
# print("Time taken: {} seconds.".format(time.time() - t0))

t0 = time.time()
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier
# dtf = DecisionTreeClassifier(random_state=42)
# cross_validation(dtf, 10, imgs, labels)
# print("Time taken: {} seconds.".format(time.time() - t0))

t0 = time.time()
# KNeighbor
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=5)
# cross_validation(knc, 10, imgs, labels)
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.33, random_state=42)
knc.fit(X_train, y_train)
result1 = knc.predict(X_test)
result2 = knc.predict_proba(X_test[0:10])
print(result2)
correct = 0
wrong = 0
for i in range(0, len(X_test)):
    if(y_test[i] == result1[i]):
        correct += 1
    else:
        wrong += 1
print("Accuracy: {}% ({}/{})".format(correct / (correct + wrong) * 100, correct, correct + wrong))
print("Time taken: {} seconds.".format(time.time() - t0))

# print("Start fit")
# svc_model.fit(imgs, labels)
# print("End fit")

# lb = LabelEncoder()
# encodedLabels = lb.fit_transform(labels)
# x_train, x_test, y_train, y_test  = train_test_split(encodedLabels, imgs, train_size=.8, shuffle=True)
# model = RidgeCV(alphas=numpy.arange(0,10,.2), cv=10)
# model.fit(x_train, y_train)
# predictions = model.predict(x_test.reshape(1, -1))
# score = model.score(x_test, y_test)
# print(score)