from PIL import Image
import numpy, os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation, binary_opening
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    resize_width = 256
    resize_height = 256

    if (width > height):
        miny -= (width - height) / 2
        maxy += (width - height) / 2
    if (height > width):
        minx -= (height - width) / 2
        maxx += (height - width) / 2

    print(type(int(maxx)))
    cropped_image = gray_image_scaled[int(miny):int(maxy), int(minx):int(maxx)]
    scaled_cropped_image = resize(cropped_image, (resize_width, resize_height), anti_aliasing=False, mode='constant')

    # fig3, (ax4) = plt.subplots(1)
    # ax4.imshow(scaled_cropped_image , cmap="gray")
    # plt.show()

    return scaled_cropped_image

path="frames/"
Xlist=[]
Ylist=[]
for directory in os.listdir(path):
    for directory2 in os.listdir(path+directory):
        for file in os.listdir(path+directory+"/"+directory2):
            print(path+directory+"/"+directory2+"/"+file)
            importimage(path+directory+"/"+directory2+"/"+file)
            img=Image.open(path+directory+"/"+directory2+"/"+file)
            featurevector=numpy.array(img).flatten()[:50] #in my case the images dont have the same dimensions, so [:50] only takes the first 50 values
            Xlist.append(featurevector)
            Ylist.append(directory)
            break
            # exit(0)
clf=AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, Xlist, Ylist)
print(scores.mean())
