import albumentations as A
import cv2

image_path = r"/Users/cam/Programming/AugmentationProject/91.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transforms = A.Compose([
    A.ImageCompression(p=1.0, quality_range=[20,45]),
    A.GaussNoise(p=1.0)
])

augmented_image = transforms(image=image)
if augmented_image is not None:
    print(type(augmented_image['image']))
    rgb_image = cv2.cvtColor(augmented_image['image'], cv2.COLOR_BGR2RGB)
    cv2.imwrite('augmented.jpg', rgb_image)