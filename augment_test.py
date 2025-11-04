import albumentations as A
import cv2

image_path = r"images/dog.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transforms = A.Compose([
    # A.ImageCompression(p=1.0, quality_range=[20,45]),
    A.GaussNoise(p=1.0)
])

image_path_split = image_path.split('.')
augmented_image_path = "".join(image_path_split[:-1]) + '_augmented.' + image_path_split[-1]
augmented_image = transforms(image=image)

if augmented_image is not None:
    rgb_image = cv2.cvtColor(augmented_image['image'], cv2.COLOR_BGR2RGB)
    cv2.imwrite(augmented_image_path, rgb_image)
    print(f"Augmented image saved to: {augmented_image_path}")

