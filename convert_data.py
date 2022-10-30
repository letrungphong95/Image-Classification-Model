# from train_neuralnet_model import CIFAR10Dataset
import pandas as pd 
import cv2
import os


def copy_data(data_path=None, source_path=None, store_path=None):
    # data 
    data = pd.read_csv(data_path)

    all_image = list(data["id"])
    all_label = list(data["label"])
    for i, (image_name,label) in enumerate(zip(all_image, all_label)):
        if i %100 ==0:
            print("Processed data: ",i)
        new_path = os.path.join(store_path, label)
        os.makedirs(new_path, exist_ok = True) 
        image = cv2.imread(os.path.join(source_path,  f"{image_name}.png"))
        cv2.imwrite(os.path.join(new_path,  f"{image_name}.png"), image)


if __name__ == "__main__":
    """
    """
    # Train data
    copy_data(
        data_path="data/trainLabels.csv", 
        source_path="data/train", 
        store_path="new_data/train"
    )

    # Test data
    copy_data(
        data_path="data/sampleSubmission.csv", 
        source_path="data/test", 
        store_path="new_data/test"
    )