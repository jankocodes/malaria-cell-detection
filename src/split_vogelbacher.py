import random
import json

# Select 145 unique numbers from 0 to 1449 and save them in a list
random.seed(42)  # For reproducibility
selected_numbers = random.sample(range(0, 1450), 145)


def split_train_data(
    data_path: str = "data/preprocessed/vogelbacher23",
) -> None:

    with open(
        data_path + "/original_train.json",
        "r",
    ) as f:
        data = json.load(f)

    train_data = {}
    val_data = {}

    train_images = []
    val_images = []
    train_annotations = []
    val_annotations = []

    # Split images and annotations based on selected numbers
    for image in data["images"]:
        if image["id"] in selected_numbers:
            val_images.append(image)
        else:
            train_images.append(image)

    for annotation in data["annotations"]:
        if annotation["image_id"] in selected_numbers:
            val_annotations.append(annotation)
        else:
            train_annotations.append(annotation)

    train_data["info"] = data["info"]
    train_data["images"] = train_images
    train_data["annotations"] = train_annotations
    train_data["licenses"] = data["licenses"]
    train_data["categories"] = data["categories"]

    val_data["info"] = data["info"]
    val_data["images"] = val_images
    val_data["annotations"] = val_annotations
    val_data["licenses"] = data["licenses"]
    val_data["categories"] = data["categories"]

    # Print the keys
    print("Images in training set:", len(train_images))
    print("Images in validation set:", len(val_images))
    print("Annotations in training set:", len(train_annotations))
    print("Annotations in validation set:", len(val_annotations))

    print("Saving training data to train.json...")

    with open(
        data_path + "/train.json",
        "w",
    ) as f:
        json.dump(train_data, f)

    print("Saving validation data to val.json...")

    with open(
        data_path + "/val.json",
        "w",
    ) as f:
        json.dump(val_data, f)


def copy_val_images(
    data_path: str = "data/preprocessed/vogelbacher23",
):
    import shutil
    import os

    source_dir = data_path + "/train/"
    dest_dir = data_path + "/val/"

    os.makedirs(dest_dir, exist_ok=True)

    for number in selected_numbers:
        filename = f"{number-1}.png"
        shutil.move(
            os.path.join(source_dir, filename), os.path.join(dest_dir, filename)
        )

    print("Validation images copied to val_images directory.")


if __name__ == "__main__":

    print(selected_numbers)
    split_train_data()
    copy_val_images()

    print("Training and validation data split completed.")
