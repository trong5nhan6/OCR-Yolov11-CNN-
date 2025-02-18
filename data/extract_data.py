from types import EllipsisType
import xml.etree.ElementTree as ET
import os
import shutil
from sklearn.model_selection import train_test_split


def extract_data_from_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []

    for img in root:
        bbs_of_img = []
        labels_of_img = []

        for bbs in img.findall('taggedRectangles'):
            for bb in bbs:
                if not bb[0].text.isalnum():
                    continue
                bbs_of_img.append([
                    float(bb.attrib['x']),
                    float(bb.attrib['y']),
                    float(bb.attrib['width']),
                    float(bb.attrib['height'])
                ])

                labels_of_img.append(bb[0].text.lower())

        img_paths.append(img[0].text)
        img_sizes.append((int(img[1].attrib['x']), int(img[1].attrib['y'])))
        bboxes.append(bbs_of_img)
        img_labels.append(labels_of_img)

    return img_paths, img_sizes, bboxes, img_labels


def convert_to_yolo_format(image_paths, image_sizes, bounding_boxes):
    yolo_data = []
    for image_path, image_size, bboxes in zip(image_paths, image_sizes, bounding_boxes):
        image_width, image_height = image_size
        yolo_labels = []

        for bbox in bboxes:
            x, y, w, h = bbox
            # calculate normalized bounding box coordinates
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            normalized_width = w / image_width
            normalized_height = h / image_height

            # Because we only have one class, we set class_id to 0
            class_id = 0

            # Convert to YOLO format
            yolo_label = f"{class_id} {center_x} {center_y} {normalized_width} {normalized_height}"
            yolo_labels.append(yolo_label)

        yolo_data.append((image_path, yolo_labels))

    return yolo_data


def save_data(data, src_img_dir, save_dir):
    # Create folder if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Make images and labels folder
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)

    for image_path, yolo_labels in data:
        # Copy imagesto images folder
        shutil.copy(
            os.path.join(src_img_dir, image_path),
            os.path.join(save_dir, 'images')
        )

        # Save labels to labels folder
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]

        with open(os.path.join(save_dir, 'labels', f'{image_name}.txt'), 'w') as f:
            for label in yolo_labels:
                f.write(f'{label}\n')


def main():
    dataset_dir = "data/SceneTrialTrain"
    words_xml_path = os.path.join(dataset_dir, "words.xml")
    img_paths, img_sizes, bboxes, img_labels = extract_data_from_xml(
        words_xml_path)

    # print(f"Number of images: {len(img_paths)}")
    # print(f"Example image path: {img_paths[0]}")
    # print(f"Example image size: {img_sizes[0]}")
    # print(f"Example bounding boxes: {bboxes[0][:2]}")
    # print(f"Example labels: {img_labels[0][:2]}")

    # Convert data to YOLO format
    yolo_data = convert_to_yolo_format(img_paths, img_sizes, bboxes)

    seed = 0
    val_size = 0.2
    test_size = 0.125
    is_shuffle = True
    train_data, test_data = train_test_split(
        yolo_data,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    test_data, val_data = train_test_split(
        test_data,
        test_size=test_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    save_yolo_data_dir = 'data/yolo_data'
    os.makedirs(save_yolo_data_dir, exist_ok=True)
    save_train_dir = os.path.join(save_yolo_data_dir, 'train')
    save_val_dir = os.path.join(save_yolo_data_dir, 'val')
    save_test_dir = os.path.join(save_yolo_data_dir, 'test')

    save_data(train_data, dataset_dir, save_train_dir)
    save_data(val_data, dataset_dir, save_val_dir)
    save_data(test_data, dataset_dir, save_test_dir)


if __name__ == "__main__":
    main()
