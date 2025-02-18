import os
from extract_data import extract_data_from_xml
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image


def plot_image_with_bbs(img_path, bbs, labels):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for idx, bb in enumerate(bbs):
        start_point = (int(bb[0]), int(bb[1]))
        end_point = (int(bb[0] + bb[2]), int(bb[1] + bb[3]))
        color = (255, 0, 0)
        thickness = 2
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        line_type = 2
        text_thickness = 2
        label = labels[idx]
        text_size, _ = cv2.getTextSize(label, font, font_scale, text_thickness)

        label_background_start = (int(bb[0]), int(bb[1] - text_size[1] - 10))
        label_background_end = (int(bb[0] + text_size[0]), int(bb[1]))
        img = cv2.rectangle(
            img, label_background_start, label_background_end, color, cv2.FILLED
        )

        cv2.putText(
            img,
            label,
            (int(bb[0]), int(bb[1] - 10)),
            font,
            font_scale,
            font_color,
            text_thickness,
            line_type,
        )

    plt.imshow(img)
    plt.axis("off")
    plt.show()


def split_bounding_boxes(img_paths, img_labels, bboxes, save_dir, dataset_dir):
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    labels = []  # List to store labels

    for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes):
        img_path = os.path.join(dataset_dir, img_path)
        img = Image.open(img_path)

        for label, bb in zip(img_label, bbs):
            # Crop image
            cropped_img = img.crop(
                (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))

            # filter out if 90% of the cropped image is black or white
            if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                continue

            if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
                continue

            # Save image
            filename = f"{count:06d}.jpg"
            cropped_img.save(os.path.join(save_dir, filename))

            new_img_path = os.path.join(save_dir, filename)

            label = new_img_path + "\t" + label

            labels.append(label)  # Append label to the list

            count += 1

    print(f"Created {count} images")

    # Write labels to a text file
    with open(os.path.join(save_dir, "labels.txt"), "w") as f:
        for label in labels:
            f.write(f"{label}\n")


def read_labels_from_text_file(save_dir):
    img_paths = []
    labels = []

    # Read labels from text file
    with open(os.path.join(save_dir, "labels.txt"), "r") as f:
        for label in f:
            labels.append(label.strip().split("\t")[1])
            img_paths.append(label.strip().split("\t")[0])


def main():
    # Extract data from xml file
    dataset_dir = "data/SceneTrialTrain"
    words_xml_path = os.path.join(dataset_dir, "words.xml")
    img_paths, img_sizes, bboxes, img_labels = extract_data_from_xml(
        words_xml_path)

    # Print total number of images and bounding boxes
    # print(f"Total images: {len(img_paths)}")
    # print(f"Total bounding boxes: {sum([len(bbs) for bbs in bboxes])}")

    # Plot random image with bounding boxes
    # i = random.randint(0, len(img_paths))
    # plot_image_with_bbs(os.path.join(dataset_dir, img_paths[i]), bboxes[i],
    #                     img_labels[i])

    save_dir = "data/ocr_dataset"
    split_bounding_boxes(img_paths, img_labels, bboxes, save_dir, dataset_dir)

    read_labels_from_text_file(save_dir)


if __name__ == "__main__":
    main()
