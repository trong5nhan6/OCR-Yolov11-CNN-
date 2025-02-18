# OCR-YOLOV11-CNN Project

## Overview

This project is designed to perform Optical Character Recognition (OCR) using a combination of YOLO (You Only Look Once) for text detection and CRNN (Convolutional Recurrent Neural Network) for text recognition. The project is divided into two main parts:

1.  **Text Detection with YOLO**: This part focuses on detecting text regions within images using the YOLO model.
    
2.  **Text Recognition with CRNN**: This part involves recognizing the text from the detected regions using a CRNN model.
    

The project is structured to facilitate easy training, testing, and deployment of the OCR pipeline.

## Project Structure

The project is organized as follows:

OCR-YOLOV11-CNN/
├── data/                    # Contains datasets and data processing scripts
├── models/                  # Contains model definitions and weights
├── Yolo/                    # YOLO model implementation and utilities
├── note_book/               # Jupyter notebooks for experimentation
├── .gitignore               # Specifies files to ignore in Git
├── README.md                # This file
└── requirements.txt         # Python dependencies for the project

## Getting Started

### Prerequisites

-   Python 3.10 or higher
    
-   PyTorch
    
-   OpenCV
    
-   Other dependencies listed in  `requirements.txt`
    

### Installation

1.  Clone the repository:

>     git clone https://github.com/trong5nhan6/OCR-Yolov11-CNN-.git
>     cd OCR-YOLOV11-CNN

    
2.  Install the required dependencies:
   

>     pip install -r requirements.txt

    
3.  Prepare the datasets:
    
    -   Place your datasets in the  `data/ocr_dataset/`  directory.
        
    -   Use the provided scripts (`extract_data.py`,  `split_bbs.py`,  `unzip_data.py`) to preprocess the data.
        

### Usage

1.  **Text Detection with YOLO**:
    
    -   Navigate to the  `Yolo/`  directory.
        
    -   Train the YOLO model using the provided scripts or notebooks.
        
    -   Use the trained model to detect text regions in images.
        
2.  **Text Recognition with CRNN**:
    
    -   Navigate to the  `CRNN/`  directory.
        
    -   Train the CRNN model using the provided scripts or notebooks.
        
    -   Use the trained model to recognize text from the detected regions.
        
3.  **End-to-End OCR Pipeline**:
    
    -   Use the  `note_book/notebook3.ipynb`  to run the complete OCR pipeline, from text detection to recognition.