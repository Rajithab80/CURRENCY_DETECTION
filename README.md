 Currency Detection using CNN

This project detects and classifies currency notes using a Convolutional Neural Network (CNN). It accepts an image of a currency note as input and predicts the corresponding denomination through a trained deep learning model.

## Features
- Automatic currency denomination detection  
- CNN-based image classification  
- Simple web interface for image upload and prediction  
- Stores prediction history for reference  

## Project Structure
- Train/ and Test/ – Training and testing image datasets  
- TRAIN.py – Script to train the CNN model and save weights  
- main.py – Main application to load the trained model and perform predictions  
- templates/ and static/ – Frontend files (HTML, CSS, JavaScript, images)  
- currency_classification.h5 – Trained CNN model file  
- detection_history.csv – Stores prediction history  
- req_3_9_0.txt – Python dependency requirements  
- upload/ – Folder for uploaded images  

## Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- Flask  
- HTML, CSS, JavaScript  

## How to Run
1. Install dependencies:
   ```bash
   pip install -r req_3_9_0.txt

2. Run the application:

python main.py


3. Open the displayed URL in your browser and upload a currency note image to view the predicted denomination.



Output

Displays the predicted currency denomination

Stores prediction details in detection_history.csv


Future Enhancements

Support for multiple currencies

Improved accuracy with larger datasets

Mobile application integration


Author

Rajitha Reddy
Computer Science Undergraduate | AI & Machine Learning Enthusiast
