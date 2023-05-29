# TTK4900: Hydrogen sulfide detection with behavioural monitoring of salmon juveniles using stereo vision and machine learning

### 1. Developing dataset of stereo images
 - Without Stereo R-CNN: annotator.py
 - With Stereo R-CNN: 	 stereorcnn_annotator.py

### 2. Dataset for Stereo R-CNN
- Training dataset: Stereo_RCNN/data/training_data_stereorcnn
- Testing dataset:  Stereo_RCNN/data/testing_data_stereorcnn

### 3. Stereo camera calibration
- stereo_calibration.py

### 4. Training Stereo R-CNN
- trainval_net.py
- With Google Colab: setup_train_Stereo_RCNN.ipynb

### 5. Evaluation of trained Stereo R-CNN model
- test_model.py

### 6. 3D position estimation of detected objects
- 3D_pos_detections.py 

### 7. Tracking objects with estimation of velocity and acceleration
- Track a specific video over a number of frames: tracker.py
- Track many videos: tracker_all_videos.py

### 8. Developing dataset of distribution positional data
- sliding_window.py

### 9. Dataset for hydrogen sulfide classification and estimation
- Training dataset: h2s_estimation/data/training_data
- Testing dataset:  h2s_estimation/data/testing_data

### 10. Classification of hydrogen sulfide
- SVM, decision tree, random forest: classify_h2s_classifiers.py
- AutoML sk-learn: classify_automl_sklearn.ipynb

### 11. Estimation of hydrogen sulfide
- SVM, decision tree, random forest: estimation_h2s_regressors.py
- AutoML sk-learn: estimation_automl_sklearn.ipynb
- H2O AutoML: estimation_automl_H2O.ipynb

