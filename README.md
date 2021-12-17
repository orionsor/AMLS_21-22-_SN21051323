# AMLS_21-22-_SN21051323

 3000 brain MRI scans are given as the dataset. Main task for the final assignment contains two parts : one is to build a classifier for binary claasification, detecting if there is tumor in the brain or not; the other one is to build a classifier for multiclassification, which is supposed to identify 4 types of MRI scan, including brain with meningioma tumor, glioma tumor, pituitary tumor and no tumor. all classifiers are built with machine learning techniques.
## File Organization
 
 All Files are saved in subcatories of folder 'Final' in the main branch. Two folder, Task1 and Task2, can be found when entering 'Final', which contains files of binary and multiclass task respectively. Structure for both task is quite similar. Taking Task1 for example, all the '.py' files are located at the first level of folder 'Final/Task1', subdirectory 'dataset' contains dataset-related files such as original images, label files and newly generated dataset files.
 
 ## Files In Final/Task1
 Files in this folder are involved with binary classification task, which can be devided into 3 part in terms of role. 
 ### Dataset Preparation
 1. 'add_binary_label.py' adds binary labels "Positive" and "Negative" to data and generates new label file in 'Final/Task1/dataset/binary_label.csv', which does not need to be run as related files are already there.
 2. 'dataset_preprocess.py' is used to generate dataset used for non-deep machine learning aprroaches. Dataset is saved in 'Final/Task1/dataset/data.p' and 'Final/Task1/dataset/label.p' in advance to save time for loading data. data.p is not in the folder because its size is to large to upload. So this file is needed to be run at first for implementation of ML models.
 3. 'dataset_cnn.py' mainly contains a Dataset class in pytorch. This file is used to generate dataset for CNN models and does not need to be run. Class in the file will be called by other files.
 ### Non-deep Machine Learning Approaches
 dataset_preprocess.py should be run before these files.
 
 1.'main.py' is implementation of rbf-kernel SVM combined with PCA. run the file and perfomance report will be printed when training and testing is done. Optimized hyper parameters are also printed during its running.
 2. 'main_linear_svm.py' is implementation of linear SVM combined with PCA. The rest is same as 1.
 3. 'main_regression.py' is implementation of linear SVM combined with PCA. The rest is same as 1.
 ### CNN based Deep Learning Approaches
 Two GPU from server in UCL is used for acceleration, please use cpu if CUDA is out of memory  
 
 1. 'main_resnet.py' contains training and tesing of ResNet 18. Loss information is realtime printed epoch by epoch, and a learning curve graph will be shown in the end.
 2. 'main_res34.py' contains training and tesing of ResNet 34. the rest is the same as above.
 3. 'main_vgg16.py' contains training and tesing of VGG 16. the rest is the same as above.
 
  ## Files In Final/Task2
 Files in this folder are involved with multiclassification task, which can be devided into 3 part in terms of role. 
 ### Dataset Preparation
 1. 'dataset_ml.py' is used to generate dataset used for non-deep machine learning aprroaches. Dataset is saved in 'Final/Task2/dataset/data.p' and 'Final/Task2/dataset/label.p' in advance to save time for loading data. data.p is not in the folder because its size is to large to upload. So this file is needed to be run at first for implementation of ML models.
 2. 'dataset_rgb.py' mainly contains a Dataset class in pytorch. This file is used to generate dataset for CNN models and does not need to be run. Class in the file will be called by other files.
 ### Non-deep Machine Learning Approaches
 dataset_ml.py should be run before these files.
 Two GPU from server in UCL is used for acceleration, please use cpu if CUDA is out of memory  
 1.'main_svm.py' is implementation of rbf-kernel SVM combined with PCA. run the file and perfomance report will be printed when training and testing is done. Optimized hyper parameters are also printed during its running.
 ### CNN based Deep Learning Approaches
 Two GPU from server in UCL is used for acceleration, please use cpu if CUDA is out of memory  
 
 1. 'main_resnet.py' contains training and tesing of ResNet 18. Loss information is realtime printed epoch by epoch, and a learning curve graph will be shown in the end.
 2. 'main_res34.py' contains training and tesing of ResNet 34. the rest is the same as above.
 3. 'main_res50.py' contains training and tesing of ResNet 50. the rest is the same as above.
 4. 'main_vgg16.py' contains training and tesing of VGG 16. the rest is the same as above.
 5. 'main_vgg19.py' contains training and tesing of VGG 19. the rest is the same as above.
 6. 'main_test.py', additional test set is input to an optimized model of ResNet 50 which is save in advance as 'Final/Task2/modelterm_best_res50_lr.pth'. Test set is saved at 'Final/Task2/dataset/test'
 7. 'pytorchtools.py' contains the function of earlystopping which is called by other files. Related code is from the open source https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
 
 ## Necessary Library
 the code run locally with python 3.6.8. following libriaries are used:
 torch, torchvision, os, numpy, pandas, matlotlib, torch.utils.data, cv2, PIL, pickle

 
 
