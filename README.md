# Detection-of-Patronising-or-Condoscending-Language (PCL)
This project involves developing a binary classification model to predict whether a text contains patronising or condescending language, and writing a report based on the findings. The task was Task 4 (Subtask 1) in the SemEval 2022 competition, with more information available from the task paper.

## Task
Implement a transformer-based model that outperforms the task's RoBERTa-base baseline (provided by the task organisers) in F1 score. The RoBERTa-baseline achieved 0.48 on the official dev dataset, and 0.49 on the test set.

## Methodology
### 1) Data analysis
This included a written analysis of the training data and a qualitative assessment of the dataset, considering the difficulty and subjectivity of the task. To assist our analysis we plotted the correlation of various features such as the length of the next, the amount of punctuation in the text, etc against the class label. 

### 2) Modelling
This section included experimentation with transformer models such as BERT, Albert, DistillBert, BART, and DeBerta. To optimise the performance we performed hyperparameter tuning for which all the experimentation and final hyperparameter setting is outlined in the report. The first baseline was the provided RoBERTa-base baseline with an F1 score of 0.49 on the test set, however we also decided to create two simple Bag of Words models to provide us with another baseline. The two models included Naive Bayes achieving an F1-score of 0.37 and SVM achieving 0.31. This section also outlines various model improvements that we incorporated to counter the class imbalance of the dataset. The improvements included upsampling and downsampling, text augmentation with backtranslation and text augmentation with GPT-based paraphrasing. 
Our final model was DeBerta-base which achieved an F1 score on the test set of 0.58, thus significantly improved the performance of the RoBERTa-base baseline. 

### 3) Analysis
A comprehensive analysis of the model performance is available in the report.

## Software + Libraries
* Python
* PyTorch
* Pandas
* Matplotlib
* HuggingFace
* Scikit-Learn
* Transformers
* Datasets
