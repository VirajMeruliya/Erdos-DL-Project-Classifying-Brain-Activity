# Harmful Brain Activity Classification using Deep Learning

This project was completed as part of the Deep Learning course at the Erdos Institute.

Team members:  
[Shreya Shukla]  
[Zilu Ma]  
[Soumya Ganguly](https://github.com/soumya123ganguly)  
[Viraj Meruliya](https://github.com/VirajMeruliya)

The link for the video presentation: [video link]

## Introduction
Electroencephalography (EEG) has long been a valuable tool in the diagnosis and monitoring of neurological disorders, including epilepsy. In critically ill hospital patients, the ability to rapidly and accurately detect seizures and harmful brain activity is of utmost importance for timely intervention and treatment. The advent of machine learning techniques offers promising avenues for automating the analysis of EEG data, paving the way for more efficient and precise neurocritical care. This summary outlines the efforts undertaken in a Kaggle competition aimed at developing a model for classifying such seizures, with the overarching goal of advancing medical research and patient care in neurology and related fields.


## Methodology
We are given two sets of data as inputs:  
1. EEG data as time-series.
2. Spectrograms as images.

The data contains information of 1950 patients. The targets are expert diagnostic votes on six classes of brain activities. After some pre-precessing, we use around 17000 instances of data.

First, we select our features following standard techniques. Among 20 raw EEG features within some time frame, we choose 8 important features which represent EEG data detected from 8 different positions of the brain. We consider differences of neighboring EEG features, de-noise them, and take them as the training/inference features. 

We prefer EEG data for several reasons. First, spectrograms can be transformed from EEG (\cite{EEG},\cite{ng2022primer}). Also, the size of EEG data is significantly smaller than that of spectrograms, which leads to less memory use and faster training. Each epoch of training for EEG data finishes in around 40 seconds (depending on the model), as compared to 80 seconds for spectrograms.

For the deep learning architecture, we roughly follow that of EEGNet. We arrange 4 convolution layers with increasing kernel sizes, followed by 8-10 ResNet blocks with a fixed kernel size. Between the conv and ResNet layers, we use various techniques such as batch normalization, average/max pooling, and dropout, and we only use ReLU as the activation. As a major difference from EEGNet, we put a GRU (gated recurrent unit) layer after ResNet blocks, which is efficient at dealing with time-series data. Finally we use a fully connected layer to map to logits of 6 classes, and the loss function is KL divergence. Our models have 0.2-0.5 million parameters.


## Results and Summary
Upon running the experiments using different sets of hyperparameters, the best results we obtained are summarized in the table below: 

## Future Directions
There was a starter notebook provided at Kaggle that used `Efficientnet' to classify the data. If we use the architecture of the starter notebook, then the minimum validation losses we get are 0.8426 (for spectrogram data), and 1.2149 (for EEG data, treated as images). We also tried with different basic models like Mobilenet2 (trained on imagenet), Yolo version 8, Untrained ResNet50V2 and all of them gave results in the same range (or slightly worse). So the most efficient model from above that we finally obtained after several trials gives significantly better results that the starter notebook. In future there is the scope of using a combination of few of these architectures and/or new architectures such as transformers to get better results. 

## References

1. Jin Jing, Zhen Lin, Chaoqi Yang, Ashley Chow, Sohier Dane, Jimeng Sun, M. Brandon Westover. (2024). HMS - Harmful Brain Activity Classification . Kaggle. https://kaggle.com/competitions/hms-harmful-brain-activity-classification


