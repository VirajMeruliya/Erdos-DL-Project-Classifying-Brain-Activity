# Harmful Brain Activity Classification using Deep Learning - Erdös Deep Learning Project

## Introduction

Electroencephalography (EEG) has long been a valuable tool in the diagnosis and monitoring of neurological disorders, including epilepsy. In critically ill hospital patients, the ability to rapidly and accurately detect seizures and harmful brain activity is of utmost importance for timely intervention and treatment. The advent of machine learning techniques offers promising avenues for automating the analysis of EEG data, paving the way for more efficient and precise neurocritical care. This summary outlines the efforts undertaken in a Kaggle competition aimed at developing a model for classifying such seizures, with the overarching goal of advancing medical research and patient care in neurology and related fields.

## Methodology

We are given two sets of data as inputs: EEG as time-series numerical data and spectrograms as images for 1950 patients within some time frame. The targets are expert diagnostic votes on six classes of brain activities. After some pre-precessing, we use around 17000 instances of data.

First, we select our features following standard techniques. Among 20 raw EEG features within some time frame, we choose 8 important features which represent EEG data detected from 8 different positions of the brain. We consider differences of neighboring EEG features, de-noise them, and take them as the training/inference features. 

We prefer EEG data for several reasons. First, spectrograms can be transformed from EEG (\cite{EEG},\cite{ng2022primer}). Also, the size of EEG data is significantly smaller than that of spectrograms, which leads to less memory use and faster training. Each epoch of training for EEG data finishes in around 40 seconds (depending on the model), as compared to 80 seconds for spectrograms.

For the deep learning architecture, we roughly follow that of EEGNet. We arrange 4 convolution layers with increasing kernel sizes, followed by 8-10 ResNet blocks with a fixed kernel size. Between the conv and ResNet layers, we use various techniques such as batch normalization, average/max pooling, and dropout, and we only use ReLU as the activation. As a major difference from EEGNet, we put a GRU (gated recurrent unit) layer after ResNet blocks, which is efficient at dealing with time-series data. Finally we use a fully connected layer to map to logits of 6 classes, and the loss function is KL divergence. Our models have 0.2-0.5 million parameters.


## Results and Summary

Upon running the experiments using different sets of hyperparameters, the best results we obtained are summarized in the table below: 
\vspace{-0.5 cm}
\begin{table}[H]
\caption{Best performance table for different models using EEG data}
\vspace{-0.5 cm}
\label{sample-table}
\begin{center}
\begin{tabular}{lllllll}
\multicolumn{1}{c}{\bf nn.GRU hidden size} &\multicolumn{1}{c}{\bf num folds} &\multicolumn{1}{c}{\bf Optimizer} &\multicolumn{1}{c}{\bf learning rate} &\multicolumn{1}{c}{\bf drop prob EEGNet} &\multicolumn{1}{c}{\bf drop prob Resnet1D} &\multicolumn{1}{c}{\bf Min Val loss}
\\ \hline \\
128	&5	&Adagrad	&1.00E-03	&0.1	&0.1	&0.5\\
128 (LSTM)	&4	&Adagrad	&1.00E-03	&0	&0	&0.5\\
128	&5	&Adadelta	&1.00E-01	&0.1	&0.1	&0.48\\
128	&5	&Adadelta	&1.00E-01	&0.1	&0.1	&0.5\\
256	&5	&AdamW	&1.00E-03	&0.1	&0.2	&0.5
\end{tabular}
\end{center}
\end{table}
