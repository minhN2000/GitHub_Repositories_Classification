# GitHub_Repositories_Classification

Classify GitHub Repositories into 7 groups: database, framework, plugin, programming language, toolkit, library, and platform using Machine Leaning method

. **Inprogress.**

## Data

The data is collected from GitHub including 13,000 training repositories. The dataset has been cleaned, and the cleaning process will be announced.

TODO:
- Apply Distill Knowledge (combine Bert and DistilBert) and Quantization
- separate training and data-processing processes


## Word Cloud Visualization

Below images are the examples of the frequencies of words appear the most in two categories: Programming Language and Database. Common keywords for Database are: data, query, sql, object, db, .etc.., and common keywords for Programming Language are: programming language, type, funciton, compiler, .etc.. You can file the rest categories' world clouds in `./data_analysis/wordcloud_visualization/`



<p float="left">
  <img src="https://github.com/minhN2000/GitHub_Repositories_Classification/blob/main/result/wordcloud_visualization/database_wordcloud.png" width="400" height="400" style="margin-right: 20px;"/>
  <img src="https://github.com/minhN2000/GitHub_Repositories_Classification/blob/main/result/wordcloud_visualization/pl_wordcloud.png" width="400" height="400"/> 
</p>

## Result

After applying DistilBert model, the accuracy improve up to ~88% compare with ~85% from applying Bidirectional LSTM. Below is the confusion matrix for the classification

![confusion matrix](https://github.com/minhN2000/GitHub_Repositories_Classification/blob/main/result/confusion_matrix/confusion%20matrix.png)
