# Demystify languages and varieties similarity: the identification of similar languages and varieties.

README.md
==========================
Date: 05-05-2019
Authors: Sudhir Singh


### Introduction: 
Demystify languages and varieties similarity: the identification of similar languages and varieties.


### Requirements: 
Following modules are required in python 3 environment:

    1) sklearn
    2) imblearn
    3) matplotlib


### Requirements files:
The requirements.txt file contains all dependant libraries to be installed.
To install all dependant libraries, please run the following command:

    pip install -r requirements.txt


### The term project can be executed via two methods. <br>
    
    1) Using Jupyter notebook
    2) Using Python code


### 1) Using Jupyter notebook: <br>
If using Jupyter notebook, make sure you have a self created environment 
or Anaconda created environment and Jupyter notebook installed.

a) start Jupyter notebook: <br>
    
    sudhirsingh$ jupyter notebook

b) In the Jupyter notebook, browse to the folder where the code is located: <br>
c) Click on the "Term Project - language identification.ipynb" file. It will open in new tab.
d) In the new tab, click on the "Cell" and the select "Run All". <br>
e) Enter train, dev test, and test data set full file path as asked by program. <br>
f) Now program will execute and output results.

### 2) Using Python code:

a) By providing command line arguments: <br>
    
    python3 TP-code.py train.txt devel.txt test-gold.txt

b) By executing the code and then providing the input file names: <br>
    
    python3 TP-code.py

### Output:

    1) Model accuracy: with two classifier (Multinomial Naive Bayes & Linear SVM classifier)
    2) Confusion matrix - without normalization
    3) Classification report
