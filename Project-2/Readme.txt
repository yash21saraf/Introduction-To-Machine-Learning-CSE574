=======================================================================
The folder only has the following dataset- 
=======================================================================
1.GSC Seen writer scheme, with corresponding features subtracted. 
2.HumanObservedDataset with unseen writer, with corresponding features 
subtracted. 

=======================================================================
Instructions for preprocessing- 
=======================================================================
-The Createpairs.py is responsible for reading the AllPairs.csv files 
which is a mixture of same and different pairs is sequential order.
-The Createpairs.py just creates a new diffn_pairs_edited file which 
contains all the different pairs in proper order. 
-This was important for furthur processing 
-The DataPreprocess.py creates all the datasets required using the same
pair file, different pair file and the Features file. The same thing 
will also work for the GSC dataset. 
-The GSC dataset takes a lot of time so the datset has been uploaded on
google drive and the link is as follows. 
=======================================================================
https://drive.google.com/drive/folders/1G0PQaPGsxDiC61VrwBVhWnOHk15Go-0-?usp=sharing
=======================================================================
=======================================================================
The folder contains the following directory structure- 
=======================================================================

--Project2
  ----ysarafProj2Report.pdf
  ----HumanObservedDataset
  ----Preprocess
      ----Createpairs.py
      ----DataPreprocess.py
      ----AllPairs.csv
      ----diffn_pairs.csv
      ----HumanObserved-Features-Data.csv
      ----same_pairs.csv
  ----GSCDataset
      ----LinearRegression.py
      ----LinearRegression-Keras.py
      ----LogisticRegression.py
      ----LogisticRegression-Keras.py
      ----NeuralNetworks.py
      ----testing.csv
      ----training.csv
      ----validation.csv
  ----HumanObservedDataset
      ----LinearRegression.py
      ----LinearRegression-Keras.py
      ----LogisticRegression.py
      ----LogisticRegression-Keras.py
      ----NeuralNetworks.py
      ----testing.csv
      ----training.csv
      ----validation.csv