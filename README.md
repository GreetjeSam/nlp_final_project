# nlp_final_project

ALL FILES SHOULD BE IN THE SAME FOLDER.

# To reduce the memory used when running our project, follow these steps:
1. Please download the data from: https://www.statmt.org/europarl/
The dataset we used is called: parallel corpus Dutch-English and is under the "Download" section. Make sure that the two datasets are in the same directory as the rest of the files.
We unfortuanately could not push it to GitHub, because the dataset is too large.
2. The different 'blocks' in Main.py are called Preprocessing, Load cleaned_pairs, Feature extraction and model creation and training and evaluation. Run Main.py with the Preprocessing block on its own by commenting out all other code, except for the Preprocessing block and the if __name__ == "__main__": part. The preprocessed pairs of sentences should now be saved locally in a .txt file called "cleaned_pairs.txt".
3. Next, comment out the Preprocessing block and uncomment the Load cleaned_pairs and Feature Extraction block and run Main.py again. This should save 8 .txt files containing dictionaries and lists.
4. Next, comment out the Feature Extraction block (keep Load cleaned_pairs block uncommented!) and uncomment the model creation, training and evaluation block. Run Main.py again with this and you should see the training and validation updates appear in the terminal as well as the first 10 outputs (<) with the target sentence (=). Lastly, the BLEU score, the baseline BLEU score using Google Translate and test loss should be printed as well. A lossplot.png should be created, showing the training and validation loss.

# As mentioned, please download the necessary datasets from the europarl website.
# Install the following packages for python:
numpy == 1.26.4
torch == 2.2.2
matplotlib == 3.8.3
torchtext == 0.17.2

# Highlight any known issues, limitations, or areas for improvement in your project.
A known issue with Main.py is that: if you have first created vocabularies based on (eg.) 2000 sentences (and they're saved locally), and then you try to run the "model creation, training and evaluation" block on more than 2000 sentences, you'll get an error, because it will contain words that are not in the vocabulary. So, make sure you make vocabularies on the number of sentences you are also running the model with.

Another issue might be that the training and validation process will get killed if your pc does not have enough memory. To fix this, you can reduce the number of paired sentences used. If you do this, be sure to change the number of epochs as well to an appropriate number. The number of epochs is the first argument in the feat_extraction.get_dataloader() call in Main.py.

Limitations of our code is that we do not use the entire dataset. Areas for improvement include using Beam search, instead of greedy search Greedy search. We also do not have many layers to our model; it is not very complex.

# Include any additional information or resources that may be useful.
https://www.statmt.org/europarl/