# nlp_final_project

To reduce the memory used when running our project, follow these steps:
1. Please download the data from: https://www.statmt.org/europarl/
The dataset we used is called: parallel corpus Dutch-English and is under the "Download" section.
We unfortuanately could not push it to GitHub, because the dataset is too large.
2. The different 'blocks' in Main.py are called Preprocessing, Feature extraction and model creation and training and evaluation. Run Main.py with the Preprocessing block on its own by commenting out all other code, except for the Preprocessing block and the if __name__ == "__main__": part. The preprocessed pairs of sentences should now be saved locally in a .txt file called "cleaned_pairs.txt".
3. Next, comment out the Preprocessing block and uncomment the Feature Extraction block and run Main.py again. This should save 8 .txt files containing dictionaries and lists.
4. Next, comment out the Feature Extraction block and uncomment the model creation, training and evaluation block. Run Main.py again with this and you should see the training and validation updates appear in the terminal as well as the first 10 outputs (<) with the target sentence (=). Lastly, the BLEU score and test loss should be printed as well. A lossplot.png should be created, showing the training and validation loss.


Mention any dependencies or setup steps required to run your code
successfully.
Highlight any known issues, limitations, or areas for improvement in
your project.
Include any additional information or resources that may be useful.


Please download the data from: https://www.statmt.org/europarl/
The dataset we used is called: parallel corpus Dutch-English and is under the "Download" section.
We unfortuanately could not push it to GitHub, because the dataset is too large.

numpy
torch
matplotlib
torchtext