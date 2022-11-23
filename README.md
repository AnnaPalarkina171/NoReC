# NoReC
Various models fine tuned  on NoReC <br>
  
train_grid.py - this is straightforward NorBert2 model finetuning on NoReC - [result slurm](https://github.com/AnnaPalarkina171/NoReC/blob/main/results/slurm-6426257.out). As you an see the scores on small classes(0,1,5) are bad so we tried to add augmented texts to these classes.
<br>

augmentation.py - script for creating augmented texts. The idea is to mask one word in the text and unmask it using roberta (also tried norbert, no difference). To choose the augmented word we:
- check if the new word is not the same as masked
- check if the new word have same sentiment polarity as masked word. For this two lists of [negative](https://github.com/AnnaPalarkina171/NoReC/blob/main/Fullform_Negative_lexicon.txt) and [positive](https://github.com/AnnaPalarkina171/NoReC/blob/main/Fullform_Positive_lexicon.txt) norwegian words were used. 
- check if the confidence of the model is at least 0.5
<br> 

Therefore, we tried to train with one augmented class at a time:
- 
<br>
