# PAN20-Authorship-Verification

Determining whether two bodies of texts are written by the same author is the core of authorship 
verification. To accurately assess whether to texts share author, there exist a need for a feature 
set which catches the key stylometric features unique for all authors. The task undertaken in this 
project was to find a suitable feature set for determining whether two texts, from a collection of 
texts with unknown authors, were written by the same author. Four different feature sets are 
presented and evaluated on how well they enable a model to simply verify whether two unknown 
texts share authors. 

<br></br>

<h2><b>Features</b></h2>
Most of the feature sets consider words and their frequency in the texts for stylometric 
representation for different authors. 

<h3><b>n most frequent words in corpus</b></h3>
One key aspect to consider with the task is that there are thousands of authors writing fictional 
texts within different fandoms. This combined with the fact that the model is tasked with 
determining whether two texts share author, there existed a need for representing every text, and 
thereby every text pair, in an equal manner. The initial approach was to extract the n most 
frequent words after the data was cleaned and return those words in a descending order.

<br></br>

<h3><b>Set of words shared by all texts</b></h3>
Some of the more common words in the corpus may not exist in all texts or even all text pairs to 
be considered. Considering the difference in word frequency between two texts where the word 
does not exist is of little value. Solely considering the words appearing in all texts 
should avoid such scenarios.

<br></br>

<h3><b>Function words</b></h3>
Function words are defined in the oxford dictionary as “a word that is important to the grammar 
of a sentence rather than its meaning, for example ‘do’ in ‘we do not live here’”.
The rate at which different words are used will vary between authors. Thus, one 
can reveal differences in writing style through simply inspecting how the usage of function 
words differ between documents. This approach was originally introduced by Frederick 
Mosteller and Douglas Wallace when attempting to attribute authorship for the 12 disputed 
essays from the famous Federalist papers. In this project 70 such function words were used. 
Every document is represented by a list containing the frequency for each function word

<br></br>

<h3><b>Most frequent n-grams</b></h3>
Considering sequences of characters instead of words is a more fine-grained approach. This 
feature set would include the n most frequent 4-character ngrams from the entire corpus.

<br></br>

<h3><b>Dataset</b></h3>
The dataset is provided by PAN for the PAN 2020 Authorship verification contest (PAN, 2021). 
PAN provided the data in two different sizes, large and small. All data provided by PAN was 
collected from the website FanFiction.net. This website enables their users to submit and share 
their own fictional works. The dataset consists of several tens of thousands of entries holding 
containing the two texts both containing  roughly 21,000 characters. Each such entity also 
containing a unique id for the author(s) of the two texts. In addition, as all texts are works of 
fiction, the appropriate fandom is provided for both texts. The dataset will not be found in this repository,
but it may be obtained through requesting access at https://zenodo.org/record/3724096
