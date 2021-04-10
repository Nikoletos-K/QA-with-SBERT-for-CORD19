![](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white)
![](https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white)
![](https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white)
![](https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)

---

# Q&A with SBERT NN to CORD-19

Developed a document retrieval system to return titles of scientific papers containing the answer to a given user question.
I used the first version of the [COVID-19 Open Research Dataset (CORD-19)](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-03-13.tar.gz) 

###  Notebook viewer

‼️ Because of memory restrictions, GitHub and Browsers can't open  always big jupyter notebooks. 
For this reason I have every notebook linked with the ✔️ __jupyter nbviewer__ ✔️ in the following table. 
If you have any problems opening the notebooks, follow the links.

|Notebook | Link to jupyter nbviewer | Link to Colab |
|:-:|:-:| :-:| 
| __SBERT_CORD19_Preprocess.ipynb__ | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Nikoletos-K/QA-with-SBERT-for-CORD19/blob/main/SBERT_CORD19_Preprocess.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ZN6QlGAbdVWcl3RtrkvnqW6n8MAO9Ygl/view?usp=sharing) |
| __SBERT_CORD19_QA_CrossEncoders.ipynb__ | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Nikoletos-K/QA-with-SBERT-for-CORD19/blob/main/SBERT_CORD19_QA_CrossEncoders.ipynb) |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1c64s-pgZzfTM7Fm8D_NIfaUZN-f1ZM7I/view?usp=sharing) |
| __SBERT_CORD19_QA_Doc2Vec.ipynb__ |  [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Nikoletos-K/QA-with-SBERT-for-CORD19/blob/main/SBERT_CORD19_QA_Doc2Vec.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ACAHfYlbfTPMBNuAPbTzCsqJ37FRQGWq/view?usp=sharing) |
| __SBERT_CORD19_QA_InferSent.ipynb__ | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Nikoletos-K/QA-with-SBERT-for-CORD19/blob/main/SBERT_CORD19_QA_InferSent.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ECIDKDVMgwmb0F5fd7j7CtY7lb2iAL8H/view?usp=sharing) |
| __SBERT_CORD19_QA_Roberta.ipynb__ | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Nikoletos-K/QA-with-SBERT-for-CORD19/blob/main/SBERT_CORD19_QA_Roberta.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1j0Lk5JHu9riH1WejT9JxtoBjV9OY8_Dl/view?usp=sharing) |


## CORD-19
Articles in the folder ```comm_use_subset```.

## Question & Answer examples

| Question examples |  Possible Answers |
|---|---|
| What are the coronoviruses?  | Coronaviruses (CoVs) are common human and animal pathogens that can transmit zoonotically and cause severe respiratory disease syndromes.   |
| What was discovered in Wuhuan in December 2019?  | In December 2019, a novel coronavirus, called COVID-19, was discovered in Wuhan, China, and has spread to different cities in China as well as to 24 other countries.  |
| What is Coronovirus Disease 2019?  |  Coronavirus Disease 2019 (COVID-19) is an emerging disease with a rapid increase in cases and deaths since its first identification in Wuhan, China, in December 2019.  |
| What is COVID-19? |  COVID-19 is a viral respiratory illness caused by a new coronavirus called SARS-CoV-2.   |
| What is caused by SARS-COV2?  | Coronavirus disease (COVID-19) is caused by SARS-COV2 and represents the causative agent of a potentially fatal disease that is of great global public health concern.  |
|  How is COVID-19 spread?  | First, although COVID-19 is spread by the airborne route, air disinfection of cities and communities is not known to be effective for disease control and needs to be stopped.   |
|  Where was COVID-19 discovered? |  In December 2019, a novel coronavirus, called COVID-19, was discovered in Wuhan, China, and has spread to different cities in China as well as to 24 other countries. |
|How does coronavirus spread?  |  The new coronavirus was reported to spread via droplets, contact and natural aerosols from human-to-human.  |


__This repository consists of 5 notebooks__

Firstly, I needed to pre-process the CORD-19 (first version) dataset, that
is consisted of multiple papers, focused on COVID-19 pandemic and disease. This
dataset gives a lot of information for every paper and so I had to choose what I should
and shouldn’t use. I decided to use the corpus of each paper, with the pre-process of:
- Small sentences, with size < 3 words are dropped.
- Removed references to other papers.
- Removed links.
- Removed list symbols and other unicodes.
Also, I tried using Summarizer from bert-extractive-summarizer because papers corpus was extremely big in some papers and training of that model needed hours to
finish.

__Data-storage__
I am using a big dictionary of: __sentence −→ (paper_id,paper_title)__
If a sentence belongs to more than one papers, it is not a problem, as question is answered from at least one paper.
This dictionary I am storing it and reading it every time I need (.pickle file)

## Embeddings
In these two tasks I had to build multiple QA models and remark their
performance in various ways of creating the embeddings. I have implemented the
following embedding approaches:
- __SentenceBert__
  - ___Notebook:___ ```SBERT_CORD19_QA_Roberta.ipynb```
  - ___Pre-trained model:___ paraphrase-distilroberta-base-v1
  - ___Results:___ Not very good and this wasn’t expected. Only a few from the given questions answered. In mine questions, half of them had been answered with questions. I had multiple questions with extremely irrelevant answers.
I noticed that summarized papers needed much less time to transform to
embeddings with not much difference in answers. Finally, the number of
papers 6000 or 9000 in summarized papers had a very small differnce in
results and still small time of execution (in CUDA).
  - Future work: I would remove the sentences from the corpus that are questions and probably I would change the pretrained NN that I used.
- __CrossEncoders__
  - ___Notebook:___ ```SBERT_CORD19_QA_CrossEncoders.ipynb```
  - ___Pre-trained model:___
    - NN: msmarco-distilbert-base-v2 and
    - cross-encoder: ms-marco-TinyBERT-L-6
  - ___Results:___ Non summarized papers had the best results so far as lots of the
  questions were answered. Summarized papers model, didn’t have good
  performance and this due to mine summarization and not due the model
  selection (in my understanding).
  - ___Future work:___ Process again the summarized papers by increasing the ratio
  and max_length
- __InferSent__
  - ___Notebook:___ ```SBERT_CORD19_QA_InferSent.ipynb```
  - Implemented but stucked to encoding. I tried to make it work but I get some
  errors that I don’t understand. No remarks for this model
- __Doc2Vec__
  - ___Notebook:___ ```SBERT_CORD19_QA_Doc2Vec.ipynb```
  - ___Results:___ Terrible results for both summarized and not papers. I can’t understand what I am doing wrong
  - ___Future work:___ Implementation again from the very start.

### Model comparison

Finally, from my experiments the best model based on criteria:

- Time needed for creating the embeddings
- Answer success

I conclude that Sentence Transformer in combination with Cross Encoders is the best model as it has the best results in answers. The time needed for creating the embeddings is approximately the same among the models. (all these remarks for the 6000 papers)

---

© Konstantinos Nikoletos
