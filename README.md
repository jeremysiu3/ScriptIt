# ScriptIt!

## Description
This is a transliteration program that allows users to convert Latin script using standard romanization to the language script of their choice (Bengali, Greek, Hindi, Javanese, Korean, Mandarin, Russian).

I did this project to step into the world of machine learning and AI, more specifically in NLP and neural networks. I learned a lot throughout this process, and I am eager to learn more about how ML models work. 

The languages which use an ML model are Bengali, Hindi, Korean, and Mandarin. When converting these languages, I took a hybrid approach having a dictionary of common words/phrases for the language and having the model take care of words that aren't in that dictionary. 

Bengali and Hindi used a bidirectional LSTM model for binary classification on vowels. The model would learn when to use a long vowel or a short vowel. For example, for Hindi, a user can type in "hindi" and the model will know to convert the second "i" to a long vowel, ensuring accurate transliteration.

Korean and Mandarin used an LSTM-based seq2seq model. Korean follows an alphabet, but it still required the use of machine learning because of the ambiguity of a Latin letter and its corresponding Korean letter. For example, the interchangeable use of k and g which gives the Korean ㄱ, while k can also give ㅋ. The model learns which character should be the output based on context of the word. Mandarin is the most inaccurate by far just because of its complexity which has multiple characters for one romanization. For example, shi4 can be 是, 事, 市, 试, etc. The model learns which character to use based on the sentence context. 

This was also my first time learning how to use Git, and I didn't start doing it until I was well into the project, so there are some missing files which I had permanently deleted before using Git.

## Installation
I had a lot of trouble deploying this so I think it's better that you run it locally on your computer:

### Prerequisites
Make sure you have the following installed:
- Python 3.13.9
- Git

### Clone the repository
``` bash
git clone https://github.com/jeremysiu3/ScriptIt.git

```
### Create venv
macOS / Linux:

python3.13 -m venv venv

source venv/bin/activate

Windows:

python -3.13 -m venv venv

venv\Scripts\activate

### Install dependencies
```bash
pip install -r requirements.txt
```
### Run
```bash
streamlit run app.py
```

