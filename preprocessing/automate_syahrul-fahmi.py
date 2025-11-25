import os

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np

seed = 0
np.random.seed(seed)
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from mpstemmer import MPStemmer

import csv
import requests
from io import StringIO
from joblib import dump

stemmer = MPStemmer()


# proses pembersihan teks dari karakter atau noise pada data review
def cleaning_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text


def case_folding_text(text):
    text = text.lower()
    return text


def tokenizing_text(text):
    text = word_tokenize(text)
    return text


def filtering_text(text):
    list_stopwords = set(stopwords.words('indonesian'))
    list_stopwords1 = set(stopwords.words('english'))
    list_stopwords.update(list_stopwords1)
    list_stopwords.update(
        ['iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', "di", "ga", "ya", "gaa", "loh", "kah", "woi", "woii", "woy"])
    filtered = []
    for txt in text:
        if txt not in list_stopwords:
            filtered.append(txt)
    text = filtered
    return text


def stemming_text(text):
    if not isinstance(text, str):
        return ''
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text


def to_sentence(list_words):
    sentence = ' '.join(word for word in list_words)
    return sentence


# proses normalisasi data
response_slangs_dataset = requests.get(
    'https://raw.githubusercontent.com/nasalsabila/kamus-alay/refs/heads/master/colloquial-indonesian-lexicon.csv'
)

slang_df = pd.read_csv(StringIO(response_slangs_dataset.text))

normalized_word_dict = {}

for index, row in slang_df.iterrows():
    key = str(row.iloc[0]).lower().strip()
    value = str(row.iloc[1]).strip()
    normalized_word_dict[key] = value


def word_normalize(text):
    words = text.split()
    fixed_words = []

    for word in words:
        clean_word = word.lower()
        if clean_word in normalized_word_dict:
            fixed_words.append(normalized_word_dict[clean_word])
        else:
            fixed_words.append(word)

    fixed_text = ' '.join(fixed_words)
    return fixed_text


# labeling menggunakan lexicon indonesia
lexicon_positive = dict()

response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')

if response.status_code == 200:
    reader = csv.reader(StringIO(response.text), delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])
else:
    print("Failed to fetch positive lexicon data")

lexicon_negative = dict()

response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')

if response.status_code == 200:
    reader = csv.reader(StringIO(response.text), delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
else:
    print("Failed to fetch negative lexicon data")


def sentiment_analysis_lexicon_indonesia(text):
    score = 0

    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        elif word in lexicon_negative:
            score -= abs(lexicon_negative[word])
    if score > 0:
        polarity = 'positif'
    elif score < 0:
        polarity = 'negatif'
    else:
        polarity = 'netral'

    return score, polarity


def preprocess_text_data(
        df,
        text_column,
        save_pipeline_path,
        save_clean_csv_path,
        test_size=0.2,
        random_state=42
):
    df['text_clean'] = df[text_column].apply(cleaning_text)
    df['text_casefolding_text'] = df['text_clean'].apply(case_folding_text)
    df['text_slangwords'] = df['text_casefolding_text'].apply(word_normalize)
    df['text_stemming'] = df['text_slangwords'].apply(stemming_text)
    df['text_tokenizingText'] = df['text_stemming'].apply(tokenizing_text)
    df['text_stopword'] = df['text_tokenizingText'].apply(filtering_text)
    df['text_akhir'] = df['text_stopword'].apply(to_sentence)

    # --- Sentiment Lexicon ---
    results = df['text_stopword'].apply(sentiment_analysis_lexicon_indonesia)
    df['polarity_score'] = results.apply(lambda x: x[0])
    df['polarity'] = results.apply(lambda x: x[1])

    # --- TF-IDF ---
    tfidf = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)
    )

    # --- Label Encoding ---
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['polarity'])

    X = df['text_akhir'].astype(str).values
    y = df['label'].values

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    tfidf.fit(x_train)

    pipeline = {
        "tfidf": tfidf,
        "label_encoder": le
    }
    dump(pipeline, save_pipeline_path)

    df_cleaned = df.drop(columns=['label'])
    df_cleaned.to_csv(save_clean_csv_path, index=False)

    return x_train, x_test, y_train, y_test, df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(BASE_DIR, "..", "ulasan_aplikasi_my_telkomsel_raw", "ulasan_aplikasi_my_telkomsel.csv")
    app_reviews_df = pd.read_csv(csv_path)

    output_path_csv = os.path.join(BASE_DIR, "ulasan_processed_dataset.csv")
    output_path_pipeline = os.path.join(BASE_DIR, "text_processing_pipeline.joblib")

    x_train, x_test, y_train, y_test, cleaned_df = preprocess_text_data(
        df=app_reviews_df,
        text_column='content',
        save_pipeline_path=output_path_pipeline,
        save_clean_csv_path=output_path_csv
    )

print("Preprocessing selesai")
print("Jumlah data bersih:", len(cleaned_df))
