import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest

# Set the maximum length of the summary (in sentences)
MAX_SUMMARY_LENGTH = 3

def summarize(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Create a list of stop words
    stop_words = set(stopwords.words('english'))

    # Count the frequency of each word
    word_freq = {}
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            if word.lower() not in stop_words:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

    # Calculate the score of each sentence based on the frequency of its words
    sentence_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence)
        score = 0
        for word in words:
            if word.lower() not in stop_words:
                score += word_freq[word]
        sentence_scores[sentence] = score

    # Select the top N sentences with the highest score
    summary_sentences = nlargest(MAX_SUMMARY_LENGTH, sentence_scores, key=sentence_scores.get)

    # Combine the summary sentences into a single summary string
    summary = ' '.join(summary_sentences)

    return summary
