import streamlit as st
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np

st.title("Task Two - Measurement of document similarity using cosine coefficient and TF-IDF")

# Make Three Text Areas Input
st.header("Enter the text below")
st.subheader("Document 1")
document1 = st.text_area("Enter text here", height=100, key="text1", value="Perkembangan teknologi komunikasi pada zaman sekarang sangat pesat. Kita bisa melihat bagaimana teknologi ini telah mempengaruhi banyak aspek kehidupan manusia, termasuk dalam bidang bisnis, pendidikan, dan hiburan. Misalnya, dengan adanya internet dan smartphone, kita bisa dengan mudah mengakses informasi dari seluruh dunia, melakukan video conference dengan kolega di luar negeri, serta menonton film atau mendengarkan musik dari platform digital.")
st.subheader("Document 2")
document2 = st.text_area("Enter text here", height=100, key="text2", value="Salah satu teknologi komunikasi yang paling populer saat ini adalah media sosial. Media sosial seperti Facebook, Instagram, dan Twitter memungkinkan pengguna untuk terhubung dengan orang lain dari seluruh dunia, berbagi konten, dan memperluas jaringan sosial mereka. Namun, penggunaan media sosial juga membawa risiko, seperti penggunaan yang berlebihan atau cyberbullying. Oleh karena itu, penting untuk menggunakan teknologi komunikasi dengan bijak dan memahami dampaknya pada kesehatan mental dan sosial kita.")
st.subheader("Document 3")
document3 = st.text_area("Enter text here", height=100, key="text3", value="Teknologi komunikasi juga memainkan peran penting dalam pemerintahan dan politik. Dalam beberapa tahun terakhir, kita telah menyaksikan penggunaan teknologi seperti media sosial dan pesan instan dalam kampanye politik dan demonstrasi. Selain itu, teknologi komunikasi juga memungkinkan pemerintah untuk berkomunikasi dengan warga negara dan memberikan layanan publik secara lebih efektif. Namun, ada juga kekhawatiran bahwa teknologi ini dapat digunakan untuk menyebarkan propaganda dan informasi yang salah atau merusak privasi dan keamanan data.")
st.write("Total Char Document 1: ", len(document1), 'Total Word Document 1: ', len(document1.split(' ')), "Total Sentence Document 1: ", len(document1.split('.')))
st.write("Total Char Document 2: ", len(document2), 'Total Word Document 2: ', len(document2.split(' ')), "Total Sentence Document 2: ", len(document2.split('.')))
st.write("Total Char Document 3: ", len(document3), 'Total Word Document 3: ', len(document3.split(' ')), "Total Sentence Document 3: ", len(document3.split('.')))
pd.set_option('display.max_colwidth', None)

def get_information_char_word_sentence(document):
    st.write("Total Chart in Result Document 1: ", len(result[document][0]), 'Total Word in Result Document 1: ', len(result[document][0].split(' ')), "Total Sentence in Result Document 1: ", len(result[document][0].split('.')))
    st.write("Total Chart in Result Document 2: ", len(result[document][1]), 'Total Word in Result Document 2: ', len(result[document][1].split(' ')), "Total Sentence in Result Document 2: ", len(result[document][1].split('.')))
    st.write("Total Chart in Result Document 3: ", len(result[document][2]), 'Total Word in Result Document 3: ', len(result[document][2].split(' ')), "Total Sentence in Result Document 3: ", len(result[document][2].split('.')))
    # Make Total Char, Word, Sentence in Result All Document
    st.write("Total Chart in Result All Document: ", len(result[document][0]) + len(result[document][1]) + len(result[document][2]), 'Total Word in Result All Document: ', len(result[document][0].split(' ')) + len(result[document][1].split(' ')) + len(result[document][2].split(' ')), "Total Sentence in Result All Document: ", len(result[document][0].split('.')) + len(result[document][1].split('.')) + len(result[document][2].split('.')))
    
# buatkan saya sebuah fungsi yang menghitung jumlah dokumen yang mengandung setiap kata pada
doc_count = {}
query_count = {}
def count_word_in_document(unique_words, doc1, doc2, doc3):
    for word in unique_words:
        count = sum([1 for doc in [doc1, doc2, doc3] if word in doc])
        doc_count[word] = count

# Show the result in a pandas dataframe two columns (id, document)
st.header("Result")
result = pd.DataFrame({"document": [document1, document2, document3]})
st.write(result)

# Processing Text
# Step 1: Case Folding With add a new column in the dataframe
st.header("Step 1: Case Folding")
result = result.assign(document_lower=result['document'].str.lower())
st.write(result)


# Step 2: Remove Punctuation With add a new column in the dataframe
st.header("Step 2: Remove Punctuation")
result = result.assign(document_lower_no_punct=result['document_lower'].str.replace('[^\w\s]',''))
st.write(result)
get_information_char_word_sentence('document_lower_no_punct')

# Step 3: Remove Stopword With add a new column in the dataframe by File stopword.txt
st.header("Step 3: Remove Stopword")
stopword = pd.read_csv('stopword.txt', header=None)
## Tranpose the dataframe
st.subheader("Stopword List")
st.write(stopword.T)
st.write("Total Stopword: ", len(stopword))
st.subheader("Result after remove stopword")
## Convert Stopword to list
stopword = stopword[0].tolist() 
result = result.assign(document_lower_no_punct_no_stopword=result['document_lower_no_punct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopword)])))
st.write(result)
get_information_char_word_sentence('document_lower_no_punct_no_stopword')

# Step 4: Stemming With add a new column in the dataframe by Sastrawi
st.header("Step 4: Stemming")
stemmer = StemmerFactory().create_stemmer()
result = result.assign(document_lower_no_punct_no_stopword_stem=result['document_lower_no_punct_no_stopword'].apply(lambda x: stemmer.stem(x)))
st.write(result)
get_information_char_word_sentence('document_lower_no_punct_no_stopword_stem')

# Step 5: Weighting With add a new column in the dataframe by TF-IDF
st.header("Step 5: Weighting")
## Make Frequency Word in Result All Document With Two Column (Word, Frequency)
st.subheader("Frequency Word in Result All Document")
frequency_word = pd.DataFrame(result['document_lower_no_punct_no_stopword_stem'].str.split(expand=True).stack().value_counts()).reset_index()
frequency_word.columns = ['word', 'frequency']
frequency_word['tf'] = frequency_word['frequency'] / frequency_word['frequency'].sum()
st.write(frequency_word.T)
highest_frequency_word = frequency_word['word'][0]
total_frequency_word = frequency_word['frequency'][0]
st.write("Based on the frequency of the word, we can see that the word ",highest_frequency_word," is the most frequent word in the document with a frequency of ",total_frequency_word,". This means that the word ",highest_frequency_word," appears ",total_frequency_word," times in the all documents.")

## Make Dataframe With Column (Kata Kunci, Dokumen 1, Dokumen 2, Dokumen 3)
tf = pd.DataFrame(columns=['kata_kunci', 'dokumen_1', 'dokumen_2', 'dokumen_3'])
tf['kata_kunci'] = frequency_word['word']
tf['dokumen_1'] = tf['kata_kunci'].apply(lambda x: result['document_lower_no_punct_no_stopword_stem'][0].count(x))
tf['dokumen_2'] = tf['kata_kunci'].apply(lambda x: result['document_lower_no_punct_no_stopword_stem'][1].count(x))
tf['dokumen_3'] = tf['kata_kunci'].apply(lambda x: result['document_lower_no_punct_no_stopword_stem'][2].count(x))
st.subheader("TF - Term Frequency")
st.write(tf.T)

count_word_in_document(frequency_word['word'], result['document_lower_no_punct_no_stopword_stem'][0], result['document_lower_no_punct_no_stopword_stem'][1], result['document_lower_no_punct_no_stopword_stem'][2])

doc_count = pd.DataFrame(doc_count.items(), columns=['word', 'count'])
st.subheader("Number of documents containing the word")
st.write(doc_count.T)

# Make Dataframe With Column (Kata Kunci, Dokumen 1 TFIDF, Dokumen 2 TFIDF, Dokumen 3 TFIDF)
tfidf = pd.DataFrame(columns=['kata_kunci', 'dokumen_1', 'dokumen_2', 'dokumen_3'])
tfidf['kata_kunci'] = frequency_word['word']
# in column dokumen 1 TFIDF = TF dokumen 1 * np.log(len(result) / number of documents containing the word)
tfidf['dokumen_1'] = tf['dokumen_1'] * (np.log(len(result) / doc_count['count']))
tfidf['dokumen_2'] = tf['dokumen_2'] * (np.log(len(result) / doc_count['count']))
tfidf['dokumen_3'] = tf['dokumen_3'] * (np.log(len(result) / doc_count['count']))
st.subheader("TFIDF - Term Frequency Inverse Document Frequency")
st.write(tfidf.T)

# Step 6: Testing With Cosine Coefficient
## Make 3 Text Area for input query

def case_folding(query):
    return query.lower()

def remove_punctuation(query):
    return query.replace('[^\w\s]','')

def remove_stopword(query):
    return ' '.join([word for word in query.split() if word not in (stopword)])

def stemming(query):
    return stemmer.stem(query)

def count_word_in_query(unique_words, query1, query2, query3):
    for word in unique_words:
        count = sum([1 for query in [query1, query2, query3] if word in query])
        query_count[word] = count


st.header("Step 6: Testing With Cosine Coefficient")
query1 = st.text_area("Query 1", key="query1", value="Perkembangan Teknologi pada beberapa negera")
query2 = st.text_area("Query 2", key="query2", value="Kemajuan Teknologi Dengan Internet di indonesia")
query3 = st.text_area("Query 3", key="query3", value="Politik Sosial dan Ekonomi")

st.subheader("Query Original")
result = pd.DataFrame([query1, query2, query3], columns=['query_original'])
st.write(result)

st.subheader("Result Query - Case Folding")
result = result.assign(query_lower=result['query_original'].apply(lambda x: case_folding(x)))
st.write(result)

st.subheader("Result Query - Remove Punctuation")
result = result.assign(query_no_punct=result['query_lower'].apply(lambda x: remove_punctuation(x)))
st.write(result)

st.subheader("Result Query - Remove Stopword")
result = result.assign(query_no_punct_no_stopword=result['query_no_punct'].apply(lambda x: remove_stopword(x)))
st.write(result)

st.subheader("Result Query - Stemming")
result = result.assign(query_no_punct_no_stopword_stem=result['query_no_punct_no_stopword'].apply(lambda x: stemming(x)))
st.write(result)

st.subheader("Result Query - Frequency Word By All Documents")
# Make A Pandas Dataframe With Column (Kata Kunci)
result_freq = pd.DataFrame(columns=['kata_kunci', 'query 1 count'])
result_freq['kata_kunci'] = frequency_word['word']
# in column query 1 count = count the word in query 1 
result_freq['query 1 count'] = result_freq['kata_kunci'].apply(lambda x: result['query_no_punct_no_stopword_stem'][0].count(x))
result_freq['query 2 count'] = result_freq['kata_kunci'].apply(lambda x: result['query_no_punct_no_stopword_stem'][1].count(x))
result_freq['query 3 count'] = result_freq['kata_kunci'].apply(lambda x: result['query_no_punct_no_stopword_stem'][2].count(x))
st.write(result_freq.T)

count_word_in_query(frequency_word['word'], result['query_no_punct_no_stopword_stem'][0], result['query_no_punct_no_stopword_stem'][1], result['query_no_punct_no_stopword_stem'][2])
query_count = pd.DataFrame(query_count.items(), columns=['word', 'count'])
st.subheader("Number of queries containing the word")
st.write(query_count.T)

tfidf_query = pd.DataFrame(columns=['kata_kunci', 'query 1', 'query 2', 'query 3'])
tfidf_query['kata_kunci'] = frequency_word['word']
tfidf_query['query 1'] = result_freq['query 1 count'] * (np.log(len(result) / query_count['count']))
tfidf_query['query 2'] = result_freq['query 2 count'] * (np.log(len(result) / query_count['count']))
tfidf_query['query 3'] = result_freq['query 3 count'] * (np.log(len(result) / query_count['count']))

# change nan value to 0 because nan value is caused by 0 count in query in tfidf_query
tfidf_query = tfidf_query.fillna(0)
st.subheader("TFIDF Query")
st.write(tfidf_query.T)



def cosine_similarity(query, document):
    numerator = sum(np.dot(query, document) for query, document in zip(query, document))
    a = []
    for i in query:
        a.append(i**2)
    a = np.sqrt(sum(a))
    b = []
    for i in document:
        b.append(i**2)
    b = np.sqrt(sum(b))
    
    return numerator / (a * b)

# Calculate Cosine Coefficient For Query 1 and All Documents
# formula is cosine_similarity(q, d) = (q * d) / (||q|| * ||d||)
# i want to calculate cosine_similarity(q1, d1), cosine_similarity(q1, d2), cosine_similarity(q1, d3)
st.subheader("Cosine Coefficient For Query 1 and All Documents")
st.write("Query 1 : ", result['query_no_punct_no_stopword_stem'][0])
st.write("Similarity With All Documents")
for i in range(3):
    st.write("Cosine Coefficient With Document ", i+1, " : ", cosine_similarity(tfidf_query['query 1'], tfidf['dokumen_'+str(i+1)]))

st.subheader("Cosine Coefficient For Query 2 and All Documents")
st.write("Query 2 : ", result['query_no_punct_no_stopword_stem'][1])
st.write("Similarity With All Documents")
for i in range(3):
    st.write("Cosine Coefficient With Document ", i+1, " : ", cosine_similarity(tfidf_query['query 2'], tfidf['dokumen_'+str(i+1)]))
    
st.subheader("Cosine Coefficient For Query 3 and All Documents")
st.write("Query 3 : ", result['query_no_punct_no_stopword_stem'][2])
st.write("Similarity With All Documents")
for i in range(3):
    st.write("Cosine Coefficient With Document ", i+1, " : ", cosine_similarity(tfidf_query['query 3'], tfidf['dokumen_'+str(i+1)]))

    
    