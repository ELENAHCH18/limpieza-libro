import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

with open("libro.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

# 1. Normalización
texto = texto.lower()
texto = re.sub(r"[^a-zA-Z\s]", " ", texto)

# 2. Tokenización
tokens = word_tokenize(texto)

# 3. Eliminar stopwords
stop_words = set(stopwords.words("english"))
tokens_limpios = [palabra for palabra in tokens if palabra not in stop_words and len(palabra) > 1]

# 4. Lematización
lemmatizer = WordNetLemmatizer()
tokens_lematizados = [lemmatizer.lemmatize(palabra) for palabra in tokens_limpios]

# 5. Texto limpio final
texto_limpio = " ".join(tokens_lematizados)

with open("libro_limpio.txt", "w", encoding="utf-8") as salida:
    salida.write(texto_limpio)

# Dividir en fragmentos para vectorizar
fragmentos = texto_limpio.split(".")
fragmentos = [frag.strip() for frag in fragmentos if frag.strip()]

if len(fragmentos) == 0:
    fragmentos = [texto_limpio]

# 6. Bag of Words
bow_vectorizer = CountVectorizer(max_features=20)
X_bow = bow_vectorizer.fit_transform(fragmentos)

# 7. TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=20)
X_tfidf = tfidf_vectorizer.fit_transform(fragmentos)

print("Proceso completado")
print("Tokens originales:", len(tokens))
print("Tokens limpios:", len(tokens_limpios))
print("Tokens lematizados:", len(tokens_lematizados))
print("Archivo generado: libro_limpio.txt")

print("\n=== BAG OF WORDS ===")
print("Dimensión:", X_bow.shape)
print("Vocabulario:", bow_vectorizer.get_feature_names_out())
print("Matriz BoW:")
print(X_bow.toarray())

print("\n=== TF-IDF ===")
print("Dimensión:", X_tfidf.shape)
print("Vocabulario:", tfidf_vectorizer.get_feature_names_out())
print("Matriz TF-IDF:")
print(X_tfidf.toarray())

