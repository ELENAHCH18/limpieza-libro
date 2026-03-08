import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gensim.models import Word2Vec

# Descargas NLTK
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# 1. Carga del libro
with open("libro.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

# 2. Limpieza / normalización
texto = texto.lower()
texto = re.sub(r"[^a-zA-Z\s\.]", " ", texto)
texto = re.sub(r"\s+", " ", texto).strip()

# 3. Tokenización por oraciones
oraciones = sent_tokenize(texto)

# 4. Stopwords y lematización
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

corpus_limpio = []
tokens_totales = []

for oracion in oraciones:
    palabras = word_tokenize(oracion)
    palabras = [p for p in palabras if p.isalpha()]
    palabras = [p for p in palabras if p not in stop_words]
    palabras = [lemmatizer.lemmatize(p) for p in palabras]
    
    if palabras:
        corpus_limpio.append(palabras)
        tokens_totales.extend(palabras)

texto_limpio = " ".join(tokens_totales)

# Guardar texto limpio
with open("libro_limpio.txt", "w", encoding="utf-8") as salida:
    salida.write(texto_limpio)

# 5. Vectorización clásica
documentos = [" ".join(oracion) for oracion in corpus_limpio[:500]]

bow_vectorizer = CountVectorizer(max_features=30)
X_bow = bow_vectorizer.fit_transform(documentos)

tfidf_vectorizer = TfidfVectorizer(max_features=30)
X_tfidf = tfidf_vectorizer.fit_transform(documentos)

print("=== LIMPIEZA ===")
print("Oraciones procesadas:", len(corpus_limpio))
print("Tokens finales:", len(tokens_totales))
print("Archivo generado: libro_limpio.txt")

print("\n=== BAG OF WORDS ===")
print("Dimensión:", X_bow.shape)
print("Vocabulario:", bow_vectorizer.get_feature_names_out())

print("\n=== TF-IDF ===")
print("Dimensión:", X_tfidf.shape)
print("Vocabulario:", tfidf_vectorizer.get_feature_names_out())

# 6. Word2Vec = semántica distribucional
modelo_w2v = Word2Vec(
    sentences=corpus_limpio,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1
)

print("\n=== WORD2VEC ===")
print("Tamaño del vocabulario:", len(modelo_w2v.wv.index_to_key))
print("Palabras más similares a 'alice' (si existe):")
if "alice" in modelo_w2v.wv:
    print(modelo_w2v.wv.most_similar("alice", topn=10))
else:
    print("La palabra 'alice' no está en el vocabulario.")

# 7. Selección de palabras para visualización
palabras = modelo_w2v.wv.index_to_key[:60]
vectores = np.array([modelo_w2v.wv[palabra] for palabra in palabras])

# 8. PCA
pca = PCA(n_components=2)
vectores_pca = pca.fit_transform(vectores)

plt.figure(figsize=(14, 10))
for i, palabra in enumerate(palabras):
    x, y = vectores_pca[i]
    plt.scatter(x, y, color="blue")
    plt.text(x + 0.01, y + 0.01, palabra, fontsize=9)

plt.title("Espacio vectorial Word2Vec - PCA")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("espacio_word2vec_pca.png", dpi=300)
plt.close()

# 9. t-SNE
tsne = TSNE(n_components=2, perplexity=10, random_state=42, init="random", learning_rate="auto")
vectores_tsne = tsne.fit_transform(vectores)

plt.figure(figsize=(14, 10))
for i, palabra in enumerate(palabras):
    x, y = vectores_tsne[i]
    plt.scatter(x, y, color="green")
    plt.text(x + 0.5, y + 0.5, palabra, fontsize=9)

plt.title("Espacio vectorial Word2Vec - t-SNE")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("espacio_word2vec_tsne.png", dpi=300)
plt.close()

print("\n=== IMÁGENES GENERADAS ===")
print("espacio_word2vec_pca.png")
print("espacio_word2vec_tsne.png")

