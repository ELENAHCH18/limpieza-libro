import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

with open("libro.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

texto = texto.lower()
texto = re.sub(r"[^a-zA-Z\s]", " ", texto)

tokens = word_tokenize(texto)

stop_words = set(stopwords.words("english"))
tokens_limpios = [palabra for palabra in tokens if palabra not in stop_words and len(palabra) > 1]

lemmatizer = WordNetLemmatizer()
tokens_lematizados = [lemmatizer.lemmatize(palabra) for palabra in tokens_limpios]

texto_limpio = " ".join(tokens_lematizados)

with open("libro_limpio.txt", "w", encoding="utf-8") as salida:
    salida.write(texto_limpio)

print("Proceso completado")
print("Tokens originales:", len(tokens))
print("Tokens limpios:", len(tokens_limpios))
print("Tokens lematizados:", len(tokens_lematizados))
print("Archivo generado: libro_limpio.txt")
