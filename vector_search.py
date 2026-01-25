from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Этот принт должен появиться сразу
print("⏳ Загружаю модель (первый раз это займет время)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
documents = [
    "Коты любят спать на солнце",
    "Собаки обожают гулять в парке",
    "Программисты пьют много кофе",
    "Python - отличный язык для AI"
]

query = "напиток для разработчика"

# Превращаем текст в цифры
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# Считаем сходство
scores = cosine_similarity(query_embedding, doc_embeddings)

print(f"\n🔍 Запрос: '{query}'\n")

# scores берет первый ряд оценок (для нашего единственного запроса)
results = sorted(zip(scores[0], documents), reverse=True) # Добавили [0]
for score, text in results:
    print(f"Оценка: {score:.4f} | Текст: {text}")
