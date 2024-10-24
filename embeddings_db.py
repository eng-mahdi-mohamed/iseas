import json

DATABASE_PATH = "faces_embeddings_db.json"

# حفظ المتجهات في قاعدة بيانات JSON
def save_embedding_to_db(id, user, embeddings):
    try:
        with open(DATABASE_PATH, 'r') as file:
            faces = json.load(file)
    except FileNotFoundError:
        faces = []

    faces[id] = {'embeddings': embeddings, 'user': user}

    with open(DATABASE_PATH, 'w') as file:
        json.dump(faces, file)

# تحميل المتجهات المخزنة مسبقاً
def load_embeddings():
    try:
        with open(DATABASE_PATH, 'r') as file:
            faces = json.load(file)
            return faces
    except FileNotFoundError:
        return []
