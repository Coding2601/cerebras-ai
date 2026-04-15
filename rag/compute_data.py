from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from gita.views import getVerse, getEngSloka, getSansSloka, getName, no_of_verses

all_embeddings = []
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

total = sum(no_of_verses(c) for c in range(1, 19))

with tqdm(total=total, desc="Generating embeddings") as pbar:
    
    for chapter in range(1, 19):
        chapter_name = getName(chapter)
        total_verses = no_of_verses(chapter)

        for verse_no in range(1, total_verses + 1):

            eng_sloka = getEngSloka(chapter, verse_no)
            sans_sloka = getSansSloka(chapter, verse_no)
            meaning = getVerse(chapter, verse_no)

            text_for_embedding = f"""
            Chapter: {chapter}
            Chapter Name: {chapter_name}
            Verse: {verse_no}

            Sanskrit:
            {sans_sloka}

            English Sloka:
            {eng_sloka}

            Meaning:
            {meaning}
            """

            vector = embedder.get_embedding(text_for_embedding)

            all_embeddings.append({
                "chapter_no": chapter,
                "verse_no": verse_no,
                "chapter_name": chapter_name,
                "verse": meaning,
                "eng_sloka": eng_sloka,
                "embedding": vector.tolist()
            })

            pbar.update(1)

with open("resources/gita_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(all_embeddings, f)

print("✅ All embeddings generated successfully")