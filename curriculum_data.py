import os
import numpy as np
import faiss
from groq import Groq

# โหลด API Key จาก .env หรือใส่ตรงนี้ก็ได้
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "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"
client = Groq(api_key=GROQ_API_KEY)

# โหลดข้อมูลหลักสูตร
with open("data/curriculum.txt", "r", encoding="utf-8") as f:
    curriculum_text = f.read()

# แบ่งเนื้อหาเป็น chunks ง่าย ๆ
chunks = [chunk.strip() for chunk in curriculum_text.split("\n\n") if chunk.strip()]

def get_embedding_groq(text: str):
    """สร้าง embedding ด้วย Groq Embeddings"""
    try:
        resp = client.embeddings.create(
            model="groq/embeddings-3-small",  # เปลี่ยนเป็นโมเดล embeddings จริงของ Groq
            input=text
        )
        return resp['data'][0]['embedding']
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดขณะสร้าง embedding: {e}")
        # fallback เป็น vector ว่าง
        return np.zeros(1536).tolist()  # สมมติ dimension 1536

# สร้าง embeddings สำหรับทุก chunk
print("📌 กำลังสร้าง embeddings ...")
embeddings = [get_embedding_groq(chunk) for chunk in chunks]
print("✅ สร้าง embeddings เสร็จแล้ว")

# สร้าง FAISS index
def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

index = build_faiss_index(embeddings)
print("✅ สร้าง FAISS index เสร็จแล้ว")
