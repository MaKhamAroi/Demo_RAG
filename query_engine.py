from curriculum_data import chunks, embeddings, index, get_embedding_groq
import faiss
import numpy as np

# ค้นหา chunk ที่ใกล้เคียงกับ query
def search_chunks(query, top_k=2):
    query_emb = get_embedding_groq(query)
    D, I = index.search(np.array([query_emb], dtype='float32'), top_k)
    results = [chunks[i] for i in I[0]]
    return results

# ตอบคำถามโดยสรุปเฉพาะข้อมูลที่เกี่ยวข้อง
def answer_query(query):
    relevant_chunks = search_chunks(query)
    # รวมข้อความจาก chunk ที่ใกล้เคียง
    context_text = "\n".join(relevant_chunks)

    # Prompt สำหรับ LLM ให้ตอบสั้น ๆ
    prompt = f"""
    คุณคือตัวช่วยตอบคำถามเกี่ยวกับหลักสูตร
    โปรดตอบคำถามด้านล่าง โดย **ตอบสั้น ๆ ตรงคำถาม** จากข้อมูลที่ให้
    ข้อมูลหลักสูตร:
    {context_text}

    คำถาม: {query}
    กรุณาตอบสั้น ๆ
    """

    from groq import Groq
    import os

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    client = Groq(api_key=GROQ_API_KEY)

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = completion.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {e}"
