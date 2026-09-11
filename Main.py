import os
import re
import json
import requests
import markdown
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from openai import OpenAI

# --- 1. データ構造定義 ---
class ProItem(BaseModel):
    point: str
    detail: str
    frequency: Literal["高", "中", "低"]

class ConItem(BaseModel):
    point: str
    detail: str
    frequency: Literal["高", "中", "低"]

class TargetAudience(BaseModel):
    recommended_for: List[str]
    not_recommended_for: List[str]

class ReviewSummarySchema(BaseModel):
    product_name: str
    overall_summary: str
    pros: List[ProItem]
    cons: List[ConItem]
    target_audience: TargetAudience

# --- 2. 前処理 ---
def preprocess_reviews(reviews: List[str]) -> List[str]:
    cleaned, seen = [], set()
    for r in reviews:
        text = re.sub(r'\s+', ' ', r).strip()
        text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]', '', text)
        if len(text) >= 10 and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned

def chunk_text_list(texts: List[str], max_chars: int = 2000) -> List[str]:
    chunks, current_chunk, current_length = [], [], 0
    for text in texts:
        if current_length + len(text) > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk, current_length = [], 0
        current_chunk.append(text)
        current_length += len(text)
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

# --- 3. AI処理 ---
def analyze_and_merge(reviews: List[str], product_name: str) -> dict:
    gemini_client = genai.Client()
    cleaned = preprocess_reviews(reviews)
    chunks = chunk_text_list(cleaned)
   
    chunk_jsons = []
    for chunk in chunks:
        prompt = f"分析してください。\n# 商品名: {product_name}\n# レビュー:\n{chunk}"
        res = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewSummarySchema,
                temperature=0.1,
            ),
        )
        chunk_jsons.append(json.loads(res.text))
   
    merge_prompt = f"統合してください。\n# 商品名: {product_name}\n# データ:\n{json.dumps(chunk_jsons, ensure_ascii=False)}"
    merged_res = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=merge_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReviewSummarySchema,
            temperature=0.2,
        ),
    )
    return json.loads(merged_res.text)

def generate_article(structured_json: dict) -> str:
    openai_client = OpenAI()
    prompt = f"記事を作成してください。\n# 構造化データ:\n{json.dumps(structured_json, ensure_ascii=False)}"
    res = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたはプロのWebライターです。"},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content

# --- 4. WordPress下書き保存 ---
def post_to_wordpress(domain: str, user: str, app_pass: str, title: str, md_content: str):
    url = f"https://{domain}/wp-json/wp/v2/posts"
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    payload = {"title": title, "content": html_content, "status": "draft"}
    requests.post(url, json=payload, auth=(user, app_pass))

# --- 実行処理 ---
if __name__ == "__main__":
    product = "ワイヤレスイヤホン Model-X"
    sample_reviews = [
        "音質が非常にクリアでボーカルの伸びが素晴らしいです。低音も効きます。",
        "長時間つけると右耳が痛くなりました。付属のピースを替えてもイマイチ。",
        "デザインは高級感があって最高！ケースもスリムでポケットに入りやすい。",
        "ノイズキャンセリング機能は強力ではないです。電車内の音は聞こえます。"
    ]
   
    json_data = analyze_and_merge(sample_reviews, product)
    article = generate_article(json_data)
   
    # WordPress情報がある場合は投稿（環境変数から取得）
    wp_domain = os.environ.get("WP_DOMAIN")
    wp_user = os.environ.get("WP_USER")
    wp_pass = os.environ.get("WP_APP_PASS")
   
    if wp_domain and wp_user and wp_pass:
        post_to_wordpress(wp_domain, wp_user, wp_pass, f"【レビュー】{product}", article)
    else:
        print("--- 生成記事内容 ---")
        print(article)
