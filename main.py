import os
import re
import json
import sys
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

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

# --- 2. 前処理機能 ---
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

# --- 3. Gemini による分析・マージ機能 ---
def analyze_and_merge(client: genai.Client, reviews: List[str], product_name: str) -> dict:
    cleaned = preprocess_reviews(reviews)
    chunks = chunk_text_list(cleaned)
   
    chunk_jsons = []
    for chunk in chunks:
        prompt = f"分析してください。\n# 商品名: {product_name}\n# レビュー:\n{chunk}"
        res = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewSummarySchema,
                temperature=0.1,
            ),
        )
        chunk_jsons.append(json.loads(res.text))
   
    merge_prompt = f"重複を整理して統合してください。\n# 商品名: {product_name}\n# データ:\n{json.dumps(chunk_jsons, ensure_ascii=False)}"
    merged_res = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=merge_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReviewSummarySchema,
            temperature=0.2,
        ),
    )
    return json.loads(merged_res.text)

# --- 4. Gemini による最終記事生成機能 ---
def generate_article_with_gemini(client: genai.Client, structured_json: dict) -> str:
    prompt = f"""
あなたはプロのWebライターです。
以下の【構造化データ】をもとに、読者の購買決定に役立つ客観的でわかりやすいレビューまとめ記事を作成してください。

# 記事作成ルール
- Markdown形式（H2, H3の見出し）で出力してください。
- 構成: 1.概要 2.メリット 3.デメリット 4.おすすめな人・向かない人 5.まとめ

# 構造化データ:
{json.dumps(structured_json, ensure_ascii=False, indent=2)}
"""
    res = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
    )
    return res.text

# --- 5. メイン実行処理 ---
def main():
    print("=== レビュー自動生成処理（Gemini単体構成）開始 ===")
   
    g_key = os.environ.get("GEMINI_API_KEY")
    if not g_key:
        print("エラー: GEMINI_API_KEY が取得できませんでした。")
        sys.exit(1)

    client = genai.Client(api_key=g_key)
    product = "ワイヤレスイヤホン Model-X"
   
    sample_reviews = [
        "音質が非常にクリアでボーカルの伸びが素晴らしいです。低音もズッシリ効きます。",
        "長時間つけると右耳が痛くなりました。付属のイヤピースを替えてもイマイチ。",
        "デザインは高級感があって最高！ケースもスリムでポケットに入りやすい。",
        "ノイズキャンセリング機能は期待ほど強力ではないです。電車内の音は聞こえます。",
        "音質最高！バッテリーも公称通りかなり持ちます。通勤用にはピッタリ。"
    ]

    print("1. レビュー解析・構造化処理中...")
    json_data = analyze_and_merge(client, sample_reviews, product)
    print("解析完了!")

    print("2. 記事本文を生成中...")
    article = generate_article_with_gemini(client, json_data)
    print("記事生成完了!\n")
   
    print("================== 生成された記事 ==================")
    print(article)
    print("====================================================")
    print("=== すべての工程が正常完了しました ===")

if __name__ == "__main__":
    main()
