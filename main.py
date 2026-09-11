import os
import re
import json
import sys
from typing import List, Literal
from pydantic import BaseModel
from google import genai
from google.genai import types
from serpapi import GoogleSearch

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

# --- 2. Webからの口コミ自動収集（SerpAPI） ---
def fetch_web_reviews(product_name: str, serpapi_key: str) -> List[str]:
    print(f"Webから「{product_name}」の口コミ・レビューを検索中...")
    params = {
        "q": f"{product_name} レビュー 口コミ 感想 評判",
        "hl": "ja",
        "gl": "jp",
        "api_key": serpapi_key
    }
   
    search = GoogleSearch(params)
    results = search.get_dict()
   
    reviews = []
    # 検索結果のスニペット（概要文）を収集
    organic_results = results.get("organic_results", [])
    for res in organic_results:
        snippet = res.get("snippet", "")
        if len(snippet) > 20:
            reviews.append(snippet)
           
    print(f"収集されたWeb口コミ数: {len(reviews)}件")
    return reviews

# --- 3. 前処理機能 ---
def preprocess_reviews(reviews: List[str]) -> List[str]:
    cleaned, seen = [], set()
    for r in reviews:
        text = re.sub(r'\s+', ' ', r).strip()
        text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]', '', text)
        if len(text) >= 10 and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned

# --- 4. Gemini による分析・マージ機能 ---
def analyze_and_merge(client: genai.Client, reviews: List[str], product_name: str) -> dict:
    cleaned = preprocess_reviews(reviews)
    text_data = "\n".join(cleaned)
   
    prompt = f"以下のWeb口コミデータを分析・整理してください。\n# 商品名: {product_name}\n# 口コミデータ:\n{text_data}"
   
    res = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReviewSummarySchema,
            temperature=0.1,
        ),
    )
    return json.loads(res.text)

# --- 5. Gemini による記事生成機能 ---
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

# --- 6. メイン実行処理 ---
def main():
    print("=== 全自動レビュー記事作成システム開始 ===")
   
    g_key = os.environ.get("GEMINI_API_KEY")
    s_key = os.environ.get("SERPAPI_API_KEY")
   
    if not g_key:
        print("エラー: GEMINI_API_KEY が未設定です。")
        sys.exit(1)
    if not s_key:
        print("エラー: SERPAPI_API_KEY が未設定です。")
        sys.exit(1)

    client = genai.Client(api_key=g_key)
   
    # 検索・レビュー収集したい商品名を設定
    product = "AirPods Pro 第2世代"
   
    # 1. 自動収集
    web_reviews = fetch_web_reviews(product, s_key)
    if not web_reviews:
        print("口コミデータが取得できませんでした。処理を停止します。")
        sys.exit(1)

    # 2. 口コミの分析と構造化
    print("\n1. 口コミ解析・構造化処理中...")
    json_data = analyze_and_merge(client, web_reviews, product)
    print("解析完了!")

    # 3. 記事出力
    print("2. 記事本文を生成中...")
    article = generate_article_with_gemini(client, json_data)
    print("記事生成完了!\n")
   
    print("================== 生成された記事 ==================")
    print(article)
    print("====================================================")
    print("=== すべての工程が正常完了しました ===")

if __name__ == "__main__":
    main()
