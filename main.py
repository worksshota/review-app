import os
import re
import json
import sys
import time
import requests
import markdown
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

# --- 2. キーワードリスト管理 ---
def get_next_product(file_path="products.txt") -> str:
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines[0] if lines else None

def remove_processed_product(file_path="products.txt"):
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[1:]) + "\n" if len(lines) > 1 else "")

# --- 3. 口コミ自動収集（SerpAPI） ---
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
    for res in results.get("organic_results", []):
        snippet = res.get("snippet", "")
        if len(snippet) > 20:
            reviews.append(snippet)
    print(f"収集件数: {len(reviews)}件")
    return reviews

def preprocess_reviews(reviews: List[str]) -> List[str]:
    cleaned, seen = [], set()
    for r in reviews:
        text = re.sub(r'\s+', ' ', r).strip()
        text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]', '', text)
        if len(text) >= 10 and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned

# --- 4. Gemini 呼び出し ---
def call_gemini_with_retry(client, model, prompt, config=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            if config:
                return client.models.generate_content(model=model, contents=prompt, config=config)
            return client.models.generate_content(model=model, contents=prompt)
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"503検出。{attempt + 1}/{max_retries} 回目の再試行を行います...")
                time.sleep(5)
            else:
                raise e
    raise Exception("再試行上限に達しました。")

def analyze_and_merge(client: genai.Client, reviews: List[str], product_name: str) -> dict:
    cleaned = preprocess_reviews(reviews)
    text_data = "\n".join(cleaned)
    prompt = f"以下のWeb口コミデータを分析・整理してください。\n# 商品名: {product_name}\n# 口コミデータ:\n{text_data}"
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ReviewSummarySchema,
        temperature=0.1,
    )
    res = call_gemini_with_retry(client, 'gemini-3.6-flash', prompt, config)
    return json.loads(res.text)

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
    res = call_gemini_with_retry(client, 'gemini-3.6-flash', prompt)
    return res.text

# --- 5. はてなブログ自動下書き投稿（AtomPub） ---
def post_to_hatena(hatena_id: str, blog_id: str, api_key: str, title: str, md_content: str):
    url = f"https://blog.hatena.ne.jp/{hatena_id}/{blog_id}/atom/entry"
   
    # はてなブログ用XMLフォーマットの作成（下書き保存: draft=yes）
    xml_payload = f"""<?xml version="1.0" encoding="utf-8"?>
<entry xmlns="http://www.w3.org/2005/Atom"
       xmlns:app="http://www.w3.org/2007/app">
  <title>{title}</title>
  <content type="text/plain">{md_content}</content>
  <app:control>
    <app:draft>yes</app:draft>
  </app:control>
</entry>
"""
    headers = {'Content-Type': 'application/xml'}
    response = requests.post(
        url,
        data=xml_payload.encode('utf-8'),
        auth=(hatena_id, api_key),
        headers=headers
    )
   
    if response.status_code == 201:
        print("はてなブログへ下書き投稿が完了しました！")
    else:
        print(f"はてなブログ投稿失敗: {response.status_code} - {response.text}")

# --- 6. メイン実行処理 ---
def main():
    print("=== 全自動レビュー記事作成＆はてなブログ投稿システム開始 ===")
   
    g_key = os.environ.get("GEMINI_API_KEY")
    s_key = os.environ.get("SERPAPI_API_KEY")
    hatena_id = os.environ.get("HATENA_ID")
    hatena_blog_id = os.environ.get("HATENA_BLOG_ID")
    hatena_api_key = os.environ.get("HATENA_API_KEY")
   
    if not g_key or not s_key:
        print("エラー: 必須のAPIキーが未設定です。")
        sys.exit(1)

    product = get_next_product()
    if not product:
        print("products.txt に対象商品がありません。処理を終了します。")
        sys.exit(0)

    print(f"【本日処理対象】: {product}")
    client = genai.Client(api_key=g_key)
   
    # 1. 収集
    web_reviews = fetch_web_reviews(product, s_key)
    if not web_reviews:
        print("口コミ取得失敗のためスキップします。")
        sys.exit(1)

    # 2. 解析
    print("\n1. 口コミ解析中...")
    json_data = analyze_and_merge(client, web_reviews, product)

    # 3. 記事生成
    print("2. 記事生成中...")
    article = generate_article_with_gemini(client, json_data)
    print("記事生成完了!")

    # 4. はてなブログへの投稿
    if hatena_id and hatena_blog_id and hatena_api_key:
        print("3. はてなブログへ下書き投稿中...")
        post_to_hatena(hatena_id, hatena_blog_id, hatena_api_key, f"【口コミ評判】{product}のメリット・デメリットまとめ", article)
    else:
        print("※はてなブログ設定が空のためログ出力のみ行いました。")

    # 5. リスト更新
    remove_processed_product()
    print(f"「{product}」を処理済みとしてリスト更新完了。")
    print("=== すべての工程が正常終了しました ===")

if __name__ == "__main__":
    main()
