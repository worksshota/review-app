import os
import json
import sys
from pydantic import BaseModel, Field
from typing import List, Literal
from google import genai
from google.genai import types
from openai import OpenAI

# --- 1. スキーマ定義 ---
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

# --- 2. 実行メイン処理 ---
def run():
    print("=== パイプライン開始 ===")
   
    # 環境変数の読み込み確認
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not gemini_key:
        print("エラー: GEMINI_API_KEY が設定されていません。Secretsを確認してください。")
        sys.exit(1)
    if not openai_key:
        print("エラー: OPENAI_API_KEY が設定されていません。Secretsを確認してください。")
        sys.exit(1)

    print("APIキーの検出に成功しました。")

    product = "ワイヤレスイヤホン Model-X"
    sample_reviews = "【レビュー1】音質がクリアで満足。【レビュー2】長時間つけると耳が少し痛い。"

    # Step 1: Gemini による要約
    print("1. Gemini API 呼び出し中...")
    try:
        gemini_client = genai.Client(api_key=gemini_key)
        prompt = f"分析してください。\n# 商品名: {product}\n# レビュー:\n{sample_reviews}"
       
        res = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewSummarySchema,
                temperature=0.1,
            ),
        )
        print("Gemini 解析完了！")
        json_data = json.loads(res.text)
    except Exception as e:
        print(f"Gemini API エラーが発生しました: {e}")
        sys.exit(1)

    # Step 2: OpenAI による記事生成
    print("2. OpenAI API 呼び出し中...")
    try:
        openai_client = OpenAI(api_key=openai_key)
        article_prompt = f"レビューまとめ記事を作成してください。\nデータ: {json.dumps(json_data, ensure_ascii=False)}"
       
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたはプロのWebライターです。"},
                {"role": "user", "content": article_prompt}
            ]
        )
        print("OpenAI 記事生成完了！")
        print("\n=== 生成結果 (先頭200文字) ===")
        print(response.choices[0].message.content[:200])
        print("\n=== 全工程成功 ===")
    except Exception as e:
        print(f"OpenAI API エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
