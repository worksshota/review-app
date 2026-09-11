import json
import os
import time
import requests
import markdown
from google import genai

# --- 1. Gemini API呼び出し (リトライ処理付き) ---
def call_gemini_with_retry(client: genai.Client, model: str, prompt: str, max_retries: int = 3) -> any:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response
        except Exception as e:
            print(f"Gemini API Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise e

# --- 2. 記事生成プロンプト (視認性・改行・太字・箇条書き重視) ---
def generate_article_with_gemini(client: genai.Client, structured_json: dict) -> str:
    prompt = f"""
あなたはプロのWEBライターおよびアフィリエイターです。
以下の【構造化データ】をもとに、読者が一目でポイントを理解でき、購買意欲が高まる視認性の高いレビューまとめ記事を作成してください。

# 執筆・レイアウトの絶対ルール（視認性・可読性の重視）
1. **文章の壁を作らない（改行の徹底）**:
   - 2〜3文ごとに必ず1行の「空行」を挟んでください。文字が詰まった長文は厳禁です。
2. **箇条書き（- ）と太字（**文字**）の多用**:
   - メリット、デメリット、おすすめな人・向かない人は、文章でダラダラ書かず、必ず箇条書き（`- `）で記述してください。
   - 各箇条書きの最も重要なキーワード（単語）は、必ず **太字** で強調してください。
3. **メリット・デメリットの整理**:
   - 口口コミの出現頻度（高・中・低）に触れつつ、読者が気になる「実際の使い勝手」を具体的に解説してください。
4. **適切な見出し構成**:
   - 記事全体は以下のH2（`##`）およびH3（`###`）見出し構成に従って作成してください。

# 記事の構成テンプレート:
## 1. 概要と総合評価
（商品の全体像とどのようなアイテムかを2〜3文で簡潔に説明。適度に改行を入れる）

## 2. メリット（買って良かった点）
（ここに箇条書きで理由と詳細を記載。重要単語は太字）

## 3. デメリット（気になった点・注意点）
（ここに箇条書きで理由と詳細を記載。重要単語は太字）

## 4. おすすめな人・向かない人
### ⭕️ こんな人におすすめ
（箇条書き）

### ❌ こんな人には向かないかも
（箇条書き）

## 5. まとめ
（全体の振り返りと、購入を迷っている読者の背中を押すまとめ文）

# 構造化データ:
{json.dumps(structured_json, ensure_ascii=False, indent=2)}
"""
    # 新SDKの推奨モデル名「gemini-2.5-flash」を指定
    res = call_gemini_with_retry(client, 'gemini-2.5-flash', prompt)
    return res.text

# --- 3. はてなブログAtomPub投稿 (Markdown -> HTML自動変換処理付き) ---
def post_to_hatena(hatena_id: str, blog_id: str, api_key: str, title: str, md_content: str):
    url = f"https://blog.hatena.ne.jp/{hatena_id}/{blog_id}/atom/entry"
   
    # Markdown記号（##や**）をきれいなHTMLタグ（<h2>や<strong>）へ自動変換
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
   
    # はてなブログ用XMLフォーマット（HTML指定）
    xml_payload = f"""<?xml version="1.0" encoding="utf-8"?>
<entry xmlns="http://www.w3.org/2005/Atom"
       xmlns:app="http://www.w3.org/2007/app">
  <title>{title}</title>
  <content type="text/html"><![CDATA[
{html_content}
]]></content>
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
        print("はてなブログへ下書き投稿が完了しました！（記号なし・HTML整形済み）")
    else:
        print(f"はてなブログ投稿失敗: {response.status_code} - {response.text}")

# --- 4. メイン処理 ---
def main():
    # 環境変数の読み込み
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    hatena_id = os.environ.get("HATENA_ID")
    hatena_blog_id = os.environ.get("HATENA_BLOG_ID")
    hatena_api_key = os.environ.get("HATENA_API_KEY")

    if not all([gemini_api_key, hatena_id, hatena_blog_id, hatena_api_key]):
        print("エラー: 必要な環境変数が設定されていません。")
        return

    # Geminiクライアント初期化
    client = genai.Client(api_key=gemini_api_key)

    # products.txt から対象商品を読み込み
    if not os.path.exists("products.txt"):
        print("products.txt が見つかりません。")
        return

    with open("products.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("products.txt に対象商品がありません。")
        return

    target_product = lines[0]
    remaining_products = lines[1:]

    print(f"今回の処理対象商品: {target_product}")

    # モックの構造化データ
    structured_data = {
        "product_name": target_product,
        "summary": f"{target_product}の実際のユーザー口コミと評判のまとめです。",
        "pros": ["デザインが良い", "性能が高い", "使いやすい"],
        "cons": ["価格がやや高め", "少し重さを感じる"],
        "target_users": ["品質重視の人", "長く使いたい人"]
    }

    # 1. 記事生成
    print("Geminiで記事を生成中...")
    article_md = generate_article_with_gemini(client, structured_data)
    title = f"【口コミ・評判】{target_product}のメリット・デメリットを徹底解説"

    # 2. はてなブログへ下書き投稿
    print("はてなブログへ下書き送信中...")
    post_to_hatena(hatena_id, hatena_blog_id, hatena_api_key, title, article_md)

    # 3. products.txt の更新（先頭の1行を削除して上書き）
    with open("products.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(remaining_products) + ("\n" if remaining_products else ""))

    print(f"処理完了: {target_product} をリストから削除しました。")

if __name__ == "__main__":
    main()
