import os
import sys

def main():
    print("=== デバッグ処理開始 ===")
   
    # 1. 環境変数の存在チェック
    g_key = os.environ.get("GEMINI_API_KEY")
    o_key = os.environ.get("OPENAI_API_KEY")
   
    print(f"GEMINI_API_KEY の存在: {'あり' if g_key else 'なし (未設定)'}")
    print(f"OPENAI_API_KEY の存在: {'あり' if o_key else 'なし (未設定)'}")
   
    if not g_key or not o_key:
        print("\n[エラー原因] GitHubのSecretsにAPIキーが正しく登録されていません。")
        sys.exit(1)
       
    # 2. ライブラリ読み込みチェック
    try:
        from google import genai
        print("google-genai ライブラリ: 読み込み成功")
    except Exception as e:
        print(f"google-genai 読み込み失敗: {e}")
        sys.exit(1)
       
    try:
        from openai import OpenAI
        print("openai ライブラリ: 読み込み成功")
    except Exception as e:
        print(f"openai 読み込み失敗: {e}")
        sys.exit(1)

    # 3. Gemini 接続テスト (モデル名を gemini-3.6-flash に更新)
    print("\n--- Gemini API 接続テスト ---")
    try:
        client = genai.Client(api_key=g_key)
        res = client.models.generate_content(
            model='gemini-3.6-flash',
            contents='Hello',
        )
        print("Gemini API 接続成功！")
    except Exception as e:
        print(f"[Geminiエラー詳細]: {e}")
        sys.exit(1)

    # 4. OpenAI 接続テスト
    print("\n--- OpenAI API 接続テスト ---")
    try:
        client = OpenAI(api_key=o_key)
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("OpenAI API 接続成功！")
    except Exception as e:
        print(f"[OpenAIエラー詳細]: {e}")
        sys.exit(1)

    print("\n=== すべてのテストに合格しました！ ===")

if __name__ == "__main__":
    main()
