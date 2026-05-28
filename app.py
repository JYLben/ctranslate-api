import os
import re
import sys
from flask import Flask, request, jsonify
# 🌟 关键修改：显式导入 MarianTokenizer，不再依赖自动类
from transformers import MarianTokenizer
import ctranslate2
import huggingface_hub

app = Flask(__name__)

# 压制不必要的警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("🔄 正在初始化超轻量 CTranslate2 推理引擎...")
try:
    # 1. 下载 CTranslate2 轻量化 int8 模型（仅 ~100MB 左右，极省内存）
    model_zh_en_path = huggingface_hub.snapshot_download(repo_id="PolyAI/opus-mt-zh-en-ct2-int8")
    model_en_zh_path = huggingface_hub.snapshot_download(repo_id="PolyAI/opus-mt-en-zh-ct2-int8")
    
    # 2. 🌟 显式加载配套的专用 Marian 分词器
    tokenizer_zh_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    tokenizer_en_zh = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    
    # 3. 载入轻量化翻译引擎
    translator_zh_en = ctranslate2.Translator(model_zh_en_path, device="cpu")
    translator_en_zh = ctranslate2.Translator(model_en_zh_path, device="cpu")
    
    print("✨ CTranslate2 轻量化模型与分词器全部加载成功！")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    sys.exit(1)

def is_english(text):
    letters = len(re.findall(r'[a-zA-Z]', text))
    if len(text) == 0: return False
    return (letters / len(text)) > 0.3

@app.route('/v1/translate', methods=['POST'])
def translate_api():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400
        
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"translation": "", "detected_lang": "unknown"})
        
    try:
        if is_english(text):
            # 英译中
            source = tokenizer_en_zh.convert_ids_to_tokens(tokenizer_en_zh.encode(text))
            results = translator_en_zh.translate_batch([source])
            target_tokens = results[0].hypotheses[0]
            translation = tokenizer_en_zh.decode(tokenizer_en_zh.convert_tokens_to_ids(target_tokens))
            return jsonify({"translation": translation, "detected_lang": "en", "target_lang": "zh"})
        else:
            # 中译英
            source = tokenizer_zh_en.convert_ids_to_tokens(tokenizer_zh_en.encode(text))
            results = translator_zh_en.translate_batch([source])
            target_tokens = results[0].hypotheses[0]
            translation = tokenizer_zh_en.decode(tokenizer_zh_en.convert_tokens_to_ids(target_tokens))
            return jsonify({"translation": translation, "detected_lang": "zh", "target_lang": "en"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)