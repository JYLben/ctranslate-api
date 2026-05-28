import os
import re
import sys
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import ctranslate2

app = Flask(__name__)

# 使用 ctranslate2 的动态下载和转换（对低内存服务器极其友好）
# 这里我们直接从 Hugging Face 引用别人已经转换好的轻量化（int8 量化）模型，内存仅需 ~100MB
ZH_EN_MODEL = "ctranslate2/opus-mt-zh-en"
EN_ZH_MODEL = "ctranslate2/opus-mt-en-zh"

print("🔄 正在初始化超轻量推理引擎...")
try:
    # 提前载入分词器
    tokenizer_zh_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    tokenizer_en_zh = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    
    # 载入轻量化翻译器（自动下载）
    translator_zh_en = ctranslate2.Translator(ZH_EN_MODEL, device="cpu")
    translator_en_zh = ctranslate2.Translator(EN_ZH_MODEL, device="cpu")
    print("✨ CTranslate2 轻量化模型加载成功！内存占用极低。")
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