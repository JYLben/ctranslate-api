import os
import re
import sys
import gc
from flask import Flask, request, jsonify
from transformers import MarianTokenizer
import ctranslate2

app = Flask(__name__)

# 压制不必要的警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 获取当前文件所在的项目根目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZH_EN_PATH = os.path.join(BASE_DIR, "model_ct2_zh_en")
EN_ZH_PATH = os.path.join(BASE_DIR, "model_ct2_en_zh")

print("🔄 正在初始化超轻量 CTranslate2 本地推理引擎...")
try:
    # 1. 显式加载配套的官方专用 Marian 分词器
    print("📋 正在加载分词器...")
    tokenizer_zh_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    tokenizer_en_zh = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    
    # 2. 直接从项目内置文件夹中载入已经量化好的轻量推理引擎
    # 💡 关键改动：强制限制单线程（inter_threads=1, intra_threads=1），防止多线程抢占内存与CPU
    print("📦 正在直接加载内置轻量化模型...")
    translator_zh_en = ctranslate2.Translator(
        ZH_EN_PATH, 
        device="cpu", 
        inter_threads=1, 
        intra_threads=1
    )
    translator_en_zh = ctranslate2.Translator(
        EN_ZH_PATH, 
        device="cpu", 
        inter_threads=1, 
        intra_threads=1
    )
    
    print("🚀 恭喜！本地单线程模型全部直接加载成功，已完美在 512MB 限制内上线。")
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
            
            # 翻译完立刻释放临时内存
            del source, results, target_tokens
            gc.collect()
            
            return jsonify({"translation": translation, "detected_lang": "en", "target_lang": "zh"})
        else:
            # 中译英
            source = tokenizer_zh_en.convert_ids_to_tokens(tokenizer_zh_en.encode(text))
            results = translator_zh_en.translate_batch([source])
            target_tokens = results[0].hypotheses[0]
            translation = tokenizer_zh_en.decode(tokenizer_zh_en.convert_tokens_to_ids(target_tokens))
            
            # 翻译完立刻释放临时内存
            del source, results, target_tokens
            gc.collect()
            
            return jsonify({"translation": translation, "detected_lang": "zh", "target_lang": "en"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "OK", 200

if __name__ == '__main__':
    # 绑定 Render 要求的环境变量端口
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)