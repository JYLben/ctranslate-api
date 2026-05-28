import os
import re
import sys
import shutil
from flask import Flask, request, jsonify
from transformers import MarianTokenizer
import ctranslate2

app = Flask(__name__)

# 压制不必要的警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 定义模型本地存储和转换路径
ZH_EN_DIR = "model_ct2_zh_en"
EN_ZH_DIR = "model_ct2_en_zh"

print("🔄 正在初始化超轻量 CTranslate2 推理引擎...")

def get_or_convert_model(repo_id, output_dir):
    """如果本地没有转换后的模型，则自动下载官方模型并转换为 CT2 int8 格式"""
    if not os.path.exists(output_dir):
        print(f"📦 正在下载官方 {repo_id} 并自动转换为超轻量量化格式...")
        # 临时转换目录
        tmp_dir = output_dir + "_tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            
        try:
            converter = ctranslate2.converters.TransformersConverter(
                repo_id,
                activation_scales=None,
                load_as_float16=False
            )
            # 使用 int8 量化，将内存和体积压缩 4 倍，完美适配 512MB 内存
            converter.convert(tmp_dir, quantization="int8", force=True)
            os.rename(tmp_dir, output_dir)
            print(f"✨ 模型 {repo_id} 转换成功！")
        except Exception as e:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            raise e
    return output_dir

try:
    # 1. 自动安全转换官方源模型（完全不依赖外部私有仓库，绝对不会找不到）
    zh_en_path = get_or_convert_model("Helsinki-NLP/opus-mt-zh-en", ZH_EN_DIR)
    en_zh_path = get_or_convert_model("Helsinki-NLP/opus-mt-en-zh", EN_ZH_DIR)
    
    # 2. 显式加载配套的官方专用 Marian 分词器
    print("📋 正在加载分词器...")
    tokenizer_zh_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    tokenizer_en_zh = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    
    # 3. 载入已经转换好的本地轻量推理引擎
    translator_zh_en = ctranslate2.Translator(zh_en_path, device="cpu")
    translator_en_zh = ctranslate2.Translator(en_zh_path, device="cpu")
    
    print("🚀 恭喜！超轻量 AI 翻译引擎已就绪，完美绕过 512MB 内存限制。")
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