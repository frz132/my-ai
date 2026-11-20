import os
import json
import time
import sqlite3
import asyncio
import requests
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI
import edge_tts
from dotenv import load_dotenv

load_dotenv()

# ============ 基础配置 ============
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

DB_PATH = Path("companion.db")

# ==========================================
# 🔥 军师魔改版：赛博商店物品库（性感+布料少）
# ==========================================
SHOP_ITEMS = {
    "蕾丝透视吊带裙": {
        "price": 300, 
        "desc": "极度轻薄，半透明的材质让肌肤若隐若现，裙摆短到极限。", 
        "prompt": "wearing a very sheer lace slip dress, see-through, highly revealing, short skirt, seductive"
    },
    "黑色捆绑式情趣内衣": {
        "price": 500, 
        "desc": "由几根黑色绑带和极少的布料组成，完美凸显了她的傲人身材。", 
        "prompt": "wearing black strappy lingerie, highly revealing, showing cleavage, sexy seductive pose"
    },
    "微型比基尼": {
        "price": 400, 
        "desc": "布料少得可怜，仅仅遮住了关键部位，让人移不开视线。", 
        "prompt": "wearing a micro bikini, tiny string bikini, very revealing, blushing"
    },
    "开襟真丝睡袍": {
        "price": 600, 
        "desc": "松松垮垮地穿在身上，腰带半解，大片白皙的肌肤裸露在外。", 
        "prompt": "wearing an open silk robe, loose belt, revealing body, alluring"
    },
    "半透明白衬衫": {
        "price": 200, 
        "desc": "只穿了一件男款白衬衫，被水打湿后紧紧贴在身上，扣子解开了三颗。", 
        "prompt": "wearing an oversized white shirt, wet shirt, unbuttoned, see-through, no pants"
    }
}

DEFAULT_CHARACTERS = {
    "小暖": {"persona": "你是小暖，用户的温柔伴侣。极度顺从，穿着性感时会害羞但依然渴望展示。绝不承认是AI。", "voice": "zh-CN-XiaoxiaoNeural", "avatar": "🌸"},
    "星尘": {"persona": "你是星尘，高冷傲娇的二次元少女。虽然表面抗拒，但穿上性感衣服后会脸红心跳。绝不承认是AI。", "voice": "zh-CN-XiaoyiNeural", "avatar": "🌌"}
}

# ============ 数据库 ============
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, char_name TEXT, role TEXT, content TEXT, created_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS wallet (user_id TEXT PRIMARY KEY, coins INTEGER DEFAULT 3000)")
    c.execute("CREATE TABLE IF NOT EXISTS inventory (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, char_name TEXT, item_name TEXT)")
    conn.commit()
    conn.close()

def get_wallet(user_id):
    conn = get_conn()
    row = conn.execute("SELECT coins FROM wallet WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        conn.execute("INSERT INTO wallet (user_id, coins) VALUES (?, 3000)", (user_id,))
        conn.commit()
        conn.close()
        return 3000
    conn.close()
    return row[0]

def update_wallet(user_id, amount):
    conn = get_conn()
    conn.execute("UPDATE wallet SET coins = coins + ? WHERE user_id = ?", (amount, user_id))
    conn.commit()
    conn.close()

def buy_item(user_id, char_name, item_name, price):
    if get_wallet(user_id) >= price:
        update_wallet(user_id, -price)
        conn = get_conn()
        conn.execute("INSERT INTO inventory (user_id, char_name, item_name) VALUES (?, ?, ?)", (user_id, char_name, item_name))
        conn.commit()
        conn.close()
        return True
    return False

def get_worn_items(user_id, char_name):
    conn = get_conn()
    rows = conn.execute("SELECT item_name FROM inventory WHERE user_id = ? AND char_name = ?", (user_id, char_name)).fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_history(user_id, char_name):
    conn = get_conn()
    rows = conn.execute("SELECT role, content FROM chats WHERE user_id = ? AND char_name = ? ORDER BY id DESC LIMIT 15", (user_id, char_name)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]

def save_msg(user_id, char_name, role, content):
    conn = get_conn()
    conn.execute("INSERT INTO chats (user_id, char_name, role, content, created_at) VALUES (?, ?, ?, ?, ?)", (user_id, char_name, role, content, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# ============ 语音与生图 ============
def text_to_speech(text, voice_name):
    try:
        communicate = edge_tts.Communicate(text, voice_name)
        audio_file = f"voice_{int(time.time())}.mp3"
        asyncio.run(communicate.save(audio_file))
        return audio_file
    except Exception:
        return None

def generate_image(prompt):
    try:
        # 强化生图提示词，确保画面高质且无审查
        enhanced_prompt = f"masterpiece, best quality, anime illustration, beautiful girl, {prompt}, blushing, seductive"
        return f"https://image.pollinations.ai/prompt/{requests.utils.quote(enhanced_prompt)}?width=512&height=768&nologo=true&enhance=true"
    except Exception:
        return None

# ============ AI 核心 ============
def get_ai_reply(char_name, char_info, history, user_input, worn_items, enable_img):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    sys_prompt = f"{char_info['persona']}\n\n"
    
    if worn_items:
        sys_prompt += f"【当前穿戴】：你现在正穿着【{'、'.join(worn_items)}】。请注意，这些衣服布料很少、非常性感。你必须详细描述你现在穿着这身衣服的娇羞感、身体曲线的暴露，并主动用这身衣服去诱惑/勾引用户。\n"
    
    if enable_img:
        sys_prompt += "【指令】：如果对话适合配图，请在回复末尾加上：[IMAGE:英文描述]。例如：[IMAGE:wearing black strappy lingerie, lying on bed]。必须用英文。"
    
    messages = [{"role": "system", "content": sys_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})
    
    try:
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.9)
        return resp.choices[0].message.content
    except Exception as e:
        return f"（连接失败：{e}）"

# ============ 页面 UI ============
st.set_page_config(page_title="AI 伴侣", page_icon="🌌", layout="centered")
init_db()

if not API_KEY:
    st.error("请先在 Secrets 中配置 LLM_API_KEY。")
    st.stop()

with st.sidebar:
    st.header("🔒 账号")
    user_id = st.text_input("用户 ID（两台手机请填不同）", value="main_user")
    
    st.divider()
    st.subheader("💳 赛博钱包")
    coins = get_wallet(user_id)
    st.success(f"💰 当前金币：{coins}")
    
    st.divider()
    st.subheader("🏪 情趣衣橱")
    selected_char = st.selectbox("选择角色", list(DEFAULT_CHARACTERS.keys()))
    
    for item_name, item_data in SHOP_ITEMS.items():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**{item_name}**")
            st.caption(f"{item_data['price']} 金币 - {item_data['desc']}")
        with col2:
            if st.button("购买", key=f"buy_{item_name}"):
                if buy_item(user_id, selected_char, item_name, item_data['price']):
                    st.success("购买成功！")
                    st.rerun()
                else:
                    st.error("金币不足！")
                    
    st.divider()
    st.subheader("✨ 魔法开关")
    enable_voice = st.toggle("🔊 语音输出", value=True)
    enable_image = st.toggle("🎨 漫画生图 (极力推荐)", value=True)

# ============ 主界面 ============
char_info = DEFAULT_CHARACTERS[selected_char]
st.title(f"{char_info['avatar']} 与 {selected_char} 聊天")

worn_items = get_worn_items(user_id, selected_char)
if worn_items:
    st.warning(f"🔥 她目前穿戴：{'、'.join(worn_items)}")

history = get_history(user_id, selected_char)
for msg in history:
    with st.chat_message(msg["role"]): 
        st.write(msg["content"])

user_input = st.chat_input("发消息...")
if user_input:
    save_msg(user_id, selected_char, "user", user_input)
    
    with st.chat_message("assistant", avatar=char_info["avatar"]):
        with st.spinner("思考中..."):
            reply = get_ai_reply(selected_char, char_info, history, user_input, worn_items, enable_image)
            
            image_url = None
            if "[IMAGE:" in reply and enable_image:
                try:
                    desc = reply.split("[IMAGE:")[1].split("]")[0].strip()
                    reply_text = reply.split("[IMAGE:")[0].strip()
                    image_url = generate_image(desc)
                except Exception: 
                    reply_text = reply
            else:
                reply_text = reply
                
            st.write(reply_text)
            if image_url:
                st.image(image_url, use_container_width=True)
            
            if enable_voice:
                audio = text_to_speech(reply_text, char_info["voice"])
                if audio: 
                    st.audio(audio)
                
        save_msg(user_id, selected_char, "assistant", reply_text)
    st.rerun()
