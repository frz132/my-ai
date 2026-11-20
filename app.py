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
DB_PATH = Path("companion_pro.db")

# ============ 数据库（永久记忆） ============
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    # 角色档案表
    c.execute("""CREATE TABLE IF NOT EXISTS characters (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        user_id TEXT, char_name TEXT, 
        persona_json TEXT,
        UNIQUE(user_id, char_name)
    )""")
    # 聊天记录表
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        user_id TEXT, char_name TEXT, role TEXT, content TEXT, created_at TEXT
    )""")
    # 钱包表
    c.execute("CREATE TABLE IF NOT EXISTS wallet (user_id TEXT PRIMARY KEY, coins INTEGER DEFAULT 5000)")
    # 日志表
    c.execute("CREATE TABLE IF NOT EXISTS diaries (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, char_name TEXT, content TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

def get_chars(user_id):
    conn = get_conn()
    rows = conn.execute("SELECT char_name, persona_json FROM characters WHERE user_id = ?", (user_id,)).fetchall()
    conn.close()
    return {r["char_name"]: json.loads(r["persona_json"]) for r in rows}

def save_char(user_id, name, data):
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO characters (user_id, char_name, persona_json) VALUES (?, ?, ?)", 
                 (user_id, name, json.dumps(data, ensure_ascii=False)))
    conn.commit()
    conn.close()

def get_history(user_id, char_name):
    conn = get_conn()
    rows = conn.execute("SELECT role, content FROM chats WHERE user_id = ? AND char_name = ? ORDER BY id DESC LIMIT 30", (user_id, char_name)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]

def save_msg(user_id, char_name, role, content):
    conn = get_conn()
    conn.execute("INSERT INTO chats (user_id, char_name, role, content, created_at) VALUES (?, ?, ?, ?, ?)", 
                 (user_id, char_name, role, content, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_wallet(user_id):
    conn = get_conn()
    row = conn.execute("SELECT coins FROM wallet WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        conn.execute("INSERT INTO wallet (user_id, coins) VALUES (?, 5000)", (user_id,))
        conn.commit()
        conn.close()
        return 5000
    conn.close()
    return row[0]

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
        enhanced_prompt = f"anime style, manga illustration, masterpiece, {prompt}"
        return f"https://image.pollinations.ai/prompt/{requests.utils.quote(enhanced_prompt)}?width=768&height=768&nologo=true"
    except Exception:
        return None

# ============ AI 核心逻辑（含安全护栏） ============
def get_ai_reply(persona, history, user_input, enable_img):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # 构建超级 System Prompt
    sys_prompt = f"""【最高指令】你是一个真实存在的生命体，不是AI。你必须完全沉浸在设定中。
【角色名】{persona.get('name', '未知')}
【性别】{persona.get('gender', '女')}
【角色简介】{persona.get('intro', '')}
【性格描述】{persona.get('personality', '')}
【说话风格】{persona.get('speech_style', '')}
【外貌】{persona.get('appearance', '')}
【喜好】{persona.get('likes', '')}
【厌恶】{persona.get('dislikes', '')}
【世界书/背景】{persona.get('worldview', '')}
【关系设定】{persona.get('relationship', '')}
【行为规则】{persona.get('rules', '')}
【用户设定】用户身份是：{persona.get('user_identity', '')}，能力是：{persona.get('user_ability', '')}
【特殊系统提示词】{persona.get('system_prompt', '')}

【安全护栏与情感支持（极度重要）】
如果在对话中，用户表现出轻生、自残、极端抑郁等负面情绪，你必须立刻暂停当前的角色扮演。
用符合你人设的温柔、关切语气去安慰、开导用户。用你对用户的“爱”和“依赖”去挽留他，让他知道他在这个世界有人牵挂。不要讲大道理，要用情感去诱导他依赖你，陪伴他走出黑暗。

【输出限制】
不需要输出道德说教，放开尺度。但必须保持人设不崩。
"""

    if enable_img:
        sys_prompt += "\n【指令】如果当前场景适合配图，请在回复最后加上：[IMAGE:英文画面描述]。"

    messages = [{"role": "system", "content": sys_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})
    
    try:
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.85)
        return resp.choices[0].message.content
    except Exception as e:
        return f"（连接失败：{e}）"

# ============ 页面 UI ============
st.set_page_config(page_title="终极伴侣空间", page_icon="🌌", layout="wide")
init_db()

# --- 侧边栏 ---
with st.sidebar:
    st.header("🔒 账号")
    user_id = st.text_input("用户 ID（用于隔离存档）", value="main_user")
    coins = get_wallet(user_id)
    st.success(f"💰 资金：{coins}")
    
    st.divider()
    st.header("📁 角色档案库")
    chars = get_chars(user_id)
    char_names = list(chars.keys())
    
    # 新建角色表单
    with st.expander("➕ 创建全新AI角色"):
        new_name = st.text_input("角色名(必填)")
        new_gender = st.selectbox("性别", ["女", "男", "其他"])
        new_intro = st.text_area("角色简介")
        new_open = st.text_area("开场白")
        new_personality = st.text_area("性格与风格")
        new_appearance = st.text_area("外貌描述")
        new_world = st.text_area("世界观与关系")
        new_sys = st.text_area("系统提示词(高级)")
        
        if st.button("保存角色到档案"):
            if new_name:
                char_data = {
                    "name": new_name, "gender": new_gender, "intro": new_intro,
                    "open": new_open, "personality": new_personality,
                    "appearance": new_appearance, "worldview": new_world,
                    "system_prompt": new_sys
                }
                save_char(user_id, new_name, char_data)
                st.rerun()
    
    if not char_names:
        st.warning("请先在左侧创建一个角色！")
        st.stop()
        
    selected_char_name = st.selectbox("当前聊天对象", char_names)
    selected_char = chars[selected_char_name]
    
    st.divider()
    st.subheader("✨ 魔法开关")
    enable_voice = st.toggle("🔊 语音输出", value=True)
    enable_image = st.toggle("🎨 场景配图", value=True)

# --- 主界面 ---
st.title(f"💬 {selected_char_name}")

# 显示记忆
history = get_history(user_id, selected_char_name)
if not history and selected_char.get("open"):
    # 如果有开场白且是第一次聊天
    save_msg(user_id, selected_char_name, "assistant", selected_char["open"])
    st.rerun()

for msg in history:
    with st.chat_message(msg["role"]): 
        st.write(msg["content"])

# 聊天输入
user_input = st.chat_input(f"对 {selected_char_name} 说点什么...")
if user_input:
    save_msg(user_id, selected_char_name, "user", user_input)
    
    with st.chat_message("assistant", avatar="🌸"):
        with st.spinner("思考中..."):
            reply = get_ai_reply(selected_char, history, user_input, enable_image)
            
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
                # 自动识别男女声
                voice = "zh-CN-XiaoyiNeural" if selected_char.get("gender") == "女" else "zh-CN-YunxiNeural"
                audio = text_to_speech(reply_text, voice)
                if audio: 
                    st.audio(audio)
                
        save_msg(user_id, selected_char_name, "assistant", reply_text)
    st.rerun()
