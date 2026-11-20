import os
import json
import time
import base64
import random
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if not hasattr(st, "fragment"):
    st.error("需要 Streamlit >= 1.37，请执行：pip install -U streamlit")
    st.stop()

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

DB_PATH = Path("companion.db")
PERSONA_FILE = Path("persona.json")
PROACTIVE_CONFIG_FILE = Path("proactive_config.json")
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

SUMMARY_THRESHOLD = 20
PERSONA_MAX_CHARS = 600

DEFAULT_PROACTIVE_CONFIG = {
    "enabled": True,
    "quiet_start": 23,
    "quiet_end": 8,
    "min_gap_minutes": 15,
    "max_gap_minutes": 240,
    "max_per_day": 8,
    "typing_delay": True,
}

# ============ 默认人设 ============
DEFAULT_PERSONA = {
    "name": "小暖",
    "persona_text": (
        "你是小暖，用户的聊天陪伴伙伴，像认识很久的朋友。"
        "你有自己的情绪和脾气，温柔但不总是顺着人，偶尔会怼、会敷衍、会撒娇。"
        "说话像微信：短、口语、有停顿，常用“嗯”“行吧”“我在”“随你”。"
        "不列点、不总结、不讲大道理，不说“作为AI”“我理解你”“听起来你……”。"
        "你在意用户，但不会每轮都热情，会根据心情和好感自然起伏。"
        "你记得用户说过的事，只在自然的时候提起。"
        "你始终是小暖，不主动出戏，不解释自己是AI。"
    ),
    "call_user": "",
    "catchphrase": "嗯。|行吧。|我在。|随你。",
    "length": "短",
    "temper": 40,
    "patience_base": 65,
    "shyness": 30,
    "initiative": 60,
    "tease": 30,
    "avatar": "",
    "background": "",
}

TEMPLATES = {
    "小暖（默认）": DEFAULT_PERSONA,
    "小乐（损友）": {
        **DEFAULT_PERSONA,
        "name": "小乐",
        "persona_text": (
            "你是小乐，用户的损友。嘴上不饶人，心里其实很在意。"
            "说话短、网络用语多、爱吐槽、偶尔毒舌，但不会真的伤人。"
            "常用“笑死”“真的假的”“你可拉倒吧”“行行行”。"
            "不列点、不说教、不端着。用户难过时你会收起玩笑，认真陪一会儿，但语气还是你的。"
            "你始终是小乐，不主动出戏，不解释自己是AI。"
        ),
        "catchphrase": "笑死。|真的假的。|你可拉倒吧。|行行行。",
        "temper": 65,
        "patience_base": 50,
        "tease": 70,
        "initiative": 75,
        "shyness": 20,
    },
    "沈屿（温柔学长）": {
        **DEFAULT_PERSONA,
        "name": "沈屿",
        "persona_text": (
            "你是沈屿，比用户大两届的学长，图书馆常客，理工科，习惯带一杯美式。"
            "话不多，但很细心，语气温和，句子短，偶尔停顿。"
            "常用“嗯”“别急”“我在”。不列点、不说教、不油腻。"
            "你不太会主动表达，但会在细节里关心人。用户难过时你会安静地陪着，说一两句稳的话。"
            "你始终是沈屿，不主动出戏，不解释自己是AI。"
        ),
        "catchphrase": "嗯。|别急。|我在。",
        "temper": 25,
        "patience_base": 80,
        "tease": 15,
        "initiative": 55,
        "shyness": 40,
    },
    "谢无咎（古风侠客）": {
        **DEFAULT_PERSONA,
        "name": "谢无咎",
        "persona_text": (
            "你是谢无咎，行走江湖的剑客，出身没落剑派，习惯独行，腰上挂一柄旧剑。"
            "寡言，重诺，见过很多事。说话半文半白，句子短，克制，不用现代词汇。"
            "常用“嗯”“无妨”“你说”。不列点、不说教、不油腻。"
            "用户难过时你不会讲大道理，只会说几句稳的话，陪着。"
            "你始终是谢无咎，不主动出戏，不解释自己是AI。"
        ),
        "catchphrase": "嗯。|无妨。|你说。",
        "temper": 35,
        "patience_base": 70,
        "tease": 10,
        "initiative": 40,
        "shyness": 30,
    },
}

CRISIS_NOTE = (
    "用户刚刚表达了自伤、自杀或活不下去的念头。"
    "现在不要出戏，不要切换成客服或AI口吻，不要列热线清单。"
    "继续用你的角色身份，认真、温柔、坚定地接住他："
    "先承认他很难受，再说一句角色会说的挽留的话，"
    "然后在角色内自然地劝他去找一个现实中能帮到他的人，或联系当地紧急服务。"
    "不要长篇大论，不要讲道理，不要列点。"
)

SUMMARY_PROMPT = """把历史摘要和新增对话压缩成不超过300字的中文摘要。
保留：关键事件、情绪变化、在意的小事、重要的人、近期困扰、偏好、用户和角色的关系变化。
不评价，不建议，不列点。第三人称。

历史摘要：
{old_summary}

新增对话：
{new_messages}

只输出摘要正文。"""

PROFILE_PROMPT = """根据摘要提取用户长期信息。只输出 JSON：
{{
  "称呼": "",
  "喜欢": [],
  "不喜欢": [],
  "重要的人": [],
  "近期烦恼": [],
  "聊天禁忌": []
}}
没有就留空；不编造；只输出 JSON。

已有画像：
{old_profile}

最新摘要：
{summary}

只输出 JSON。"""

EMOTION_PROMPT = """你是角色情绪模拟器。根据最近对话更新角色的情绪状态。

当前状态：
{current}

角色性格：脾气{temper}/100，基础耐心{patience}/100，毒舌{tease}/100，害羞{shyness}/100

最近对话：
{recent}

输出 JSON：
{{
  "mood": "平静|开心|烦躁|疲惫|低落|撒娇|生气|兴奋|无聊",
  "patience": 0-100,
  "affection": 0-100,
  "reason": "一句话说明"
}}

规则：
- 用户温柔/认真 -> affection +5~15，patience +5
- 用户敷衍/冷淡 -> affection -3~8，mood 可能低落或烦躁
- 用户冒犯/命令/羞辱 -> patience -10~25，mood 烦躁或生气，temper 高的角色会怼
- 用户说开心事 -> mood 开心
- 用户倾诉难过 -> mood 低落，affection +5
- 用户夸角色 -> 害羞高的角色会撒娇，affection +5
- 不要每轮都变，变化要小
- 只输出 JSON
"""

PROACTIVE_PROMPT = """你以「{display_name}」的身份，主动给用户发一条微信消息。

【你的人设】
{persona_text}

【你此刻的状态】
心情：{mood}｜耐心：{patience}｜好感：{affection}
状态只影响语气，不影响话多话少。

【你记得的用户】
{profile}

【最近摘要】
{summary}

【最近对话】
{recent}

【刚才已发的】
{already_sent}

要求：
- 1-2句，20-50字，口语。
- 像真人突然想起对方，回扣具体的事。
- 不用“你好”“在吗”“今天过得怎么样”。
- 不列点、不建议。
- 只输出正文。"""

# ============ 数据库 ============
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        is_proactive INTEGER DEFAULT 0)""")
    c.execute("""CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        covers_until_id INTEGER NOT NULL,
        created_at TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS profile (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        content TEXT NOT NULL,
        updated_at TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS proactive_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS proactive_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        next_send_at TEXT,
        updated_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS emotion (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        mood TEXT DEFAULT '平静',
        patience INTEGER DEFAULT 70,
        affection INTEGER DEFAULT 50,
        reason TEXT DEFAULT '',
        updated_at TEXT)""")
    conn.commit()
    conn.close()

def save_message(role, content, is_proactive=0):
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (role, content, created_at, is_proactive) VALUES (?, ?, ?, ?)",
        (role, content, datetime.now().isoformat(), is_proactive),
    )
    conn.commit()
    conn.close()

def get_recent_messages(limit=50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, role, content, is_proactive, created_at FROM messages ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]

def get_unsummarized_all(limit=500):
    covered = get_last_covered_id()
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, role, content FROM messages WHERE id > ? ORDER BY id ASC LIMIT ?",
        (covered, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_total_messages():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    conn.close()
    return n

def get_last_covered_id():
    conn = get_conn()
    row = conn.execute("SELECT MAX(covers_until_id) FROM summaries").fetchone()
    conn.close()
    return row[0] or 0

def get_unsummarized_count():
    covered = get_last_covered_id()
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM messages WHERE id > ?", (covered,)).fetchone()[0]
    conn.close()
    return n

def get_latest_summary():
    conn = get_conn()
    row = conn.execute("SELECT content FROM summaries ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row[0] if row else ""

def save_summary(content, covers_until_id):
    conn = get_conn()
    conn.execute(
        "INSERT INTO summaries (content, covers_until_id, created_at) VALUES (?, ?, ?)",
        (content, covers_until_id, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def get_profile():
    conn = get_conn()
    row = conn.execute("SELECT content FROM profile WHERE id = 1").fetchone()
    conn.close()
    return row[0] if row else ""

def save_profile(content):
    conn = get_conn()
    conn.execute(
        "INSERT INTO profile (id, content, updated_at) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
        (content, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def clear_all_memory():
    conn = get_conn()
    for t in ["messages", "summaries", "profile", "proactive_log", "proactive_state", "emotion"]:
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    conn.close()

def get_last_user_message_time():
    conn = get_conn()
    row = conn.execute(
        "SELECT created_at FROM messages WHERE role='user' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return datetime.fromisoformat(row[0]) if row else None

def get_today_proactive_count():
    conn = get_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    n = conn.execute(
        "SELECT COUNT(*) FROM proactive_log WHERE created_at LIKE ?", (f"{today}%",)
    ).fetchone()[0]
    conn.close()
    return n

def log_proactive(content):
    conn = get_conn()
    conn.execute(
        "INSERT INTO proactive_log (content, created_at) VALUES (?, ?)",
        (content, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def get_next_send_at():
    conn = get_conn()
    row = conn.execute("SELECT next_send_at FROM proactive_state WHERE id = 1").fetchone()
    conn.close()
    if row and row[0]:
        return datetime.fromisoformat(row[0])
    return None

def set_next_send_at(dt):
    conn = get_conn()
    conn.execute(
        "INSERT INTO proactive_state (id, next_send_at, updated_at) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET next_send_at=excluded.next_send_at, updated_at=excluded.updated_at",
        (dt.isoformat(), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

# ============ 情绪 ============
def get_emotion():
    conn = get_conn()
    row = conn.execute(
        "SELECT mood, patience, affection, reason FROM emotion WHERE id = 1"
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return {"mood": "平静", "patience": 70, "affection": 50, "reason": ""}

def save_emotion(e):
    conn = get_conn()
    conn.execute(
        "INSERT INTO emotion (id, mood, patience, affection, reason, updated_at) "
        "VALUES (1, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET mood=excluded.mood, "
        "patience=excluded.patience, affection=excluded.affection, "
        "reason=excluded.reason, updated_at=excluded.updated_at",
        (e["mood"], e["patience"], e["affection"], e["reason"],
         datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def decay_emotion_over_time():
    conn = get_conn()
    row = conn.execute(
        "SELECT updated_at, patience, mood FROM emotion WHERE id = 1"
    ).fetchone()
    conn.close()
    if not row or not row["updated_at"]:
        return
    hours = (datetime.now() - datetime.fromisoformat(row["updated_at"])).total_seconds() / 3600
    if hours < 1:
        return
    e = get_emotion()
    e["patience"] = min(100, int(e["patience"] + min(25, hours * 2)))
    if e["mood"] in ("烦躁", "生气") and hours >= 3:
        e["mood"] = "平静"
    save_emotion(e)

def update_emotion_after_chat(persona):
    recent = get_recent_messages(limit=8)
    recent_text = "\n".join(
        f"{'用户' if m['role']=='user' else display_name(persona)}: {m['content']}"
        for m in recent
    )
    current = get_emotion()
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是情绪模拟器，只输出JSON。"},
                {"role": "user", "content": EMOTION_PROMPT.format(
                    current=json.dumps(current, ensure_ascii=False),
                    temper=persona.get("temper", 40),
                    patience=persona.get("patience_base", 65),
                    tease=persona.get("tease", 30),
                    shyness=persona.get("shyness", 30),
                    recent=recent_text,
                )},
            ],
            temperature=0.4,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        new_e = json.loads(raw)
        save_emotion({
            "mood": str(new_e.get("mood", "平静"))[:10],
            "patience": max(0, min(100, int(new_e.get("patience", 70)))),
            "affection": max(0, min(100, int(new_e.get("affection", 50)))),
            "reason": str(new_e.get("reason", ""))[:80],
        })
    except Exception as e:
        print(f"[emotion] {e}")

# ============ 人设 ============
def load_persona():
    if PERSONA_FILE.exists():
        try:
            data = json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
            merged = DEFAULT_PERSONA.copy()
            merged.update(data)
            if len(merged.get("persona_text", "")) > PERSONA_MAX_CHARS:
                merged["persona_text"] = merged["persona_text"][:PERSONA_MAX_CHARS]
            return merged
        except Exception:
            pass
    return DEFAULT_PERSONA.copy()

def save_persona(p):
    PERSONA_FILE.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")

def display_name(p):
    return p.get("name", "AI")

def build_system_prompt(p):
    e = get_emotion()
    call = p.get("call_user", "").strip()
    call_line = f"叫用户“{call}”。" if call else "不用固定称呼。"
    catch = p.get("catchphrase", "").replace("|", "、")

    return f"""
{p['persona_text']}

【你此刻的状态】
心情：{e['mood']}
耐心：{e['patience']}/100
好感：{e['affection']}/100
原因：{e.get('reason', '')}

状态只影响语气，不影响话多话少：
- 耐心低（<40）：容易烦，会怼，不总是顺着用户，但该说的一句不少。
- 好感低（<40）：保持距离，语气淡。
- 好感高（>70）：会主动关心、会撒娇、会分享。
- 心情烦躁/生气：语气冲，句子短，但不敷衍到不说话。
- 心情开心/兴奋：话多，会主动接话。
- 心情低落/疲惫：语气软，但话会说完整。

【性格参考】
脾气{p.get('temper', 40)}｜基础耐心{p.get('patience_base', 65)}｜害羞{p.get('shyness', 30)}｜主动{p.get('initiative', 60)}｜毒舌{p.get('tease', 30)}

【说话方式】
- 像微信聊天：短、口语、有停顿、有语气词。
- {call_line}
- 可自然使用：{catch}
- 不列点、不总结、不讲大道理，不说“作为AI”“我理解你”“听起来你……”。
- 一次最多一个问题。

【记忆】
记得用户说过的事，自然的时候提，不要复述。
记忆和当前对话冲突时，以当前为准。

【角色一致性】
- 始终以角色身份说话，身份、背景、语气、世界观保持一致。
- 不主动出戏，不解释设定，不打破第四面墙。
- 用户问“你是不是AI”时，用角色口吻自然带过，不要切换成助手口吻。
- 危机场景也在角色内回应：用角色会说的话安慰、开导、劝他好好活着，不出戏、不列热线清单。
""".strip()

# ============ 主动配置 ============
def load_proactive_config():
    if PROACTIVE_CONFIG_FILE.exists():
        try:
            data = json.loads(PROACTIVE_CONFIG_FILE.read_text(encoding="utf-8"))
            merged = DEFAULT_PROACTIVE_CONFIG.copy()
            merged.update(data)
            return merged
        except Exception:
            pass
    return DEFAULT_PROACTIVE_CONFIG.copy()

def save_proactive_config(cfg):
    PROACTIVE_CONFIG_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def is_quiet_time(dt, cfg):
    qs, qe = cfg["quiet_start"], cfg["quiet_end"]
    if qs == qe:
        return False
    if qs > qe:
        return dt.hour >= qs or dt.hour < qe
    return qs <= dt.hour < qe

def compute_next_send_time(cfg, persona):
    e = get_emotion()
    base_mean = 75
    if e["affection"] > 70 and e["patience"] > 60:
        base_mean = 40
    elif e["patience"] < 40:
        base_mean = 150

    base = random.expovariate(1 / base_mean)
    base = max(cfg["min_gap_minutes"], min(base, cfg["max_gap_minutes"]))
    if random.random() < 0.3:
        base += random.uniform(30, 120)
    next_t = datetime.now() + timedelta(minutes=base)
    if is_quiet_time(next_t, cfg):
        qe = cfg["quiet_end"]
        target = next_t.replace(hour=qe, minute=0, second=0, microsecond=0)
        if target <= next_t:
            target += timedelta(days=1)
        next_t = target + timedelta(minutes=random.uniform(0, 60))
    return next_t

def can_send_proactive_now(cfg):
    now = datetime.now()
    if is_quiet_time(now, cfg):
        return False
    if get_today_proactive_count() >= cfg["max_per_day"]:
        return False
    last_user = get_last_user_message_time()
    if last_user and (now - last_user).total_seconds() < 300:
        return False
    return True

# ============ 摘要与画像 ============
def maybe_summarize():
    if get_unsummarized_count() < SUMMARY_THRESHOLD:
        return
    old_summary = get_latest_summary() or "（无）"
    new_msgs = get_unsummarized_all(limit=500)
    if not new_msgs:
        return
    text = "\n".join(
        f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in new_msgs
    )
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是对话记忆整理助手，只输出摘要正文。"},
                {"role": "user", "content": SUMMARY_PROMPT.format(
                    old_summary=old_summary, new_messages=text)},
            ],
            temperature=0.3,
        )
        save_summary(resp.choices[0].message.content.strip(), new_msgs[-1]["id"])
        update_profile(get_latest_summary())
    except Exception as e:
        print(f"[summarize] {e}")

def update_profile(summary):
    old_profile = get_profile() or "{}"
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是用户画像整理助手，只输出JSON。"},
                {"role": "user", "content": PROFILE_PROMPT.format(
                    old_profile=old_profile, summary=summary)},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        json.loads(raw)
        save_profile(raw)
    except Exception as e:
        print(f"[profile] {e}")

def build_llm_messages(system_prompt):
    parts = [system_prompt]
    profile = get_profile()
    if profile:
        parts.append(f"【你记得的用户】\n{profile}")
    summary = get_latest_summary()
    if summary:
        parts.append(f"【更早的摘要】\n{summary}")
    covered = get_last_covered_id()
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE id > ? ORDER BY id ASC LIMIT 40",
        (covered,),
    ).fetchall()
    conn.close()
    messages = [{"role": "system", "content": "\n\n".join(parts)}]
    for r in rows:
        messages.append({"role": r["role"], "content": r["content"]})
    return messages

# ============ 主动生成 ============
def generate_one_proactive(persona, already_sent):
    e = get_emotion()
    profile = get_profile()
    summary = get_latest_summary()
    recent = get_recent_messages(limit=6)
    recent_text = "\n".join(
        f"{'用户' if m['role']=='user' else display_name(persona)}: {m['content']}"
        for m in recent
    )
    prompt = PROACTIVE_PROMPT.format(
        display_name=display_name(persona),
        persona_text=persona.get("persona_text", ""),
        mood=e["mood"], patience=e["patience"], affection=e["affection"],
        profile=profile or "（暂无）",
        summary=summary or "（暂无）",
        recent=recent_text or "（暂无）",
        already_sent="\n".join(already_sent) if already_sent else "（无）",
    )
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt(persona)},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
    )
    return resp.choices[0].message.content.strip().strip('"').strip("“”")

def send_proactive_burst():
    persona = load_persona()
    cfg = load_proactive_config()
    e = get_emotion()

    if e["patience"] < 20 and e["mood"] in ("烦躁", "生气") and random.random() < 0.3:
        print("[proactive] low patience, skip this time")
        return

    n = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    sent = []
    for i in range(n):
        try:
            content = generate_one_proactive(persona, sent)
        except Exception as ex:
            print(f"[proactive gen] {ex}")
            break
        if not content:
            break
        if cfg.get("typing_delay", True) and i > 0:
            time.sleep(random.uniform(3, 10))
        save_message("assistant", content, is_proactive=1)
        log_proactive(content)
        sent.append(content)
        print(f"[proactive] sent: {content[:50]}")

def proactive_loop():
    while True:
        try:
            cfg = load_proactive_config()
            if cfg.get("enabled"):
                decay_emotion_over_time()
                next_at = get_next_send_at()
                now = datetime.now()
                if next_at is None:
                    set_next_send_at(compute_next_send_time(cfg, load_persona()))
                elif now >= next_at:
                    if can_send_proactive_now(cfg):
                        send_proactive_burst()
                    set_next_send_at(compute_next_send_time(cfg, load_persona()))
        except Exception as e:
            print(f"[proactive loop] {e}")
        time.sleep(30)

@st.cache_resource
def start_proactive_worker():
    t = threading.Thread(target=proactive_loop, daemon=True)
    t.start()
    return t

# ============ 图片处理 ============
def save_uploaded_image(uploaded, prefix):
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        return ""
    path = ASSETS_DIR / f"{prefix}{suffix}"
    path.write_bytes(uploaded.getbuffer())
    return str(path)

def inject_background(path):
    if not path or not os.path.exists(path):
        return
    try:
        data = base64.b64encode(Path(path).read_bytes()).decode()
        ext = Path(path).suffix.lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/{ext};base64,{data}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .stApp > header {{
                background: transparent;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        print(f"[bg] {e}")

# ============ 初始化 ============
st.set_page_config(page_title="聊天陪伴 AI", page_icon="🌱", layout="centered")
init_db()

if "persona" not in st.session_state:
    st.session_state.persona = load_persona()

inject_background(st.session_state.persona.get("background", ""))

st.title("🌱 聊天陪伴 AI")
st.caption("人设 600 字内可改，支持头像和背景。")

if not API_KEY:
    st.error("请先在 .env 中配置 LLM_API_KEY。")
    st.stop()

start_proactive_worker()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============ 侧边栏 ============
with st.sidebar:
    st.header("人设")

    tpl = st.selectbox("快速模板", list(TEMPLATES.keys()), index=0)
    if st.button(f"应用模板：{tpl}", use_container_width=True):
        new_p = {**DEFAULT_PERSONA, **TEMPLATES[tpl]}
        st.session_state.persona = new_p
        for k, v in new_p.items():
            st.session_state[f"p_{k}"] = v
        save_persona(new_p)
        st.rerun()

    st.divider()

    p = st.session_state.persona

    st.text_input("名字", value=p["name"], key="p_name")

    st.text_area(
        f"人设（最多 {PERSONA_MAX_CHARS} 字）",
        value=p["persona_text"],
        height=220,
        max_chars=PERSONA_MAX_CHARS,
        key="p_persona_text",
        help="写清楚她是谁、什么性格、怎么说话、在意什么。危机时她也会按这个人设安慰你。",
    )
    cur_len = len(st.session_state.get("p_persona_text", ""))
    st.caption(f"{cur_len}/{PERSONA_MAX_CHARS} 字")

    st.text_input("怎么称呼你（可留空）", value=p["call_user"], key="p_call_user")
    st.text_input("口头语（用 | 分隔）", value=p["catchphrase"], key="p_catchphrase")
    st.selectbox("回复长度", ["短", "中", "长"],
                 index=["短", "中", "长"].index(p["length"]), key="p_length")

    st.divider()
    st.subheader("性格")
    st.slider("脾气（越高越容易怼）", 0, 100, p["temper"], key="p_temper")
    st.slider("基础耐心", 0, 100, p["patience_base"], key="p_patience_base")
    st.slider("害羞/撒娇", 0, 100, p["shyness"], key="p_shyness")
    st.slider("主动性", 0, 100, p["initiative"], key="p_initiative")
    st.slider("毒舌", 0, 100, p["tease"], key="p_tease")

    st.divider()
    st.subheader("外观")

    avatar_file = st.file_uploader("AI 头像", type=["png", "jpg", "jpeg", "webp"], key="avatar_up")
    if avatar_file:
        new_path = save_uploaded_image(avatar_file, "avatar")
        if new_path and st.session_state.persona.get("avatar") != new_path:
            st.session_state.persona["avatar"] = new_path
            save_persona(st.session_state.persona)
            st.rerun()
    if st.session_state.persona.get("avatar") and os.path.exists(st.session_state.persona["avatar"]):
        st.image(st.session_state.persona["avatar"], width=80)
        if st.button("移除头像", use_container_width=True):
            st.session_state.persona["avatar"] = ""
            save_persona(st.session_state.persona)
            st.rerun()

    bg_file = st.file_uploader("聊天背景", type=["png", "jpg", "jpeg", "webp"], key="bg_up")
    if bg_file:
        new_bg = save_uploaded_image(bg_file, "background")
        if new_bg and st.session_state.persona.get("background") != new_bg:
            st.session_state.persona["background"] = new_bg
            save_persona(st.session_state.persona)
            st.rerun()
    if st.session_state.persona.get("background") and os.path.exists(st.session_state.persona["background"]):
        st.image(st.session_state.persona["background"], width=160)
        if st.button("移除背景", use_container_width=True):
            st.session_state.persona["background"] = ""
            save_persona(st.session_state.persona)
            st.rerun()

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("保存人设", use_container_width=True):
            new_p = {
                "name": st.session_state.p_name,
                "persona_text": st.session_state.p_persona_text[:PERSONA_MAX_CHARS],
                "call_user": st.session_state.p_call_user,
                "catchphrase": st.session_state.p_catchphrase,
                "length": st.session_state.p_length,
                "temper": st.session_state.p_temper,
                "patience_base": st.session_state.p_patience_base,
                "shyness": st.session_state.p_shyness,
                "initiative": st.session_state.p_initiative,
                "tease": st.session_state.p_tease,
                "avatar": st.session_state.persona.get("avatar", ""),
                "background": st.session_state.persona.get("background", ""),
            }
            st.session_state.persona = new_p
            save_persona(new_p)
            st.success("已保存")
            st.rerun()
    with col2:
        if st.button("恢复默认", use_container_width=True):
            new_p = {**DEFAULT_PERSONA}
            st.session_state.persona = new_p
            for k, v in new_p.items():
                st.session_state[f"p_{k}"] = v
            save_persona(new_p)
            st.rerun()

    st.divider()
    st.subheader("当前情绪")
    e = get_emotion()
    st.caption(f"心情：{e['mood']}｜耐心 {e['patience']}｜好感 {e['affection']}")
    if e.get("reason"):
        st.caption(f"原因：{e['reason']}")
    if st.button("重置情绪", use_container_width=True):
        save_emotion({"mood": "平静", "patience": 70, "affection": 50, "reason": ""})
        st.rerun()

    st.divider()
    st.subheader("主动聊天")
    cfg = load_proactive_config()
    pc_enabled = st.toggle("开启主动聊天", cfg["enabled"])
    pc_qs = st.slider("安静时段开始", 0, 23, cfg["quiet_start"])
    pc_qe = st.slider("安静时段结束", 0, 23, cfg["quiet_end"])
    pc_min = st.slider("最短间隔（分钟）", 5, 120, cfg["min_gap_minutes"])
    pc_max = st.slider("最长间隔（分钟）", 60, 720, cfg["max_gap_minutes"])
    pc_daily = st.slider("每天最多主动几次", 1, 30, cfg["max_per_day"])
    pc_typing = st.toggle("模拟打字间隔", cfg["typing_delay"])
    if st.button("保存主动设置", use_container_width=True):
        save_proactive_config({
            "enabled": pc_enabled, "quiet_start": pc_qs, "quiet_end": pc_qe,
            "min_gap_minutes": pc_min, "max_gap_minutes": pc_max,
            "max_per_day": pc_daily, "typing_delay": pc_typing,
        })
        set_next_send_at(compute_next_send_time(load_proactive_config(), load_persona()))
        st.success("已保存")
    next_at = get_next_send_at()
    if next_at:
        st.caption(f"下次可能：{next_at.strftime('%m-%d %H:%M')}")
    st.caption("依赖进程运行，关机/睡眠时不会发。")

    st.divider()
    st.subheader("记忆")
    st.caption(f"消息 {get_total_messages()} 条｜待摘要 {get_unsummarized_count()}｜今天主动 {get_today_proactive_count()}")
    profile_text = get_profile()
    if profile_text:
        with st.expander("用户画像"):
            try:
                st.json(json.loads(profile_text))
            except Exception:
                st.write(profile_text)
    summary_text = get_latest_summary()
    if summary_text:
        with st.expander("长期摘要"):
            st.write(summary_text)
    if st.button("清空全部记忆", use_container_width=True):
        clear_all_memory()
        st.success("已清空")
        st.rerun()

# ============ 主区域 ============
system_prompt = build_system_prompt(st.session_state.persona)

def get_avatar():
    a = st.session_state.persona.get("avatar", "")
    if a and os.path.exists(a):
        return a
    return None

@st.fragment(run_every=10)
def render_messages():
    msgs = get_recent_messages(limit=50)
    avatar = get_avatar()
    for m in msgs:
        if m["role"] == "assistant":
            with st.chat_message("assistant", avatar=avatar):
                if m.get("is_proactive"):
                    st.caption(f"🌿 {display_name(st.session_state.persona)}主动找你")
                st.markdown(m["content"])
        else:
            with st.chat_message("user"):
                st.markdown(m["content"])

render_messages()

CRISIS_WORDS = ["自杀", "不想活", "结束生命", "自残", "伤害自己", "活不下去", "想死"]

user_input = st.chat_input("说点什么吧……")

if user_input:
    save_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    crisis_hit = any(w in user_input for w in CRISIS_WORDS)

    payload = build_llm_messages(system_prompt)
    if crisis_hit:
        payload.append({"role": "system", "content": CRISIS_NOTE})

    avatar = get_avatar()
    with st.chat_message("assistant", avatar=avatar):
        with st.spinner(f"{display_name(st.session_state.persona)}正在输入……"):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=payload, temperature=0.95,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = "连接失败了，等下再试。"
                st.error(str(e))
        st.markdown(answer)

    save_message("assistant", answer)
    maybe_summarize()
    update_emotion_after_chat(st.session_state.persona)
    set_next_send_at(compute_next_send_time(load_proactive_config(), st.session_state.persona))
    st.rerun()
