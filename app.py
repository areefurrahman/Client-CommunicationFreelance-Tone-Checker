import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Freelance Tone Radar", page_icon="🎯", layout="wide")

MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

URGENT_WORDS = ["asap", "immediately", "urgent", "right now", "deadline",
                "overdue", "hours ago", "still waiting", "yet?", "by tonight"]
PASSIVE_WORDS = ["as i mentioned", "as i said", "per my last", "i thought we agreed",
                 "just to be clear", "not what we discussed", "i'm sure you're busy"]

TONES = {
    "frustrated": {"label": "Frustrated / Negative", "color": "#dc2626", "angle": "De-escalate",
        "meaning": "Client is upset. Stay calm and fix the problem.",
        "tip": "Say sorry for the trouble, then give a clear fix and a time.",
        "template": "Hi {name}, I'm sorry for the trouble. I understand this isn't what you expected. "
                    "Here's my plan: I'll fix [issue] and send the update by [time]. Does that work for you?"},
    "urgent": {"label": "Demanding / Urgent", "color": "#dc2626", "angle": "Give a clear status",
        "meaning": "Client feels time pressure or is losing patience.",
        "tip": "Give an exact status and delivery time. Don't be vague.",
        "template": "Hi {name}, thanks for checking in. Here's where things stand: [status]. "
                    "I'll deliver by [exact time]. I'll message you right away if anything changes."},
    "passive": {"label": "Passive-Aggressive / Tense", "color": "#d97706", "angle": "Set a clear boundary",
        "meaning": "Polite words, but real unhappiness underneath.",
        "tip": "Restate what was agreed, kindly, and keep it in writing.",
        "template": "Hi {name}, thanks for the feedback. To keep us aligned, our agreement was [scope/revisions]. "
                    "I'm happy to adjust within that. For anything extra, I can send a quick quote."},
    "friendly": {"label": "Friendly / Satisfied", "color": "#16a34a", "angle": "Upsell (gently)",
        "meaning": "Client is happy and open to more work.",
        "tip": "Say thanks, then softly offer more work or ask for a review.",
        "template": "Thank you, {name}! I'm glad you like it. If you'd like, I can also help with [related service]. "
                    "And a short review would mean a lot to me."},
    "neutral": {"label": "Professional / Neutral", "color": "#d97706", "angle": "Be clear and brief",
        "meaning": "Calm, business-like message.",
        "tip": "Answer clearly and confirm the next step.",
        "template": "Hi {name}, thanks for the message. Next step: [action]. I'll update you by [time]."},
}

SAMPLES = {
    "😡 Angry Client": "This is not what we discussed. I've asked for changes three times and it's still wrong. "
                       "This is unacceptable and I'm very disappointed.",
    "😊 Happy Client": "Thanks for the quick turnaround! Loved the color palette. You're amazing, let's work again soon.",
    "😒 Passive-Aggressive Client": "As I mentioned before, I thought we agreed the logo would be simpler. "
                                    "Just to be clear, this isn't quite what I expected. But I'm sure you're busy.",
}

def get_default_token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return ""

@st.cache_resource(show_spinner="Loading AI model (first run only)...")
def load_model(token):
    return pipeline("text-classification", model=MODEL_ID, top_k=None, token=token or None)

def analyze(text, clf, anger_thr, joy_thr):
    out = clf(text, truncation=True, max_length=512)
    if isinstance(out[0], list):
        out = out[0]
    scores = {d["label"]: d["score"] for d in out}
    low = text.lower()
    urgent = [w for w in URGENT_WORDS if w in low]
    passive = [w for w in PASSIVE_WORDS if w in low]
    negative = scores.get("anger", 0) + scores.get("disgust", 0)

    if negative >= anger_thr:
        key, conf = "frustrated", negative
    elif urgent:
        key, conf = "urgent", min(0.95, 0.55 + 0.15 * len(urgent))
    elif passive:
        key, conf = "passive", min(0.90, 0.50 + 0.15 * len(passive))
    elif scores.get("joy", 0) >= joy_thr:
        key, conf = "friendly", scores["joy"]
    else:
        key, conf = "neutral", scores.get("neutral", 0.5)
    return key, conf, scores, urgent, passive

def set_sample(text):
    st.session_state["msg"] = text

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    default_token = get_default_token()
    if default_token:
        st.success("Token loaded from secrets ✅")
        hf_token = default_token
    else:
        hf_token = st.text_input("Hugging Face Token (optional)", type="password",
                                 help="Public models work without it.")
    anger_thr = st.slider("Frustration threshold", 0.10, 0.90, 0.35, 0.05,
                          help="Lower = flags anger more easily.")
    joy_thr = st.slider("Friendly threshold", 0.10, 0.95, 0.50, 0.05,
                        help="Lower = flags happy messages more easily.")
    st.divider()
    st.subheader("💡 Tips for difficult clients")
    st.markdown(
        "- Wait 10 minutes before replying to an angry message.\n"
        "- Say sorry for the *feeling*, not for things you didn't do.\n"
        "- Always offer a clear next step and time.\n"
        "- Keep scope and revisions in writing.\n"
        "- Move big disputes to a call, then confirm in writing."
    )

# ---------------- Main ----------------
st.title("🎯 Freelance Tone Radar - Client Message Analyzer")
st.caption("Paste a client message to see its tone and get a smart reply angle.")

st.write("**Quick tests:**")
cols = st.columns(len(SAMPLES))
for col, (name, text) in zip(cols, SAMPLES.items()):
    col.button(name, on_click=set_sample, args=(text,), use_container_width=True)

message = st.text_area("Client message", key="msg", height=180,
                       placeholder="Paste a client message, email, or revision note here...")
client_name = st.text_input("Client name (for the reply template)", value="there")

if st.button("🔍 Analyze Tone", type="primary"):
    if not message.strip():
        st.warning("Please paste a message first.")
    else:
        clf = load_model(hf_token)
        with st.spinner("Analyzing..."):
            key, conf, scores, urgent, passive = analyze(message, clf, anger_thr, joy_thr)
        t = TONES[key]

        st.markdown(
            f"<div style='padding:14px 18px;border-radius:12px;background:{t['color']};"
            f"color:white;font-size:1.4rem;font-weight:700;display:inline-block'>"
            f"{t['label']}</div>", unsafe_allow_html=True)
        st.write("")

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{conf*100:.1f}%")
        c2.metric("Reply angle", t["angle"])
        c3.metric("Warning phrases found", len(urgent) + len(passive))

        left, right = st.columns(2)
        with left:
            st.subheader("Emotion breakdown")
            for label, val in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                st.progress(float(val), text=f"{label.title()}: {val*100:.1f}%")
            if urgent or passive:
                st.caption("Phrases spotted: " + ", ".join(urgent + passive))
        with right:
            st.subheader("🧠 Smart AI Reply Advice")
            st.info(f"**What it means:** {t['meaning']}")
            st.success(f"**Tip:** {t['tip']}")
            reply = t["template"].format(name=client_name or "there")
            st.text_area("Suggested reply (edit before sending)", reply, height=150)

        report = (
            "FREELANCE TONE RADAR REPORT\n"
            "===========================\n"
            f"Message: {message.strip()}\n\n"
            f"Tone: {t['label']}\n"
            f"Confidence: {conf*100:.1f}%\n"
            f"Meaning: {t['meaning']}\n"
            f"Reply angle: {t['angle']}\n"
            f"Tip: {t['tip']}\n\n"
            f"Suggested reply:\n{reply}\n"
        )
        st.subheader("📋 Report")
        st.code(report, language=None)  # the copy icon appears at the top right
        st.download_button("⬇️ Download report (.txt)", report, "tone_report.txt")

st.divider()
st.caption("⚠️ This tool gives a guide, not a fact. It can miss sarcasm. Always use your own judgment.")
