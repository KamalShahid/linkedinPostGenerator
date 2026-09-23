"""
LinkedIn Post Generator
-----------------------
A Streamlit web app that turns a topic or rough idea into a LinkedIn post,
using a large language model (LLM) served through the Groq API.

Where the Groq API key comes from (the app checks in this order):
  1. Streamlit secrets  -> on Streamlit Community Cloud, or .streamlit/secrets.toml on your PC
  2. Environment variable GROQ_API_KEY -> used when testing inside Google Colab
The key is never written in this file.
"""

import os

import groq                      # used for its error types (groq.AuthenticationError, ...)
import streamlit as st
from groq import Groq            # the client object that talks to the Groq API


# ---------------------------------------------------------------------------
# 1. PAGE SETUP (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="✍️",
    layout="centered",
)


# ---------------------------------------------------------------------------
# 2. SETTINGS YOU CAN EDIT
# ---------------------------------------------------------------------------
# Label shown in the app  ->  model ID that Groq understands.
# If Groq retires a model, replace the ID here (see console.groq.com/docs/models).
MODELS = {
    "GPT-OSS 120B (best quality)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (fastest)": "openai/gpt-oss-20b",
}

TONES = [
    "Professional",
    "Conversational",
    "Educational",
    "Inspirational",
    "Storytelling",
    "Thought leadership",
]

LENGTHS = {
    "Short (50–100 words)": "between 50 and 100 words",
    "Medium (120–200 words)": "between 120 and 200 words",
    "Long (220–300 words)": "between 220 and 300 words",
}

CTA_STYLES = {
    "Ask a question to invite comments": "End with one genuine, specific question that invites readers to share their own view or experience.",
    "Invite people to share or repost": "End with a short, natural invitation to share the post with someone who would find it useful.",
    "Point to a link, event or resource": "End with a brief line pointing readers to a link, event or resource (use a placeholder like [link] if none was given).",
    "Invite people to connect or follow": "End with a short, natural invitation to connect or follow for more on this topic.",
}

LINKEDIN_CHAR_LIMIT = 3000  # LinkedIn's maximum length for a normal post


# ---------------------------------------------------------------------------
# 3. THE "RULES" WE GIVE THE AI (system prompt)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an experienced LinkedIn ghostwriter who writes posts that read as if a thoughtful professional wrote them personally.

How you write:
- Open with a strong first line (the "hook") that makes a clear, specific point or observation. It must be honest: no clickbait, no fake suspense, no "You won't believe...".
- Use short paragraphs of one to three sentences, separated by a blank line, so the post is easy to read on a phone.
- Stay on one main idea. Give concrete, practical substance: an insight, a lesson, an example or a few actionable points.
- Sound like a real person: plain words, varied sentence length, natural rhythm.
- Avoid overused AI and corporate phrases such as "In today's fast-paced world", "game-changer", "delve", "unlock", "leverage synergies", "I'm thrilled to announce" (unless it really is an announcement), and "Let that sink in".
- Do not use em dashes. Use commas, full stops or colons instead.
- Never invent statistics, names, companies, quotes or personal experiences. If a specific detail would make the post stronger, insert a clearly marked placeholder in square brackets, e.g. [your result] or [company name].
- LinkedIn does not render Markdown, so do not use **bold**, *italics*, # headings or tables. Simple line breaks and, where helpful, a short list using "•" or "→" are fine.

Output only the finished post text. No title, no explanation, no quotation marks around it, no notes before or after."""


# ---------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def get_api_key():
    """Return the Groq API key from Streamlit secrets or an environment variable."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        # No secrets file exists (normal inside Colab) -> fall through to the environment variable
        pass
    key = os.environ.get("GROQ_API_KEY", "")
    return key.strip() or None


def build_user_prompt(topic, audience, tone, length, use_emojis, use_hashtags, add_cta, cta_style, extra):
    """Combine everything the user chose into one clear instruction for the AI."""
    lines = [
        f"Write a LinkedIn post about the following topic or idea:\n\"\"\"{topic.strip()}\"\"\"",
        "",
        f"Target audience: {audience.strip() if audience.strip() else 'general LinkedIn professionals'}",
        f"Tone: {tone}",
        f"Length: {LENGTHS[length]}",
    ]

    if use_emojis:
        lines.append("Emojis: use at most 2 or 3 relevant emojis, placed naturally. Do not start every line with one.")
    else:
        lines.append("Emojis: do not use any emojis.")

    if use_hashtags:
        lines.append("Hashtags: add 3 to 5 relevant hashtags on the final line only. Never put hashtags inside sentences.")
    else:
        lines.append("Hashtags: do not use any hashtags.")

    if add_cta:
        lines.append(f"Call-to-action: {CTA_STYLES[cta_style]}")
    else:
        lines.append("Call-to-action: none. End on a strong closing thought instead.")

    if extra.strip():
        lines.append(f"Additional instructions from the user: {extra.strip()}")

    return "\n".join(lines)


def clean_post(text):
    """Remove leftover Markdown symbols that LinkedIn would show literally."""
    text = text.replace("**", "").replace("__", "")
    return text.strip().strip('"').strip()


def generate_post(api_key, model_id, user_prompt, temperature):
    """Send the prompt to Groq and return the generated post as plain text."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=2048,  # upper limit on the reply size (includes the model's hidden reasoning)
        reasoning_effort="low",      # GPT-OSS models "think" first; "low" keeps it fast
    )
    text = response.choices[0].message.content
    if not text:
        raise ValueError("The model returned an empty response. Please click Generate again.")
    return clean_post(text)


# ---------------------------------------------------------------------------
# 5. SIDEBAR (settings)
# ---------------------------------------------------------------------------
api_key = get_api_key()

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("AI model", list(MODELS.keys()))
    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = more predictable wording. Higher = more varied and creative.",
    )
    st.divider()
    if api_key:
        st.success("Groq API key loaded.")
    else:
        st.error(
            "No Groq API key found. Add GROQ_API_KEY to Streamlit secrets "
            "(or set it as an environment variable in Colab)."
        )
    st.caption("Always review a generated post before publishing it.")


# ---------------------------------------------------------------------------
# 6. MAIN PAGE (inputs)
# ---------------------------------------------------------------------------
st.title("✍️ LinkedIn Post Generator")
st.write("Describe what you want to post about, choose a few options, and get a ready-to-edit LinkedIn post.")

topic = st.text_area(
    "What do you want to post about?",
    placeholder="e.g. Three lessons I learned from supervising final-year engineering projects this semester",
    height=130,
    max_chars=1500,
)

col1, col2 = st.columns(2)
with col1:
    audience = st.text_input("Target audience (optional)", placeholder="e.g. engineering students, HR leaders")
    tone = st.selectbox("Tone", TONES)
    length = st.selectbox("Post length", list(LENGTHS.keys()), index=1)
with col2:
    use_emojis = st.toggle("Use a few emojis", value=False)
    use_hashtags = st.toggle("Add hashtags (3–5)", value=True)
    add_cta = st.toggle("End with a call-to-action", value=True)
    cta_style = st.selectbox("Call-to-action style", list(CTA_STYLES.keys()), disabled=not add_cta)

extra = st.text_input(
    "Anything else the post should include? (optional)",
    placeholder="e.g. mention our free webinar on 15 October",
)

generate_clicked = st.button("Generate LinkedIn Post", type="primary")


# ---------------------------------------------------------------------------
# 7. WHAT HAPPENS WHEN THE BUTTON IS CLICKED (with error handling)
# ---------------------------------------------------------------------------
if generate_clicked:
    if not api_key:
        st.error("The Groq API key is missing, so the post can't be generated. See the sidebar message.")
    elif len(topic.strip()) < 5:
        st.warning("Please enter a topic or idea (at least a few words) before generating.")
    else:
        prompt = build_user_prompt(topic, audience, tone, length, use_emojis, use_hashtags, add_cta, cta_style, extra)
        try:
            with st.spinner("Writing your post..."):
                post = generate_post(api_key, MODELS[model_label], prompt, temperature)
            # Save the result so it stays on screen when the page refreshes
            st.session_state["post"] = post
            st.session_state["post_id"] = st.session_state.get("post_id", 0) + 1

        except groq.AuthenticationError:
            st.error("Groq rejected the API key (error 401). Check that the key is copied correctly and hasn't been deleted at console.groq.com/keys.")
        except groq.PermissionDeniedError:
            st.error("Access was denied (error 403). Your Groq account may not have access to this model, or your network is blocking Groq. Try the other model in the sidebar.")
        except groq.RateLimitError:
            st.error("Too many requests right now (error 429). Wait about a minute and try again.")
        except groq.NotFoundError:
            st.error(f"The model '{MODELS[model_label]}' was not found. It may have been retired. Pick the other model in the sidebar, or update MODELS in app.py.")
        except groq.BadRequestError as e:
            st.error(f"Groq could not process the request (error 400): {e.message}")
        except groq.APIConnectionError:
            st.error("Could not reach the Groq servers. Check the internet connection and try again.")
        except groq.APIStatusError as e:
            st.error(f"The Groq API returned an error (status {e.status_code}). Please try again shortly.")
        except Exception as e:
            st.error(f"Something unexpected went wrong: {e}")


# ---------------------------------------------------------------------------
# 8. SHOW THE RESULT
# ---------------------------------------------------------------------------
if "post" in st.session_state:
    post = st.session_state["post"]
    st.divider()
    st.subheader("Your LinkedIn post")

    edited = st.text_area(
        "Edit the post here if you like:",
        value=post,
        height=380,
        key=f"post_editor_{st.session_state['post_id']}",
    )

    words = len(edited.split())
    chars = len(edited)
    st.caption(f"{words} words · {chars} of {LINKEDIN_CHAR_LIMIT} characters")
    if chars > LINKEDIN_CHAR_LIMIT:
        st.warning("This is longer than LinkedIn allows. Shorten it before posting.")

    with st.expander("Copy with one click"):
        st.code(edited, language=None, wrap_lines=True)
        st.caption("Use the copy icon in the top-right corner of the box above.")

    st.download_button(
        "Download as .txt",
        data=edited,
        file_name="linkedin_post.txt",
        mime="text/plain",
    )
