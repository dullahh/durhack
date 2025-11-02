# pages/hotels.py
import os, json, inspect
import streamlit as st
import google.generativeai as genai

print(">>> USING hotels.py at:", __file__)

# ---- SAFE GEMINI CONFIG ----
API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Missing GOOGLE_API_KEY. Set it in .streamlit/secrets.toml or as an env var.")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"  # current model

# Optional: sanity print to server logs
print(">>> GenAI module file:", getattr(genai, "__file__", "(n/a)"))
print(">>> GenerativeModel from:", inspect.getfile(genai.GenerativeModel))

def ask_gemini_for_hotels_hub(city, start_iso, end_iso, party_size, budget="mid", vibe=None):
    model = genai.GenerativeModel(MODEL_NAME)
    system = (
        'You are a travel assistant. Return ONLY valid JSON (no markdown) with the shape: '
        '{ "hotels": [ { "name": str, "neighborhood": str, "description": str, '
        '"sustainability_hint": str, "distance_km": number, '
        '"price_tier": "budget|mid|upper|luxury", '
        '"why_good_for_group": str, "suggested_booking_sites": [str] } ] }'
    )
    user = (
        f"Destination: {city}\nDates: {start_iso} to {end_iso}\n"
        f"Group size: {party_size}\nBudget: {budget}\n"
        f"Vibe: {vibe or 'business-friendly, near transit, quiet'}\n"
        "Prefer hotels with meeting rooms, coworking spaces, and airport access."
    )
    try:
        resp = model.generate_content([system, user])
        text = (resp.text or "").strip()
    except Exception as e:
        # Friendly message if model name/SDK is out of date or key is bad
        st.error(f"Gemini request failed: {e}")
        st.info("Tip: `pip install -U google-generativeai` and use MODEL_NAME='gemini-2.5-flash'.")
        return []

    # Strip fences & clamp to outermost braces to tolerate minor format drift
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]

    try:
        return json.loads(text).get("hotels", [])
    except Exception as e:
        st.warning(f"Gemini parse error: {e}")
        return []

# (Optional) remove this guard; it's not useful within the same module.
# def __getattr__(name):
#     if name == "ask_gemini_for_hotels":
#         raise RuntimeError("Old function name used. Use ask_gemini_for_hotels_hub instead.")
#     raise AttributeError

# ---------------- UI ----------------
st.title("🏨 Hotel Suggestions")
city = st.session_state.get("chosen_city")
start_iso = st.session_state.get("event_start_iso")
end_iso = st.session_state.get("event_end_iso")
party = st.session_state.get("party_size")

if not city or not start_iso or not end_iso:
    st.info("Run the optimizer first to select a host city.")
    st.stop()

st.caption(f"{city} • {start_iso} → {end_iso} • {party} attendees")

colA, colB, colC = st.columns([1, 2, 1])
with colA:
    budget = st.selectbox("Budget", ["budget", "mid", "upper", "luxury"], index=1)
with colB:
    vibe = st.text_input("Vibe / constraints", "business-friendly, near transit, quiet")
with colC:
    run = st.button(f"Ask Gemini for {city} hotels", use_container_width=True)

if run:
    with st.spinner(f"Finding hotels in {city}..."):
        hotels = ask_gemini_for_hotels_hub(city, start_iso, end_iso, party, budget, vibe)

    if not hotels:
        st.warning("No structured results returned. Try simpler vibe or different budget.")
    else:
        for h in hotels:
            with st.container(border=True):
                st.markdown(f"### {h.get('name','(Unnamed)')}")
                sub = []
                if h.get("neighborhood"): sub.append(h["neighborhood"])
                if h.get("price_tier"):   sub.append(h["price_tier"])
                if h.get("distance_km") is not None: sub.append(f"{h['distance_km']} km")
                if sub: st.caption(" • ".join(map(str, sub)))
                if h.get("description"): st.write(h["description"])
                if h.get("sustainability_hint"): st.markdown(f"♻️ *{h['sustainability_hint']}*")
                if h.get("why_good_for_group"): st.markdown(f"**Why good for group:** {h['why_good_for_group']}")
                if h.get("suggested_booking_sites"):
                    st.markdown("**Suggested booking sites:** " + ", ".join(h["suggested_booking_sites"]))
