# app_streamlit_custom.py (updated integrated frontend)
import streamlit as st, requests, json, base64, io, os, urllib.parse
from pathlib import Path
from PIL import Image
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="CareBot Integrated", layout="wide")
st.title("CareBot — integrated demo")

# keep simple auth & cases export as before (reuse your existing users.json file)
BASE = Path.cwd().parent if Path.cwd().name == "frontend" else Path.cwd()
USERS_FILE = BASE / "users.json"
CASES_FILE = BASE / "cases.jsonl"

def read_users():
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf8"))
        except:
            return {}
    return {}

def save_case_local(case):
    with open(CASES_FILE, "a", encoding="utf8") as f:
        f.write(json.dumps(case) + "\n")

# session init
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "user": None}
if "chat" not in st.session_state:
    st.session_state.chat = {"step":0, "answers":{}, "messages":[]}
if "tta" not in st.session_state:
    st.session_state.tta = 2
if "gradcam_b64" not in st.session_state:
    st.session_state.gradcam_b64 = None

# Sidebar admin token
st.sidebar.header("Controls")
admin_token = st.sidebar.text_input("Admin token (optional)", type="password")
if st.sidebar.button("Refresh admin"):
    st.rerun()

# login/signup quick UI
if not st.session_state.auth["logged_in"]:
    st.subheader("Sign in")
    username = st.text_input("Username", key="u")
    password = st.text_input("Password", key="p", type="password")
    if st.button("Sign in"):
        users = read_users()
        if username in users and users[username]["pw_hash"] == __import__("hashlib").sha256(password.encode()).hexdigest():
            st.session_state.auth = {"logged_in":True, "user":username}
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.subheader("Or sign up")
    su = st.text_input("New username", key="su")
    supw = st.text_input("New password", key="supw", type="password")
    if st.button("Create account"):
        usr = read_users()
        if su in usr:
            st.error("Username exists")
        else:
            if not su or not supw:
                st.error("Provide both")
            else:
                usr[su] = {"pw_hash": __import__("hashlib").sha256(supw.encode()).hexdigest()}
                USERS_FILE.write_text(json.dumps(usr, indent=2), encoding="utf8")
                st.success("Account created")
else:
    user = st.session_state.auth["user"]
    st.sidebar.markdown(f"Signed in as **{user}**")
    if st.sidebar.button("Logout"):
        st.session_state.auth = {"logged_in":False, "user":None}
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("TTA (test-time augmentation)")
    st.session_state.tta = st.sidebar.selectbox("TTA", [1,2,3], index=1)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Admin functions")
    if admin_token:
        try:
            r = requests.get(f"{API_BASE}/admin/cases", params={"token": admin_token}, timeout=10)
            if r.status_code == 200:
                j = r.json()
                st.sidebar.success(f"Admin: {j['count']} cases")
                if st.sidebar.button("Show cases (sidebar)"):
                    st.sidebar.write(j["cases"][:50])
            else:
                st.sidebar.warning("Invalid admin token")
        except Exception as e:
            st.sidebar.error("Admin fetch failed: " + str(e))

    st.header("CareBot — conversation")
    # conversational flow simplified for demo
    if len(st.session_state.chat["messages"]) == 0:
        st.session_state.chat["messages"].append({"sender":"carebot","text":"Hello — I'm CareBot. I'll ask a few quick questions."})

    for msg in st.session_state.chat["messages"]:
        if msg["sender"]=="carebot":
            st.info("CareBot: " + msg["text"])
        else:
            st.markdown("**You:** " + msg["text"])

    # questions -> reuse simple flow
    if st.session_state.chat["step"] == 0:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Sex", ["male","female","other","unknown"])
        address = st.text_input("City / region (optional)")
        when_found = st.text_input("When first noticed (e.g., '2 weeks')")
        itches = st.selectbox("Itching/pain?", ["no","yes","unsure"])
        if st.button("Next"):
            st.session_state.chat["answers"] = {
                "age": age, "sex": sex, "address": address, "when_found": when_found, "itches": itches
            }
            st.session_state.chat["messages"].append({"sender":"user","text":"answered basic questions"})
            st.session_state.chat["step"] = 1
            st.rerun()
    elif st.session_state.chat["step"] == 1:
        st.markdown("### Upload lesion image")
        uploaded = st.file_uploader("Image", type=["jpg","jpeg","png"])
        if uploaded:
            st.image(uploaded, width=380)
        if st.button("Analyze image"):
            if not uploaded:
                st.error("Upload an image first")
            else:
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    r = requests.post(f"{API_BASE}/predict", files=files, data={"tta": str(st.session_state.tta)}, timeout=60)
                    if r.status_code != 200:
                        st.error("Prediction failed: " + r.text)
                    else:
                        j = r.json()
                        res = j.get("result") or j.get("stack_result") or j.get("result")
                        pred = res["pred_label"] if "pred_label" in res else res.get("pred_label")
                        st.session_state.chat["messages"].append({"sender":"carebot","text":f"Analysis complete. Predicted: {pred}"})
                        # save full case locally
                        local_case = {
                            "user": user, "timestamp": datetime.utcnow().isoformat(), "answers": st.session_state.chat["answers"],
                            "image": uploaded.name, "pred_label": pred
                        }
                        save_case_local(local_case)
                        st.session_state.chat["step"] = 2
                        st.session_state["_last_image_bytes"] = uploaded.getvalue()
                        st.session_state["_last_pred"] = pred
                        st.rerun()
                except Exception as e:
                    st.error("Request failed: " + str(e))

    else:
        # show results; option to request gradcam and booking
        pred = st.session_state.get("_last_pred", None)
        st.markdown(f"## Final result: **{pred}**")
        st.markdown("**Care plan (informational only)**")
        # lightweight guidance map (same as server mapping idea); keep short
        GUID = {
            "mel":"Suspicious for melanoma — see dermatologist urgently.",
            "nv":"Likely benign mole — monitor for change.",
            "bcc":"Possible BCC — see dermatologist.",
            "bkl":"Likely seborrheic keratosis — clinician review if symptomatic.",
            "akiec":"Actinic/precancerous lesion — dermatologist review advised.",
            "df":"Likely dermatofibroma — consult if symptomatic.",
            "vasc":"Vascular lesion—usually benign; consult if bleeding or painful."
        }
        st.info(GUID.get(pred, "Please consult a dermatologist for definitive advice."))

        # Grad-CAM toggle & display
        if st.button("Show Grad-CAM overlay"):
            try:
                img_bytes = st.session_state.get("_last_image_bytes")
                if not img_bytes:
                    st.error("No image available")
                else:
                    files = {"file": ("img.jpg", img_bytes, "image/jpeg")}
                    r = requests.post(f"{API_BASE}/gradcam", files=files, data={"model_key": ""}, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        b64 = data.get("overlay_b64")
                        st.session_state.gradcam_b64 = b64
                    else:
                        st.error("Grad-CAM failed: " + r.text)
            except Exception as e:
                st.error("Grad-CAM request error: " + str(e))
        if st.session_state.gradcam_b64:
            st.image(st.session_state.gradcam_b64, caption="Grad-CAM overlay", use_column_width=False, width=480)

        # Book appointment form
        st.markdown("### Book appointment")
        contact_email = st.text_input("Your contact email")
        notes = st.text_area("Notes (optional)")
        if st.button("Request booking"):
            req = {
                "user": user,
                "patient_age": st.session_state.chat["answers"].get("age"),
                "patient_sex": st.session_state.chat["answers"].get("sex"),
                "address": st.session_state.chat["answers"].get("address"),
                "pred_label": pred,
                "notes": notes,
                "contact_email": contact_email
            }
            try:
                r = requests.post(f"{API_BASE}/book", json=req, timeout=30)
                if r.status_code == 200:
                    st.success("Booking request submitted.")
                else:
                    st.warning("Booking saved locally. SMTP email not configured.")
            except Exception as e:
                st.error("Booking failed: " + str(e))

        if st.button("Start new case"):
            st.session_state.chat = {"step":0,"answers":{},"messages":[]}
            st.session_state.gradcam_b64 = None
            st.rerun()
