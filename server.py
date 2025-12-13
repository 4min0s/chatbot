import os
import re
import json
import joblib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from google import genai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


# =========================
# CONFIG
# =========================

FILE_NAME_ID = "in_domain_banking_en.txt"
RAG_PATH = "rag_index.joblib"

# Your category labels (must match candidate_labels below)
CANDIDATE_LABELS = ["cartes", "clients", "accounts"]

# threshold for category confidence
CATEGORY_THRESHOLD = 0.5

# RAG threshold
RAG_SIM_THRESHOLD = 0.30


# =========================
# HELPERS
# =========================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def record_to_text(record: Dict[str, Any], category: str) -> str:
    """Convert a structured record into a searchable text chunk."""
    if category == "clients":
        a = record.get("address", {}) or {}
        e = record.get("employment", {}) or {}
        k = record.get("kyc", {}) or {}
        return (
            f"[CLIENT] client_id={record.get('client_id')} "
            f"name={record.get('full_name')} gender={record.get('gender')} birth_date={record.get('birth_date')} "
            f"nationality={record.get('nationality')} phone={record.get('phone')} email={record.get('email')} "
            f"address_city={a.get('city')} governorate={a.get('governorate')} postal_code={a.get('postal_code')} "
            f"employment_status={e.get('status')} sector={e.get('sector')} job_title={e.get('job_title')} "
            f"monthly_net_income_tnd={e.get('monthly_net_income_tnd')} "
            f"kyc_level={k.get('level')} pep={k.get('pep')} risk_score={k.get('risk_score')} risk_band={k.get('risk_band')} "
            f"status={record.get('status')} created_at={record.get('created_at')}"
        )

    if category == "accounts":
        od = record.get("overdraft", {}) or {}
        fees = record.get("fees", {}) or {}
        limits = record.get("limits", {}) or {}
        flags = record.get("flags", {}) or {}
        interest = record.get("interest", {}) or {}
        return (
            f"[ACCOUNT] account_id={record.get('account_id')} client_id={record.get('client_id')} "
            f"bank={record.get('bank_name')} branch_code={record.get('branch_code')} iban={record.get('iban')} "
            f"currency={record.get('currency')} type={record.get('account_type')} package={record.get('package')} "
            f"opened_date={record.get('opened_date')} status={record.get('status')} "
            f"balance_tnd={record.get('balance_tnd')} available_balance_tnd={record.get('available_balance_tnd')} "
            f"overdraft_enabled={od.get('enabled')} overdraft_limit_tnd={od.get('limit_tnd')} overdraft_used_tnd={od.get('used_tnd')} "
            f"monthly_fee_tnd={fees.get('monthly_fee_tnd')} other_bank_atm_fee_tnd={fees.get('atm_withdrawal_fee_other_bank_tnd')} "
            f"daily_withdrawal_tnd={limits.get('daily_withdrawal_tnd')} daily_transfer_tnd={limits.get('daily_transfer_tnd')} "
            f"salary_domiciliation={flags.get('salary_domiciliation')} blocked={flags.get('blocked')} block_reason={flags.get('block_reason')} "
            f"interest_rate_apr={interest.get('rate_apr')} interest_payout={interest.get('interest_payout')}"
        )

    if category == "cards":
        limits = record.get("limits", {}) or {}
        sec = record.get("security", {}) or {}
        wallets = sec.get("tokenized_wallets", []) or []
        return (
            f"[CARD] card_id={record.get('card_id')} client_id={record.get('client_id')} account_id={record.get('account_id')} "
            f"bank={record.get('bank_name')} type={record.get('card_type')} network={record.get('network')} product={record.get('product')} "
            f"masked_pan={record.get('masked_pan')} expiry={record.get('expiry')} status={record.get('status')} "
            f"contactless={record.get('contactless')} ecommerce_enabled={record.get('ecommerce_enabled')} atm_enabled={record.get('atm_enabled')} "
            f"daily_pos_tnd={limits.get('daily_pos_tnd')} daily_atm_tnd={limits.get('daily_atm_tnd')} monthly_ecom_tnd={limits.get('monthly_ecom_tnd')} "
            f"pin_failed_attempts={sec.get('pin_failed_attempts')} last_pin_change={sec.get('last_pin_change')} "
            f"tokenized_wallets={','.join(wallets)} fraud_risk_score={sec.get('fraud_risk_score')} issued_at={record.get('issued_at')}"
        )

    return json.dumps(record, ensure_ascii=False)


def build_rag_index(embedder, clients_path="clients.jsonl", accounts_path="accounts.jsonl", cards_path="cards.jsonl"):
    clients = load_jsonl(clients_path)
    accounts = load_jsonl(accounts_path)
    cards = load_jsonl(cards_path)

    docs = []
    meta = []

    for r in clients:
        docs.append(record_to_text(r, "clients"))
        meta.append({"category": "clients", "id": r.get("client_id")})

    for r in accounts:
        docs.append(record_to_text(r, "accounts"))
        meta.append({"category": "accounts", "id": r.get("account_id"), "client_id": r.get("client_id")})

    for r in cards:
        docs.append(record_to_text(r, "cards"))
        meta.append({"category": "cards", "id": r.get("card_id"), "client_id": r.get("client_id"), "account_id": r.get("account_id")})

    emb = embedder.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    return {"docs": docs, "meta": meta, "emb": emb, "embedder": embedder}


def save_rag_index(rag_index: Dict[str, Any], path="rag_index.joblib"):
    to_save = {
        "docs": rag_index["docs"],
        "meta": rag_index["meta"],
        "emb": rag_index["emb"],
        "embedder_name": "all-MiniLM-L6-v2",
    }
    joblib.dump(to_save, path, compress=3)


def load_rag_index(path="rag_index.joblib"):
    data = joblib.load(path)
    embedder = SentenceTransformer(data["embedder_name"])
    rag_index = {
        "docs": data["docs"],
        "meta": data["meta"],
        "emb": data["emb"],
        "embedder": embedder,
    }
    return rag_index


def cosine_top_k(query_emb: np.ndarray, emb_matrix: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    sims = emb_matrix @ query_emb.reshape(-1, 1)
    sims = sims.reshape(-1)
    idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def map_predicted_category(predicted_category: str) -> str:
    # candidate labels are: cartes, clients, accounts
    if predicted_category == "cartes":
        return "cards"
    if predicted_category == "clients":
        return "clients"
    if predicted_category == "accounts":
        return "accounts"
    return ""


def retrieve_context(query: str, rag_index: Dict[str, Any], predicted_category: str, top_k: int = 6):
    embedder = rag_index["embedder"]
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]

    docs = rag_index["docs"]
    meta = rag_index["meta"]
    emb = rag_index["emb"]

    cat = map_predicted_category(predicted_category) if predicted_category else ""
    if cat:
        mask_idx = [i for i, m in enumerate(meta) if m.get("category") == cat]
        if mask_idx:
            sub_emb = emb[mask_idx]
            idx_sub, sims_sub = cosine_top_k(query_emb, sub_emb, k=min(top_k, len(mask_idx)))
            chosen = [mask_idx[i] for i in idx_sub]
            items = [(docs[i], meta[i], float(s)) for i, s in zip(chosen, sims_sub)]
            max_sim = float(sims_sub[0]) if len(sims_sub) else 0.0
        else:
            items, max_sim = [], 0.0
    else:
        idx, sims = cosine_top_k(query_emb, emb, k=min(top_k, len(docs)))
        items = [(docs[i], meta[i], float(s)) for i, s in zip(idx, sims)]
        max_sim = float(sims[0]) if len(sims) else 0.0

    context_lines = [f"- (sim={s:.3f}) {t}" for (t, m, s) in items]
    return "\n".join(context_lines), max_sim, items


def detect_category(text, classifier, candidate_labels, threshold=0.5):
    if not text:
        return None, 0.0, False
    try:
        result = classifier(text, candidate_labels)
        label = result["labels"][0]
        score = float(result["scores"][0])
        return label, score, score >= threshold
    except Exception:
        return None, 0.0, False


def is_in_domain(text, ocsvm_model, if_model, embedder):
    if not text:
        return False, "Empty text."

    X_test = embedder.encode([text])
    ocsvm_prediction = ocsvm_model.predict(X_test)[0]  

    is_id = (ocsvm_prediction == 1) 
    if is_id:
        return True, "In-domain (OCSVM=ID, IF=ID)."

    status = f"OCSVM={'ID' if ocsvm_prediction == 1 else 'OOD'}"
    return False, f"Out-of-domain or suspicious. Details: {status}"


def traduction(client, user_text: str, detected_lang: str) -> str:
    """Translate to English if language is not English."""
    if detected_lang != "en":
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Return ONLY the translated sentence, nothing else. Translate this text to English: " + user_text,
            )
            return (response.text or "").strip() or user_text
        except Exception:
            return user_text
    return user_text


# =========================
# ANALYTICS
# =========================

def is_analytics_query(q: str) -> bool:
    q = (q or "").lower()
    patterns = [
        r"\bhighest\b", r"\bmaximum\b", r"\bmax\b",
        r"\blowest\b", r"\bminimum\b", r"\bmin\b",
        r"\btotal\b", r"\bsum\b",
        r"\baverage\b", r"\bavg\b", r"\bmean\b", r"\bmedian\b",
        r"\bcount\b", r"\bhow many\b",
        r"\btop\s+\d+\b",
    ]
    return any(re.search(p, q) for p in patterns)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def detect_top_n(q: str, default=1) -> int:
    m = re.search(r"\btop\s+(\d+)\b", q.lower())
    if m:
        return max(1, int(m.group(1)))
    return default


def analytics_accounts(query: str, accounts: List[Dict[str, Any]]) -> str:
    q = query.lower()
    if not accounts:
        return "I can't compute this because accounts.jsonl is empty or missing."

    if "count" in q or "how many" in q:
        return f"There are {len(accounts)} accounts in the dataset."

    if "total" in q or "sum" in q:
        if "available" in q:
            total = sum(safe_float(a.get("available_balance_tnd", 0)) for a in accounts)
            return f"The total available balance across all accounts is {total:.2f} TND."
        total = sum(safe_float(a.get("balance_tnd", 0)) for a in accounts)
        return f"The total balance across all accounts is {total:.2f} TND."

    if "average" in q or "avg" in q or "mean" in q:
        if "available" in q:
            vals = [safe_float(a.get("available_balance_tnd", 0)) for a in accounts]
            return f"The average available balance is {sum(vals)/max(1,len(vals)):.2f} TND."
        vals = [safe_float(a.get("balance_tnd", 0)) for a in accounts]
        return f"The average balance is {sum(vals)/max(1,len(vals)):.2f} TND."

    if "median" in q:
        if "available" in q:
            vals = sorted(safe_float(a.get("available_balance_tnd", 0)) for a in accounts)
            metric = "available balance"
        else:
            vals = sorted(safe_float(a.get("balance_tnd", 0)) for a in accounts)
            metric = "balance"

        n = len(vals)
        med = vals[n//2] if n % 2 == 1 else (vals[n//2 - 1] + vals[n//2]) / 2
        return f"The median {metric} is {med:.2f} TND."

    want_max = any(w in q for w in ["highest", "maximum", "max", "top"])
    want_min = any(w in q for w in ["lowest", "minimum", "min"])

    if want_max or want_min:
        n = detect_top_n(q, default=1)
        metric_key = "available_balance_tnd" if "available" in q else "balance_tnd"
        metric_name = "available balance" if "available" in q else "balance"

        sorted_acc = sorted(
            accounts,
            key=lambda a: safe_float(a.get(metric_key, -1)),
            reverse=want_max,
        )
        top_list = sorted_acc[:n]

        if not top_list:
            return f"I couldn't find {metric_name} values in accounts."

        if n == 1:
            a = top_list[0]
            return (
                f"The {'highest' if want_max else 'lowest'} {metric_name} is "
                f"{safe_float(a.get(metric_key, 0)):.2f} TND "
                f"(account_id={a.get('account_id')}, client_id={a.get('client_id')}, bank={a.get('bank_name')})."
            )

        lines = []
        for i, a in enumerate(top_list, 1):
            lines.append(
                f"{i}) {safe_float(a.get(metric_key, 0)):.2f} TND "
                f"(account_id={a.get('account_id')}, bank={a.get('bank_name')})"
            )
        return f"Top {n} accounts by {metric_name}:\n" + "\n".join(lines)

    return "I detected an analytics question, but I don’t support that exact metric yet."


def analytics_cards(query: str, cards: List[Dict[str, Any]]) -> str:
    q = query.lower()
    if not cards:
        return "I can't compute this because cards.jsonl is empty or missing."

    if "count" in q or "how many" in q:
        return f"There are {len(cards)} cards in the dataset."

    if "fraud" in q and any(w in q for w in ["max", "maximum", "highest"]):
        best = max(cards, key=lambda c: safe_float((c.get("security") or {}).get("fraud_risk_score", -1)))
        score = safe_float((best.get("security") or {}).get("fraud_risk_score", 0))
        return (
            f"The highest fraud risk score is {score:.0f} "
            f"(card_id={best.get('card_id')}, bank={best.get('bank_name')}, client_id={best.get('client_id')})."
        )

    return "I detected an analytics question about cards, but I don’t support that exact metric yet."


def analytics_router(query: str,
                     predicted_category: str,
                     accounts_data: List[Dict[str, Any]],
                     cards_data: List[Dict[str, Any]],
                     clients_data: List[Dict[str, Any]]) -> str:
    q = query.lower()
    cat = map_predicted_category(predicted_category)

    if cat == "accounts" or "balance" in q or "account" in q:
        return analytics_accounts(query, accounts_data)

    if cat == "cards" or "card" in q:
        return analytics_cards(query, cards_data)

    return analytics_accounts(query, accounts_data)


# =========================
# TRAIN ID MODELS
# =========================

def load_and_train_models(file_name_id: str):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    try:
        with open(file_name_id, "r", encoding="utf-8") as f:
            in_domain_texts_en = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise RuntimeError(f"FATAL: '{file_name_id}' not found. Create it and add in-domain English banking queries.")

    if not in_domain_texts_en:
        raise RuntimeError(f"FATAL: '{file_name_id}' is empty. Add in-domain English queries.")

    X_train = embedding_model.encode(in_domain_texts_en, show_progress_bar=True)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.15, gamma="auto")
    isolation_forest = IsolationForest(contamination=0.15, random_state=42)

    ocsvm.fit(X_train)
    isolation_forest.fit(X_train)

    return embedding_model, ocsvm, isolation_forest


# =========================
# BOT STATE (GLOBAL)
# =========================

BOT: Dict[str, Any] = {}


def init_bot():
  

    client = genai.Client(api_key="AIzaSyDizqGzWjriB5KpmYTUf5rNqJZbedlns10")

    # Train/load ID models
    embedding_model, ocsvm, isolation_forest = load_and_train_models(FILE_NAME_ID)

    # Load/build RAG index
    if os.path.exists(RAG_PATH):
        rag_index = load_rag_index(RAG_PATH)
    else:
        rag_index = build_rag_index(embedding_model, "clients.jsonl", "accounts.jsonl", "cards.jsonl")
        save_rag_index(rag_index, RAG_PATH)

    # Load structured datasets for analytics
    accounts_data = load_jsonl("accounts.jsonl")
    cards_data = load_jsonl("cards.jsonl")
    clients_data = load_jsonl("clients.jsonl")

    # Language detector
    model_name = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    language_detector = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Category classifier
    category_model_name = "joeddav/xlm-roberta-large-xnli"
    category_classifier = pipeline("zero-shot-classification", model=category_model_name)

    BOT.update({
        "client": client,
        "embedding_model": embedding_model,
        "ocsvm": ocsvm,
        "isolation_forest": isolation_forest,
        "rag_index": rag_index,
        "accounts_data": accounts_data,
        "cards_data": cards_data,
        "clients_data": clients_data,
        "language_detector": language_detector,
        "category_classifier": category_classifier,
    })


def chat_once(user_text: str, return_debug: bool = False) -> Dict[str, Any]:
    user_text = (user_text or "").strip()
    if not user_text:
        return {"answer": ""}

    client = BOT["client"]
    language_detector = BOT["language_detector"]
    category_classifier = BOT["category_classifier"]

    embedding_model = BOT["embedding_model"]
    ocsvm = BOT["ocsvm"]
    isolation_forest = BOT["isolation_forest"]

    rag_index = BOT["rag_index"]
    accounts_data = BOT["accounts_data"]
    cards_data = BOT["cards_data"]
    clients_data = BOT["clients_data"]

    # 1) language detect
    lang_result = language_detector(user_text)
    detected_lang = lang_result[0]["label"]

    # 2) translate to EN for internal detection/retrieval
    text_for_detection = traduction(client, user_text, detected_lang)

    # 3) category
    predicted_category, cat_score, cat_confident = detect_category(
        text_for_detection,
        category_classifier,
        CANDIDATE_LABELS,
        threshold=CATEGORY_THRESHOLD,
    )

    # 4) in-domain
    is_id, status_message = is_in_domain(text_for_detection, ocsvm, isolation_forest, embedding_model)

    # If you want to ALWAYS answer even if OOD, set this True.
    # But for safety/banking assistant, you normally keep it False.
    # force_answer = False
    force_answer = True  # (match your current behavior)

    if not is_id and not force_answer:
        if detected_lang == "fr":
            ans = "Je suis désolé, je ne réponds qu'aux questions liées aux services bancaires (comptes, cartes, clients)."
        else:
            ans = "I can only answer banking-related questions (accounts, cards, clients)."
        out = {"answer": ans}
        if return_debug:
            out["debug"] = {
                "detected_lang": detected_lang,
                "predicted_category": predicted_category,
                "cat_score": cat_score,
                "cat_confident": cat_confident,
                "status_message": status_message,
            }
        return out

    # 5) Analytics first
    if is_analytics_query(text_for_detection):
        ans = analytics_router(text_for_detection, predicted_category, accounts_data, cards_data, clients_data)
        out = {"answer": ans}
        if return_debug:
            out["debug"] = {
                "detected_lang": detected_lang,
                "predicted_category": predicted_category,
                "cat_score": cat_score,
                "cat_confident": cat_confident,
                "status_message": status_message,
                "route": "analytics",
            }
        return out

    # 6) RAG retrieve
    context_text, max_sim, items = retrieve_context(text_for_detection, rag_index, predicted_category, top_k=6)

    too_general = (max_sim < RAG_SIM_THRESHOLD) or (predicted_category is None) or (not cat_confident)
    category_hint = f"Category: {predicted_category}." if predicted_category else ""

    # 7) Prompt
    if not too_general and context_text.strip():
        prompt_response = (
            "You are a helpful bank assistant. Use ONLY the provided dataset context to answer precisely. "
            "If the context does not contain enough information, give a short safe general answer (do NOT say you lack data).\n\n"
            f"DATASET CONTEXT:\n{context_text}\n\n"
            f"USER QUESTION: {text_for_detection}\n"
            f"{category_hint}\n"
             f"Answer must be entirely in this language {detected_lang}."
        )
    else:
        prompt_response = (
            "You are a helpful bank assistant. The user asked a general banking question. "
            "Answer directly (do NOT say you lack data). Assume Tunisia banking context. "
            "If needed, give security advice on protecting credentials only if the client tells you to GIVE ME SECURITY ADVICES .\n\n"
            f"USER QUESTION: {text_for_detection}\n"
            f"{category_hint}\n"
            f"Answer must be entirely in this language {detected_lang}."
        )

    # 8) Call Gemini
    try:
        response_gemini = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_response,
        ).text
    except Exception as e:
        response_gemini = f"Service error: {e}"

    out = {"answer": response_gemini or ""}
    if return_debug:
        out["debug"] = {
            "detected_lang": detected_lang,
            "predicted_category": predicted_category,
            "cat_score": cat_score,
            "cat_confident": cat_confident,
            "status_message": status_message,
            "max_sim": float(max_sim),
            "too_general": bool(too_general),
            "route": "rag+gemini" if not too_general else "general+gemini",
        }
    return out


# =========================
# FASTAPI
# =========================

app = FastAPI()

# CORS: allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production: restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    debug: Optional[bool] = False

@app.on_event("startup")
def startup_event():
    init_bot()

@app.post("/chat")
def chat(req: ChatRequest):
    return chat_once(req.message, return_debug=bool(req.debug))
