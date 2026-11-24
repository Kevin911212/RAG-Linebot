# ═══ 多策略 RAG 文件問答（Streamlit 線上版）· Groq + ChromaDB + LINE Bot ═══
# 執行：streamlit run app.py
# 使用者在側欄自行貼上 Groq API Key 與 LINE Channel Access Token（只存在該使用者的連線 session，不寫入伺服器）
from __future__ import annotations

import csv
import io
import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
import streamlit as st
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph
from groq import Groq
from linebot.v3.messaging import ApiClient, BroadcastRequest, Configuration, MessagingApi, TextMessage
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 設定 ──
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")   # 可換 multilingual-e5-base、BAAI/bge-m3
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 500, 100, 200, 40, 7000
LINE_LIMIT = 500  # 推送到 LINE 的文字上限（字）

STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf)
            buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p
        fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


@dataclass
class Doc:
    sid: str
    name: str
    chunks: list
    labels: list
    n_child: int = 0
    tfidf: object = field(default=None, repr=False)
    tm: object = field(default=None, repr=False)


class LLM:
    def __init__(self, key, model):
        key = (key or "").strip()
        self.client = Groq(api_key=key) if key else None
        self.model = model or MODELS[0]

    def __call__(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client:
            return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try:
                r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower():
                    raise
                kw.pop("reasoning_effort", None)
                kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err:
                raise
            return None


def list_models(key):
    try:
        ids = [m.id for m in Groq(api_key=key.strip()).models.list().data if getattr(m, "active", True) is not False]
    except Exception:
        return []
    return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))


@st.cache_resource(show_spinner=False)
def get_embedder():
    """嵌入模型只載入一次，所有使用者 session 共用。"""
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    m.max_seq_length = min(512, m.max_seq_length or 512)
    return m


@st.cache_resource(show_spinner=False)
def get_db():
    return chromadb.EphemeralClient()  # 記憶體模式；各文件以 sid 區隔，彼此不會混用


_ENC_LOCK = threading.Lock()


class RAG:
    """每個使用者 session 一個實例：只持有自己目前載入的單一文件。"""

    def __init__(self):
        self.e5 = "e5" in EMBED_MODEL.lower()
        self.db = get_db()
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL.split("/")[-1])[:40].strip("_")
        self.col = self.db.get_or_create_collection(f"p_{slug}", metadata={"hnsw:space": "cosine"})
        self.ccol = self.db.get_or_create_collection(f"c_{slug}", metadata={"hnsw:space": "cosine"})
        self.doc: Doc | None = None

    def embedder(self):
        return get_embedder()

    # ── 文件載入 ──
    def load(self, path):
        t0, path = time.time(), Path(path)
        if path.suffix.lower() not in (".pdf", ".docx"):
            raise ValueError("僅支援 PDF / DOCX")
        segs = self._pdf(path) if path.suffix.lower() == ".pdf" else self._docx(path)
        pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
        if not pairs:
            raise ValueError("無可擷取文字（可能是掃描檔，需先 OCR）")
        doc = Doc(uuid.uuid4().hex, path.name, [p[0] for p in pairs], [p[1] for p in pairs])
        try:
            self._add(self.col, [f"{doc.sid}_{i}" for i in range(len(pairs))], doc.chunks,
                      [{"sid": doc.sid, "idx": i} for i in range(len(pairs))])
            kids = [(f"{doc.sid}_{p}_{c}", t, p) for p, par in enumerate(doc.chunks)
                    for c, t in enumerate(split_text(par, CHILD, CHILD_OVERLAP))]
            doc.n_child = len(kids)
            self._add(self.ccol, [k[0] for k in kids], [k[1] for k in kids], [{"sid": doc.sid, "idx": k[2]} for k in kids])
            doc.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            doc.tm = doc.tfidf.fit_transform(doc.chunks)
        except Exception:
            self.drop(doc)
            raise
        self.drop(self.doc)  # 新文件成功後才釋放舊文件
        self.doc = doc
        return f"✓ {doc.name}\n{len(doc.chunks)} 片段 · {time.time() - t0:.1f}s"

    def drop(self, doc):
        if doc is None:
            return
        for c in (self.col, self.ccol):
            try:
                c.delete(where={"sid": doc.sid})
            except Exception:
                pass

    def _pdf(self, path):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(str(path)).pages, 1) if (t := p.extract_text() or "").strip()]

    def _docx(self, path):
        doc, segs, head, buf, nt = Document(str(path)), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc)
                txt = para.text.strip()
                if not txt:
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush()
                    head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1
                rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t):
                            cells.append(t)
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    flush()
                    segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def _enc(self, texts, query=False):
        if self.e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        with _ENC_LOCK:  # 多人同時使用時，避免同時佔用嵌入模型
            return self.embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _add(self, col, ids, docs, metas, b=500):
        for i in range(0, len(docs), b):
            col.add(ids=ids[i:i + b], documents=docs[i:i + b], metadatas=metas[i:i + b], embeddings=self._enc(docs[i:i + b]))

    def vec(self, doc, qs, k, child=False):
        total = doc.n_child if child else len(doc.chunks)
        if not total:
            return [[] for _ in qs]
        res = (self.ccol if child else self.col).query(query_embeddings=self._enc(qs, True), n_results=max(1, min(k, total)),
                                                       where={"sid": doc.sid}, include=["metadatas"])
        return [[int(m["idx"]) for m in ms] for ms in res["metadatas"]]

    @staticmethod
    def kw(doc, q, k):
        s = (doc.tm @ doc.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)):
                sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, doc, q, strategy, k, llm):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec(doc, [q], n * 2)[0], self.kw(doc, q, n * 2)], n)
        if strategy == "semantic":
            idx = self.vec(doc, [q], k)[0]
        elif strategy == "tfidf":
            idx = self.kw(doc, q, k)
        elif strategy == "parent_child":
            idx = list(dict.fromkeys(self.vec(doc, [q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec(doc, [q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec(doc, [q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12))
            idx = cands[:k]
            if llm.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {doc.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                        15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else:  # hybrid、compress
            idx = hybrid(k)
        hits = [(i, doc.chunks[i]) for i in idx]
        if strategy == "compress" and llm.client and hits:
            def comp(h):
                r = llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex:
                hits = list(ex.map(comp, hits))
        return hits

    def answer(self, q, strategy, k, model, key):
        """回傳 (回答, 參考片段, 耗時說明)。"""
        doc = self.doc
        if doc is None:
            return "請先載入 PDF 或 DOCX 文件。", "", ""
        llm = LLM(key, model)
        t0 = time.time()
        hits = self.retrieve(doc, q, strategy, int(k), llm)
        t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{doc.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX:
                break
            blocks.append(b)
            used += len(b)
        ctx = "\n\n".join(blocks)
        src = f"📄 {doc.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s\n{'─' * 40}\n\n{ctx}"
        if not hits:
            return "找不到相關片段，請換個問法或策略。", src, ""
        if not llm.client:
            return "⚠ 尚未設定 Groq API Key（檢索結果請看「檢索片段」分頁）", src, ""
        ans = llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                  f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                  1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        return ans or "（模型未回傳內容）", src, f"{llm.model} · 檢索 {t1 - t0:.1f}s · 生成 {time.time() - t1:.1f}s"


# ── 匯出：Word / CSV ──
def csv_bytes(records):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
    return buf.getvalue().encode("utf-8-sig")  # utf-8-sig：Excel 開啟中文不亂碼


def docx_bytes(records, with_src=True):
    buf = io.BytesIO()
    export_docx(records, buf, with_src)
    return buf.getvalue()


def _md_runs(par, text):  # 將 **粗體** 轉成 Word 粗體
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            par.add_run(part).bold = bool(i % 2)


def export_docx(records, path, with_src=True):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Microsoft JhengHei", Pt(11)
    st.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Microsoft JhengHei")
    doc.add_heading("RAG 文件問答紀錄", 0)
    meta = doc.add_paragraph(f"文件：{records[-1]['文件']}　·　匯出時間：{datetime.now():%Y-%m-%d %H:%M}　·　共 {len(records)} 題")
    meta.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    for n, r in enumerate(records, 1):
        doc.add_heading(f"Q{n}．{r['問題']}", level=1)
        info = doc.add_paragraph(f"{r['時間']}　·　{r['策略']}　·　{r['模型']}")
        info.runs[0].font.size, info.runs[0].font.color.rgb = Pt(9), RGBColor(0x6B, 0x72, 0x80)
        for line in r["回答"].splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            if m := re.match(r"^\s*#{1,6}\s+(.*)", line):
                doc.add_heading(m.group(1).replace("**", ""), level=2)
                continue
            if m := re.match(r"^(\s*)[-*•]\s+(.*)", line):
                par = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet")
                _md_runs(par, m.group(2))
            elif m := re.match(r"^\s*\d+[.)]\s+(.*)", line):
                par = doc.add_paragraph(style="List Number")
                _md_runs(par, m.group(1))
            else:
                _md_runs(doc.add_paragraph(), line)
        if with_src and r["參考片段"]:
            h = doc.add_paragraph()
            run = h.add_run("參考片段")
            run.bold, run.font.size = True, Pt(9)
            for blk in r["參考片段"].split("\n\n"):
                p = doc.add_paragraph(blk.strip())
                p.paragraph_format.left_indent = Pt(12)
                for run in p.runs:
                    run.font.size, run.font.color.rgb = Pt(8.5), RGBColor(0x55, 0x5B, 0x66)
    doc.save(path)


def plain(md: str) -> str:
    """Tk 文字框不渲染 Markdown：移除粗體標記，標題轉成一般行。"""
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    return re.sub(r"^\s*#{1,6}\s+", "", md, flags=re.M)


# ── LINE Bot ──
def format_line(q: str, ans: str, limit: int = LINE_LIMIT) -> str:
    """問題 + 回答合併後不超過 limit 字；移除 Markdown 與 [片段n] 標註。"""
    body = re.sub(r"\[片段\d+\]", "", plain(ans))
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    q = q.strip().replace("\n", " ")
    head = f"❓ {q[:60]}{'…' if len(q) > 60 else ''}\n\n🤖 "
    room = limit - len(head)
    if len(body) > room:
        body = body[:room - 1].rstrip() + "…"
    return head + body


def line_check(token: str) -> str:
    """驗證 Channel Access Token，回傳官方帳號名稱。"""
    with ApiClient(Configuration(access_token=token)) as c:
        return MessagingApi(c).get_bot_info().display_name


def line_broadcast(token: str, text: str) -> None:
    """廣播給所有已加入官方帳號的好友。"""
    with ApiClient(Configuration(access_token=token)) as c:
        MessagingApi(c).broadcast(BroadcastRequest(messages=[TextMessage(text=text)]))


# ═══════════════ Streamlit 介面 ═══════════════
st.set_page_config(page_title="RAG 文件問答", page_icon="📚", layout="wide")
ss = st.session_state
if "rag" not in ss:
    ss.rag = RAG()
    ss.messages, ss.records, ss.flash = [], [], []
    ss.models, ss.groq_checked, ss.file_sig, ss.load_msg, ss.load_err = [], None, None, "", ""

for icon, msg in ss.flash:  # 上一輪留下的提示（rerun 後才顯示）
    st.toast(msg, icon=icon)
ss.flash = []


def do_push(q, ans, token):
    if not token:
        ss.flash.append(("⚠️", "請先在側欄貼上 LINE Token"))
        return
    text = format_line(q, ans)
    try:
        line_broadcast(token, text)
        ss.flash.append(("✅", f"已推送到 LINE（{len(text)} 字）"))
    except Exception as e:  # noqa: BLE001
        ss.flash.append(("❌", f"LINE 推送失敗：{type(e).__name__}"))


# ── 側欄 ──
with st.sidebar:
    st.title("📚 RAG 文件問答")
    st.caption("Groq × ChromaDB × LINE")

    st.subheader("🔑 Groq API Key")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_…", label_visibility="collapsed").strip()
    if st.button("驗證 Key", use_container_width=True):
        if groq_key:
            with st.spinner("驗證中…"):
                ss.models = list_models(groq_key)
            ss.groq_checked = (groq_key, bool(ss.models))
        else:
            ss.groq_checked = None
            st.warning("請先貼上 Groq API Key")
    if ss.groq_checked and ss.groq_checked[0] == groq_key:
        st.success(f"✓ Key 有效 · {len(ss.models)} 個模型") if ss.groq_checked[1] else st.error("✗ Key 無效或無法取得模型")
    elif not groq_key:
        st.caption("⚠ 尚未設定 Key")

    models = ss.models or MODELS
    model = st.selectbox("模型", models, index=models.index(MODELS[0]) if MODELS[0] in models else 0)

    st.subheader("📂 文件")
    up = st.file_uploader("上傳 PDF / DOCX", type=["pdf", "docx"], label_visibility="collapsed")
    if up is not None and ss.file_sig != (up.name, up.size):
        with st.spinner("載入嵌入模型、解析並建立索引中…（首次需下載模型，請稍候）"):
            ss.file_sig = (up.name, up.size)
            try:
                with tempfile.TemporaryDirectory() as d:
                    p = Path(d) / Path(up.name).name
                    p.write_bytes(up.getvalue())
                    ss.load_msg, ss.load_err = ss.rag.load(p), ""
            except Exception as e:  # noqa: BLE001
                ss.load_err = str(e) if isinstance(e, ValueError) else f"載入失敗：{type(e).__name__}: {e}"
    elif up is None and ss.rag.doc is not None:  # 使用者移除檔案 → 釋放索引
        ss.rag.drop(ss.rag.doc)
        ss.rag.doc, ss.file_sig, ss.load_msg = None, None, ""
    if ss.load_err and up is not None:
        st.error(f"✗ {ss.load_err}")
    elif ss.rag.doc is not None:
        st.success(ss.load_msg)

    st.subheader("🔎 檢索設定")
    strategy = st.selectbox("檢索策略", list(STRATEGIES), format_func=lambda k: STRATEGIES[k][0])
    st.caption(STRATEGIES[strategy][1])
    top_k = st.slider("片段數量 Top-K", 1, 10, 4)

    st.subheader("💬 LINE Bot")
    line_token = st.text_input("Channel Access Token", type="password", placeholder="貼上 LINE Token").strip()
    if st.button("驗證 Token", use_container_width=True):
        if not line_token:
            st.warning("請先貼上 LINE Token")
        else:
            try:
                with st.spinner("驗證中…"):
                    st.success(f"✓ 已連線：{line_check(line_token)}")
            except Exception:  # noqa: BLE001
                st.error("✗ Token 無效或無法連線")
    auto_push = st.checkbox(f"回答後自動推送（≤{LINE_LIMIT} 字）")
    st.caption("⚠ 推送為廣播：會發給所有加入該官方帳號的好友，並消耗訊息額度。")
    with st.popover("📤 推送最新回答", use_container_width=True):
        st.write("將最新回答廣播給所有好友，確定嗎？")
        if st.button("確定推送", type="primary"):
            if ss.records:
                do_push(ss.records[-1]["問題"], ss.records[-1]["回答"], line_token)
            else:
                ss.flash.append(("ℹ️", "目前沒有可推送的回答"))
            st.rerun()

    st.subheader("⬇️ 匯出問答紀錄")
    with_src = st.checkbox("Word 包含參考片段", value=True)
    if ss.records:
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        st.download_button("📄 下載 Word", docx_bytes(ss.records, with_src), f"RAG問答紀錄_{stamp}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        st.download_button("📊 下載 CSV", csv_bytes(ss.records), f"RAG問答紀錄_{stamp}.csv", "text/csv", use_container_width=True)
    else:
        st.caption("尚無問答紀錄")
    if st.button("🗑 清除對話", use_container_width=True):
        ss.messages, ss.records = [], []
        st.rerun()

# ── 主區：對話 ──
st.title("📚 RAG 文件問答")
if ss.rag.doc is None:
    st.info("請先在左側上傳 PDF / DOCX，並貼上自己的 Groq API Key，即可開始提問。")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("timing"):
            st.caption(m["timing"])
        if m.get("src"):
            with st.expander("🔎 檢索片段"):
                st.text(m["src"])

if prompt := st.chat_input("輸入問題…"):
    if ss.rag.doc is None:
        st.warning("請先上傳 PDF 或 DOCX 文件。")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("思考中…"):
            try:
                ans, src, timing = ss.rag.answer(prompt, strategy, top_k, model, groq_key)
            except Exception as e:  # noqa: BLE001
                ans, src, timing = f"✗ {type(e).__name__}: {e}", "", ""
        ss.messages += [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": ans, "src": src, "timing": timing}]
        if timing:  # timing 非空 = 模型實際生成的回答
            ss.records.append({"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "文件": ss.rag.doc.name, "問題": prompt,
                               "回答": ans, "策略": STRATEGIES[strategy][0], "模型": model,
                               "參考片段": src.split("─" * 40, 1)[-1].strip()})
            if auto_push:
                do_push(prompt, ans, line_token)
        st.rerun()
