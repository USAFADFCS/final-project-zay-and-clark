import os
import json
import asyncio
import logging
import re
from typing import List

from dotenv import load_dotenv
load_dotenv()

from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    SimpleReActPlanner,
    SimpleAgent,
    RoleDefinition,
    AbstractTool
)
try:
    from transformers import pipeline as hf_pipeline
except Exception as e:
    hf_pipeline = None
    logging.getLogger(__name__).warning("transformers not available, DeBERTa tool will raise: %s", e)

class LogReader(AbstractTool):
    name = "logReader"
    description = "Read the first N lines from a log file. Input is a path, or 'path::N'. Returns plain text."

    def use(self, tool_input: str) -> str:
        if tool_input and not tool_input.strip().startswith("{"):
            # Allow simple "path" or "path::N"
            raw = tool_input.strip().strip("'").strip('"')
            parts = raw.split("::")
            path = parts[0]
            try:
                n = int(parts[1]) if len(parts) > 1 else 500
            except Exception:
                n = 500
        else:
            try:
                cfg = json.loads(tool_input)
                path = cfg.get("path")
                n = int(cfg.get("n", 500))
            except Exception:
                return json.dumps({"error": "Expected 'path' or JSON with {'path': ..., 'n': 500}."})

        if not path or not os.path.exists(path):
            return json.dumps({"error": f"Path not found: {path}"})

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[:n]
        return "".join(lines)


class HierarchicalDebertaAnalyzer(AbstractTool):
    name = "deberta_hier"
    description = "Hierarchical zero-shot classification for Linux security logs."

    _classifier = None
    _model_name = os.getenv("HF_ZSC_REPO", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

    _HYPOTHESIS_TOP = "This Linux security log indicates {}."
    _HYPOTHESIS_FINE = "This Linux security log shows {}."

    _TOP_LEVEL_LABELS = [
        "initial access",
        "reconnaissance",
        "execution",
        "privilege escalation",
        "persistence",
        "lateral movement",
        "credential access",
        "defense evasion",
        "collection",
        "exfiltration",
        "command and control",
        "impact or destruction",
        "benign activity",
    ]

    _CATEGORY_TO_FINE = {
        "initial access": [
            "remote ssh login",
            "unauthorized login attempt",
            "login from unknown ip",
            "brute force login attempt",
            "successful authentication from suspicious host",
            "login using stolen credentials",
        ],
        "reconnaissance": [
            "network scan for open ports",
            "service enumeration",
            "user enumeration attempt",
            "system information enumeration",
            "listing running processes",
            "enumerating installed packages",
        ],
        "execution": [
            "new process creation",
            "execution of suspicious command",
            "execution of shell script",
            "unauthorized binary execution",
            "execution of encoded or obfuscated command",
            "execution of reverse shell",
        ],
        "privilege escalation": [
            "privilege escalation attempt",
            "failed attempt to escalate privileges",
            "user added to sudoers group",
            "root shell access granted",
            "use of su command",
            "execution of process with root privileges",
        ],
        "persistence": [
            "new cron job created",
            "startup script modified",
            "malicious service installed",
            "unauthorized background service started",
            "system autostart modified",
        ],
        "lateral movement": [
            "remote connection established",
            "ssh connection between internal hosts",
            "rdp or vnc session opened",
            "use of remote management tools",
            "connection to internal host via ssh key reuse",
        ],
        "credential access": [
            "credential dumping activity",
            "access to etc shadow or etc passwd",
            "password changed for another user",
            "user password brute forced successfully",
            "use of password stealing utilities",
        ],
        "defense evasion": [
            "log file deletion",
            "disabling auditd or syslog service",
            "clearing bash history",
            "tampering with firewall rules",
            "disabling selinux or apparmor",
            "masking process names or hiding files",
        ],
        "collection": [
            "sensitive file accessed by unauthorized user",
            "archiving data for exfiltration",
            "collecting credentials and tokens",
        ],
        "exfiltration": [
            "data exfiltration attempt",
            "large outbound file transfer",
            "unauthorized sftp upload",
            "connection to known exfiltration endpoint",
        ],
        "command and control": [
            "reverse shell connection",
            "beaconing to external command and control server",
            "periodic outbound http or dns communication",
            "connection to suspicious external ip",
        ],
        "impact or destruction": [
            "file deletion by unauthorized process",
            "data encryption event possible ransomware",
            "filesystem wipe or overwrite",
            "system reboot initiated unexpectedly",
            "kernel panic triggered intentionally",
            "denial of service via resource exhaustion",
        ],
        "benign activity": [
            "system startup",
            "scheduled system update",
            "user login during work hours",
            "file opened by authorized user",
            "expected network connection to trusted host",
            "normal background service activity",
        ],
    }

    _DEFAULT_CONF_THRESH = 0.30
    _DEFAULT_MAX_CHARS = 4000
    _DEFAULT_MULTI_LABEL = True

    @classmethod
    def _get_classifier(cls):
        if hf_pipeline is None:
            raise RuntimeError("transformers not installed. Install transformers, sentencepiece, accelerate, torch")
        if cls._classifier is None:
            cls._classifier = hf_pipeline("zero-shot-classification", model=cls._model_name)
        return cls._classifier

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if not text:
            return ""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        head = text[: max_chars // 2]
        tail = text[-(max_chars // 2):]
        return head + "\n...\n" + tail

    def _classify_top(self, text: str, multi_label: bool):
        clf = self._get_classifier()
        res = clf(
            text,
            candidate_labels=self._TOP_LEVEL_LABELS,
            hypothesis_template=self._HYPOTHESIS_TOP,
            multi_label=multi_label,
        )
        labels = res["labels"]
        scores = res["scores"]
        top_idx = 0
        if multi_label and len(labels) > 1:
            best_score = scores[0]
            for i, (lab, sc) in enumerate(zip(labels, scores)):
                if lab != "benign activity" and sc >= best_score - 0.05:
                    top_idx = i
                    break
        return labels[top_idx], float(scores[top_idx]), res

    def _refine(self, text: str, chosen_top_label: str):
        fine_labels = self._CATEGORY_TO_FINE.get(chosen_top_label.lower())
        if not fine_labels:
            return None, None, None
        clf = self._get_classifier()
        res = clf(
            text,
            candidate_labels=fine_labels,
            hypothesis_template=self._HYPOTHESIS_FINE,
            multi_label=False,
        )
        return res["labels"][0], float(res["scores"][0]), res

    def use(self, tool_input: str) -> str:
        text = tool_input
        conf_thresh = self._DEFAULT_CONF_THRESH
        max_chars = self._DEFAULT_MAX_CHARS
        multi_label_top = self._DEFAULT_MULTI_LABEL
        if tool_input and tool_input.strip().startswith("{"):
            try:
                payload = json.loads(tool_input)
                text = payload.get("text", "")
                conf_thresh = float(payload.get("conf_threshold", conf_thresh))
                max_chars = int(payload.get("max_chars", max_chars))
                multi_label_top = bool(payload.get("multi_label_top", multi_label_top))
            except Exception:
                text = tool_input
        if not text:
            return json.dumps({"error": "No text provided"})
        text = self._truncate(text, max_chars)

        top_label, top_score, top_res = self._classify_top(text, multi_label_top)
        refined = None
        if top_label.lower() != "benign activity" and top_score >= conf_thresh:
            fine_label, fine_score, fine_res = self._refine(text, top_label)
            if fine_label is not None:
                refined = {"label": fine_label, "score": fine_score, "full_result": fine_res}

        out = {
            "top_level": {"label": top_label, "score": top_score, "full_result": top_res},
            "refined": refined,
        }
        return json.dumps(out, ensure_ascii=False)


class HierarchicalDebertaBatchAnalyzer(AbstractTool):
    name = "deberta_batch"
    description = "Chunk a big log and run hierarchical ZSC per chunk, then aggregate."

    # Benign heuristics for batch mode
    KNOWN_BENIGN_REGEX = [
        r"CRON\[\d+\]: \(root\) CMD \(run-parts /etc/cron\.daily\)",
        r"systemd\[1\]: (Starting|Started|Finished) (Rotate log files|Daily apt download activities|Daily man-db regeneration)",
        r"systemd\[1\]: man-db\.service: Succeeded\.",
        r"NetworkManager\[\d+\]: .*state change: activated -> activated.*\(reason 'refresh'\)",
        r"man-db\[\d+\]: Building manual page index",
        r"sshd\[\d+\]: Accepted publickey for \w+ from (10\.|192\.168\.|172\.(1[6-9]|2\d|3[0-1])\.)",
        r"sudo\[\d+\]:\s+\w+\s+: .* COMMAND=/usr/bin/apt\b",
        r"sudo\[\d+\]: pam_unix\(sudo:session\): session (opened|closed) for user root",
        r"kernel: \[\s*\d+\.\d+\] iwlwifi .* Unhandled alg: 0x[0-9a-fA-F]+",
    ]
    BENIGN_MAJORITY_RATIO = 0.6
    BENIGN_MARGIN = 0.10

    def __init__(self):
        super().__init__()
        self.single = HierarchicalDebertaAnalyzer()

    @classmethod
    def _chunk_is_benign(cls, lines):
        if not lines:
            return False
        hits = 0
        for ln in lines:
            for rx in cls.KNOWN_BENIGN_REGEX:
                if re.search(rx, ln):
                    hits += 1
                    break
        return hits / max(1, len(lines)) >= cls.BENIGN_MAJORITY_RATIO

    @staticmethod
    def _windows(lines: List[str], chunk_lines: int, stride_lines: int, max_chunks=None):
        i = 0
        n = len(lines)
        count = 0
        while i < n:
            j = min(n, i + chunk_lines)
            yield (i, j, "".join(lines[i:j]))
            i += max(1, chunk_lines - stride_lines)
            count += 1
            if max_chunks and count >= max_chunks:
                break

    def use(self, tool_input: str) -> str:
        if not hasattr(self, "single") or self.single is None:
            self.single = HierarchicalDebertaAnalyzer()

        try:
            cfg = json.loads(tool_input)
            path = cfg.get("path")
            conf_thr = float(cfg.get("conf_threshold", 0.30))
            max_chars = int(cfg.get("max_chars", 4000))
            chunk_lines = int(cfg.get("chunk_lines", 200))
            stride_lines = int(cfg.get("stride_lines", 50))
            max_chunks = cfg.get("max_chunks")
            min_score = float(cfg.get("min_score", 0.55))
            benign_floor = float(cfg.get("benign_floor", 0.40))
            multi_label_top = bool(cfg.get("multi_label_top", True))
        except Exception:
            return json.dumps({"error": "Expected JSON with at least {'path': ...}."})

        if not path or not os.path.exists(path):
            return json.dumps({"error": f"Path not found: {path}"})

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()

        events = []
        tally_top = {}
        tally_ref = {}
        alt_scores = {}

        suspicious_count = 0
        benign_count = 0

        for (i, j, text) in self._windows(all_lines, chunk_lines, stride_lines, max_chunks):
            lines = text.splitlines()

            force_benign = self._chunk_is_benign(lines)

            payload = {
                "text": text[:max_chars],
                "conf_threshold": conf_thr,
                "max_chars": max_chars,
                "multi_label_top": multi_label_top,
            }
            out_json = self.single.use(json.dumps(payload))
            try:
                data = json.loads(out_json)
            except Exception:
                continue

            tl = data.get("top_level") or {}
            rf = data.get("refined")
            top_label = (tl.get("label") or "").lower()
            top_score = float(tl.get("score") or 0.0)

            fr = tl.get("full_result") or {}
            labels = [l.lower() for l in fr.get("labels", [])]
            scores = fr.get("scores", [])
            label2score = {lab: float(sc) for lab, sc in zip(labels, scores)} if labels and scores else {}

            best_lab = top_label
            best_sc = top_score
            benign_sc = label2score.get("benign activity")

            decide_benign = False
            if force_benign:
                decide_benign = True
                best_lab, best_sc = "benign activity", 0.99
            elif benign_sc is not None:
                if (best_lab != "benign activity" and (best_sc - benign_sc) <= self.BENIGN_MARGIN) or (benign_sc >= benign_floor):
                    decide_benign = True
                    best_lab, best_sc = "benign activity", benign_sc

            if decide_benign:
                benign_count += 1
                tally_top["benign activity"] = tally_top.get("benign activity", 0) + 1
            else:
                if best_sc >= min_score:
                    suspicious_count += 1
                    tally_top[best_lab] = tally_top.get(best_lab, 0) + 1

                    if rf and rf.get("label"):
                        rlab = (rf["label"] or "").lower()
                        rsc = float(rf.get("score") or 0.0)
                        if rsc >= min_score:
                            tally_ref[rlab] = tally_ref.get(rlab, 0) + 1
                            alt_scores[rlab] = max(alt_scores.get(rlab, 0.0), rsc)

                    events.append({
                        "range": [i, j],
                        "top_level": {"label": best_lab, "score": best_sc},
                        "refined": rf
                    })

            if label2score:
                for lab, sc in label2score.items():
                    alt_scores[lab] = max(alt_scores.get(lab, 0.0), sc)

        denom = max(1, benign_count + suspicious_count)
        suspicious_rate = suspicious_count / float(denom)

        health = "healthy"
        if suspicious_rate >= 0.35 or any("impact" in k or "command and control" in k for k in tally_top):
            health = "compromised"
        elif suspicious_rate >= 0.10:
            health = "needs further analysis"

        alternatives = [{"label": lab, "score": sc} for lab, sc in sorted(alt_scores.items(), key=lambda x: -x[1])]

        result = {
            "summary": {
                "total_chunks": denom,
                "suspicious_rate": round(suspicious_rate, 4),
                "health": health,
            },
            "top_level_counts": tally_top,
            "refined_counts": tally_ref,
            "events": events,
            "alternatives": alternatives,
        }
        return json.dumps(result, ensure_ascii=False)




class AdvisorTool(AbstractTool):
    name = "advisor"
    description = "Grounded mitigation advice using a simple local retriever over .txt files."

    class _Msg:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content
        def to_dict(self):
            return {"role": self.role, "content": self.content}

    def __init__(self, llm_for_advice=None):
        self.llm = llm_for_advice or HuggingFaceAdapter("mistral-7b")

    @staticmethod
    def _read_kb(kb_dir):
        docs = []
        if not kb_dir or not os.path.isdir(kb_dir):
            return docs
        for root, _, files in os.walk(kb_dir):
            for fn in files:
                if fn.lower().endswith(".txt"):
                    p = os.path.join(root, fn)
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            docs.append((p, f.read()))
                    except Exception:
                        pass
        return docs

    @staticmethod
    def _score_para(para, queries):
        text = para.lower()
        score = 0
        for q in queries:
            ql = q.lower()
            if ql in text:
                score += 3
            score += text.count(ql)
        return score

    @staticmethod
    def _top_paragraphs(docs, queries, k):
        paras = []
        for path, txt in docs:
            for para in txt.split("\n\n"):
                sc = AdvisorTool._score_para(para, queries)
                if sc > 0:
                    paras.append((sc, path, para.strip()))
        paras.sort(key=lambda x: -x[0])
        return paras[:k]

    def use(self, tool_input: str) -> str:
        try:
            cfg = json.loads(tool_input)
        except Exception:
            return json.dumps({"error": "advisor expects JSON input"})

        analysis = cfg.get("analysis_json") or {}
        kb_dir = cfg.get("kb_dir")
        k = int(cfg.get("k", 6))

        tl = analysis.get("top_level_counts", {}) or {}
        rf = analysis.get("refined_counts", {}) or {}
        queries = list({*tl.keys(), *rf.keys()})

        docs = self._read_kb(kb_dir)
        support = self._top_paragraphs(docs, queries, k)

        ctx_blocks = "\n\n".join(
            f"[{i+1}] {os.path.basename(p)}:\n{para[:1200]}"
            for i, (_, p, para) in enumerate(support)
        )

        system = (
            "You are a cybersecurity incident advisor. "
            "Use the provided context to produce practical mitigations. "
            "Return strict JSON with keys: determination, justification, recommendations, sources."
        )
        user = (
            f"Findings summary:\n{json.dumps(analysis.get('summary'), indent=2)}\n\n"
            f"Top-level counts:\n{json.dumps(tl, indent=2)}\n\n"
            f"Refined counts:\n{json.dumps(rf, indent=2)}\n\n"
            f"Context snippets:\n{ctx_blocks}\n\n"
            "Return only JSON. determination must be one of: compromised, needs further analysis, healthy. "
            "recommendations must be a list of concrete steps. sources must be the list of filenames you used."
        )

        messages = [self._Msg("system", system), self._Msg("user", user)]
        try:
            advice = self.llm.chat(messages)
        except TypeError:
            advice = self.llm.invoke(messages, temperature=0.2)
        return advice

    @staticmethod
    def _score_para(para, queries):
        text = para.lower()
        score = 0
        for q in queries:
            ql = q.lower()
            if ql in text:
                score += 3
            score += text.count(ql)
        return score

    @staticmethod
    def _top_paragraphs(docs, queries, k):
        paras = []
        for path, txt in docs:
            for para in txt.split("\n\n"):
                sc = AdvisorTool._score_para(para, queries)
                if sc > 0:
                    paras.append((sc, path, para.strip()))
        paras.sort(key=lambda x: -x[0])
        return paras[:k]

    def use(self, tool_input: str) -> str:
        try:
            cfg = json.loads(tool_input)
        except Exception:
            return json.dumps({"error": "advisor expects JSON input"})

        analysis = cfg.get("analysis_json") or {}
        kb_dir = cfg.get("kb_dir")
        k = int(cfg.get("k", 6))

        tl = analysis.get("top_level_counts", {})
        rf = analysis.get("refined_counts", {})
        queries = list({*tl.keys(), *rf.keys()})
        docs = self._read_kb(kb_dir)
        support = self._top_paragraphs(docs, queries, k)

        ctx_blocks = "\n\n".join([f"[{i+1}] {os.path.basename(p)}:\n{para[:1200]}"
                                  for i, (_, p, para) in enumerate(support)])

        system = (
            "You are a cybersecurity incident advisor. "
            "Use the context to produce practical mitigations. "
            "Return strict JSON with keys: determination, justification, recommendations, sources."
        )
        user = (
            f"Findings summary:\n{json.dumps(analysis.get('summary'), indent=2)}\n\n"
            f"Top-level counts:\n{json.dumps(tl, indent=2)}\n\n"
            f"Refined counts:\n{json.dumps(rf, indent=2)}\n\n"
            f"Context snippets:\n{ctx_blocks}\n\n"
            "Return only JSON. determination must be one of: compromised, needs further analysis, healthy. "
            "recommendations must be a list of concrete steps. sources must be the list of filenames you used."
        )

        msgs = [self._Msg("system", system), self._Msg("user", user)]
        try:
            advice = self.llm.chat(msgs)
        except TypeError:
            advice = self.llm.invoke(msgs, temperature=0.2)
        return advice


class HtmlReportTool(AbstractTool):
    name = "html_report"
    description = "Render an HTML dashboard for health, confidence bars, and recommendations."

    def use(self, tool_input: str) -> str:
        try:
            cfg = json.loads(tool_input)
        except Exception:
            return json.dumps({"error": "html_report expects JSON"})

        analysis = cfg.get("analysis_json") or {}
        advisor = cfg.get("advisor_json") or {}
        title = cfg.get("title", "Classification Output")
        out_path = cfg.get("out_path", "report.html")
        auto_open = bool(cfg.get("auto_open", False))

        from format_agent import Entry, Result, to_html
        alt = analysis.get("alternatives", []) or []
        entries = [Entry(e["label"], float(e["score"])) for e in alt if "label" in e and "score" in e]
        top_entry = entries[0] if entries else None
        refined_entry = entries[1] if len(entries) > 1 else None
        base_html = to_html(Result(top_entry, refined_entry, entries), title=title)

        health = (analysis.get("summary") or {}).get("health", "unknown")
        srate = (analysis.get("summary") or {}).get("suspicious_rate", 0.0)
        try:
            if isinstance(advisor, str):
                advisor = json.loads(advisor)
            recs = advisor.get("recommendations") or []
        except Exception:
            recs = []

        from html import escape
        health_card = f"""
        <div class="card">
          <h3 style="margin:0 0 8px 0">System Health</h3>
          <div class="muted">State: <strong>{escape(str(health)).title()}</strong> • Suspicious rate: {float(srate):.2%}</div>
        </div>
        <div class="card">
          <h3 style="margin:0 0 8px 0">Recommended Actions</h3>
          <ol>{"".join(f"<li>{escape(str(r))}</li>" for r in recs) or "<li class='muted'>No recommendations</li>"}</ol>
        </div>
        """
        inject_anchor = "<div class=\"card\"><h3 style=\"margin:0 0 8px 0\">JSON</h3>"
        html = base_html.replace(inject_anchor, health_card + inject_anchor)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        if auto_open:
            try:
                import webbrowser, pathlib
                webbrowser.open(pathlib.Path(out_path).resolve().as_uri())
            except Exception:
                pass

        return json.dumps({"html_path": out_path, "opened": auto_open})


async def run_full_pipeline(log_path: str, kb_dir: str, html_out: str = "report.html"):
    manager_llm = HuggingFaceAdapter("mistral-7b")
    advisor_llm = HuggingFaceAdapter("mistral-7b")

    tool_registry = ToolRegistry()
    reading_tool = LogReader()
    analyzer_tool = HierarchicalDebertaAnalyzer()
    batch_tool = HierarchicalDebertaBatchAnalyzer()
    advisor_tool = AdvisorTool(advisor_llm)
    html_tool = HtmlReportTool()
    for t in [reading_tool, analyzer_tool, batch_tool, advisor_tool, html_tool]:
        tool_registry.register_tool(t)

    analysis = json.loads(batch_tool.use(json.dumps({
        "path": log_path,
        "conf_threshold": 0.30,
        "chunk_lines": 200,
        "stride_lines": 50,
        "max_chars": 4000,
        "min_score": 0.25,
        "multi_label_top": True
    })))

    advisor = advisor_tool.use(json.dumps({
        "analysis_json": analysis,
        "kb_dir": kb_dir,
        "k": 8
    }))

    html = html_tool.use(json.dumps({
        "analysis_json": analysis,
        "advisor_json": advisor,
        "title": f"System Health - {os.path.basename(log_path)}",
        "out_path": html_out,
        "auto_open": True
    }))
    print("HTML:", html)

    class _Msg:
        def __init__(self, role, content):
            self.role = role
            self.content = content
        def to_dict(self):
            return {"role": self.role, "content": self.content}

    summary_system = (
        "You are a concise incident response summarizer. Do not call tools. "
        "Read the provided JSON and produce exactly three bullets: "
        "1) Overall health and suspicious rate. "
        "2) Top suspicious families with counts. "
        "3) The top three concrete actions to take next."
    )
    summary_user = f"analysis_json={json.dumps(analysis)}\nadvisor_json={advisor}"

    try:
        summary = manager_llm.chat([_Msg("system", summary_system), _Msg("user", summary_user)])
    except TypeError:
        summary = manager_llm.invoke([_Msg("system", summary_system), _Msg("user", summary_user)], temperature=0.2)

    print("Summary:", summary)



async def interactive_loop():
    print("Commands: classify <path> [thr] [max_chars], batch_classify <path> [thr] [chunk] [stride], full_pipeline <log> <kb_dir> [html_out], exit")

    manager_llm = HuggingFaceAdapter("mistral-7b")

    tool_registry = ToolRegistry()
    reading_tool = LogReader()
    analyzer_tool = HierarchicalDebertaAnalyzer()
    batch_tool = HierarchicalDebertaBatchAnalyzer()
    advisor_tool = AdvisorTool(manager_llm)
    html_tool = HtmlReportTool()
    for t in [reading_tool, analyzer_tool, batch_tool, advisor_tool, html_tool]:
        tool_registry.register_tool(t)

    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()
    planner = SimpleReActPlanner(manager_llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a system monitoring expert. Identify errors and anomalies. "
        "Categorize activity using 'deberta_hier' or 'deberta_batch'. "
        "When asked for mitigations, call 'advisor' and then 'html_report'."
    )
    agent = SimpleAgent(llm=manager_llm, planner=planner, tool_executor=executor, memory=memory, max_steps=12)

    async def classify_once(path: str, conf_threshold: float = 0.30, max_chars: int = 4000):
        text = reading_tool.use(path)
        payload = {"text": text, "conf_threshold": conf_threshold, "max_chars": max_chars, "multi_label_top": True}
        out_json = analyzer_tool.use(json.dumps(payload))
        print(out_json)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Bye.")
                break
            if user_input.lower().startswith("classify "):
                parts = user_input.split()
                file_path = parts[1]
                thr = float(parts[2]) if len(parts) > 2 else 0.30
                mx = int(parts[3]) if len(parts) > 3 else 4000
                await classify_once(file_path, thr, mx)
                continue
            if user_input.lower().startswith("batch_classify "):
                parts = user_input.split()
                file_path = parts[1]
                thr = float(parts[2]) if len(parts) > 2 else 0.30
                chunk = int(parts[3]) if len(parts) > 3 else 200
                stride = int(parts[4]) if len(parts) > 4 else 50
                out = HierarchicalDebertaBatchAnalyzer().use(json.dumps({
                    "path": file_path,
                    "conf_threshold": thr,
                    "chunk_lines": chunk,
                    "stride_lines": stride,
                    "multi_label_top": True,
                    "min_score": 0.55,
                    "benign_floor": 0.40
                }))
                print(out)
                continue
            if user_input.lower().startswith("full_pipeline "):
                parts = user_input.split()
                file_path = parts[1]
                kb_dir = parts[2]
                html_out = parts[3] if len(parts) > 3 else "report.html"
                await run_full_pipeline(file_path, kb_dir, html_out)
                continue

            resp = await agent.arun(user_input)
            print(f"Agent: {resp}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full_pipeline":
        if len(sys.argv) < 4:
            print("Usage: python main.py full_pipeline <log> <kb_dir> [html_out]")
            sys.exit(1)
        log_path = sys.argv[2]
        kb_dir = sys.argv[3]
        html_out = sys.argv[4] if len(sys.argv) > 4 else "report.html"
        asyncio.run(run_full_pipeline(log_path, kb_dir, html_out))
    else:
        asyncio.run(interactive_loop())
