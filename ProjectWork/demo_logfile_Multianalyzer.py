import os
import asyncio
import logging
import argparse
from pathlib import Path
import json
import asyncio
import yaml
from pathlib import Path

from fairlib.utils.document_processor import DocumentProcessor
from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    ToolExecutor,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent, 
    SimpleReActPlanner,
    RoleDefinition,
    AbstractTool
)

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------- Optional: transformers import (lazy-safe) ----------------
try:
    from transformers import pipeline as hf_pipeline
except Exception as e:
    hf_pipeline = None
    logger.warning("transformers not available; hierarchical DeBERTa tool will raise on use. %s", e)

# ---------------- Tools ----------------

class logReader2ElectricBoogaloo(AbstractTool):
    """
    Reads the first ~500 lines of a log file and returns them as a string.
    """
    name = "logReader"
    description = "Read the first ~500 lines from a log file at the given path."

    def use(self, tool_input: str) -> str:
        logfile = open(tool_input.replace("'", ""), 'r', encoding="utf-8", errors="ignore")
        try:
            log_lines = logfile.readlines()[:10]
        finally:
            logfile.close()
        return "\n".join(log_lines)


class HierarchicalDebertaAnalyzer(AbstractTool):
    """
    Hierarchical zero-shot classifier for Linux security logs using DeBERTa-v3.
    - Top-level 12 categories (mutually exclusive pass).
    - Optional fine-grained refinement within the chosen category if confidence >= threshold.
    Input: raw log text (string) or JSON envelope:
        {"text": "...", "conf_threshold": 0.30, "max_chars": 4000}
    Output: JSON with top-level & refined predictions (+ full HF results if needed).
    """
    name = "deberta_hier"
    description = ("Hierarchical zero-shot classification for Linux security logs. "
                   "Input should be raw log text or JSON with 'text' and optional "
                   "'conf_threshold' and 'max_chars'.")

    _classifier = None
    _model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

    # Top-level labels (exact phrasing you provided)
    _TOP_LEVEL_LABELS = [
        "This Linux security log is an example of Initial Access",
        "This Linux security log is an example of Reconnaissance",
        "This Linux security log is an example of Execution",
        "This Linux security log is an example of Privilege Escalation",
        "This Linux security log is an example of Persistence",
        "This Linux security log is an example of Lateral Movement",
        "This Linux security log is an example of Credential Access",
        "This Linux security log is an example of Defense Evasion",
        "This Linux security log is an example of Collection & Exfiltration",
        "This Linux security log is an example of Command & Control",
        "This Linux security log is an example of Impact / Destruction",
        "This Linux security log is an example of Normal / Benign Activity",
    ]

    # Fine-grained mapping
    _CATEGORY_TO_FINE = {
        "Initial Access": [
            "This Linux security log is an example of remote SSH login",
            "This Linux security log is an example of unauthorized login attempt",
            "This Linux security log is an example of login from an unknown IP address",
            "This Linux security log is an example of brute-force login attempt",
            "This Linux security log is an example of successful authentication from suspicious host",
            "This Linux security log is an example of login using stolen credentials",
        ],
        "Reconnaissance": [
            "This Linux security log is an example of network scan for open ports",
            "This Linux security log is an example of service enumeration",
            "This Linux security log is an example of user enumeration attempt",
            "This Linux security log is an example of system information enumeration",
            "This Linux security log is an example of listing running processes",
            "This Linux security log is an example of enumerating installed packages",
        ],
        "Execution": [
            "This Linux security log is an example of new process creation",
            "This Linux security log is an example of execution of a suspicious command",
            "This Linux security log is an example of execution of shell script",
            "This Linux security log is an example of unauthorized binary execution",
            "This Linux security log is an example of execution of encoded or obfuscated command",
            "This Linux security log is an example of execution of reverse shell",
        ],
        "Privilege Escalation": [
            "This Linux security log is an example of privilege escalation",
            "This Linux security log is an example of failed attempt to escalate privileges",
            "This Linux security log is an example of user added to sudoers group",
            "This Linux security log is an example of root shell access granted",
            "This Linux security log is an example of use of the su command",
            "This Linux security log is an example of execution of process with root privileges",
        ],
        "Persistence": [
            "This Linux security log is an example of new cron job created",
            "This Linux security log is an example of startup script modified",
            "This Linux security log is an example of malicious service installed",
            "This Linux security log is an example of unauthorized background service started",
            "This Linux security log is an example of system autostart modified",
        ],
        "Lateral Movement": [
            "This Linux security log is an example of remote connection established",
            "This Linux security log is an example of SMB or SSH connection between internal hosts",
            "This Linux security log is an example of RDP or VNC session opened",
            "This Linux security log is an example of use of remote management tools",
            "This Linux security log is an example of connection to internal host via SSH key reuse",
        ],
        "Credential Access": [
            "This Linux security log is an example of credential dumping activity",
            "This Linux security log is an example of access to /etc/shadow or /etc/passwd",
            "This Linux security log is an example of password changed for another user",
            "This Linux security log is an example of user password brute-forced successfully",
            "This Linux security log is an example of use of password-stealing utilities",
        ],
        "Defense Evasion": [
            "This Linux security log is an example of log file deletion",
            "This Linux security log is an example of disabling auditd or syslog service",
            "This Linux security log is an example of clearing bash history",
            "This Linux security log is an example of tampering with firewall rules",
            "This Linux security log is an example of disabling SELinux or AppArmor",
            "This Linux security log is an example of masking process names or hiding files",
        ],
        "Collection & Exfiltration": [
            "This Linux security log is an example of data exfiltration attempt",
            "This Linux security log is an example of large outbound file transfer",
            "This Linux security log is an example of unauthorized SFTP upload",
            "This Linux security log is an example of sensitive file accessed by unauthorized user",
            "This Linux security log is an example of network connection to known exfiltration endpoint",
        ],
        "Command & Control": [
            "This Linux security log is an example of reverse shell connection",
            "This Linux security log is an example of beaconing to external command-and-control server",
            "This Linux security log is an example of periodic outbound HTTP or DNS communication",
            "This Linux security log is an example of connection to suspicious external IP",
        ],
        "Impact / Destruction": [
            "This Linux security log is an example of file deletion by unauthorized process",
            "This Linux security log is an example of data encryption event (possible ransomware)",
            "This Linux security log is an example of filesystem wipe or overwrite",
            "This Linux security log is an example of system reboot initiated unexpectedly",
            "This Linux security log is an example of kernel panic triggered intentionally",
            "This Linux security log is an example of denial of service via resource exhaustion",
        ],
        "Normal / Benign Activity": [
            "This Linux security log is an example of system startup",
            "This Linux security log is an example of scheduled system update",
            "This Linux security log is an example of user login during work hours",
            "This Linux security log is an example of file opened by authorized user",
            "This Linux security log is an example of expected network connection to trusted host",
            "This Linux security log is an example of normal background service activity",
        ],
    }

    _DEFAULT_CONF_THRESH = 0.30
    _DEFAULT_MAX_CHARS = 4000  # keep under typical ZS sequence limits

    @classmethod
    def _get_classifier(cls):
        if hf_pipeline is None:
            raise RuntimeError(
                "transformers is not installed. Install with: pip install transformers sentencepiece accelerate torch torchvision"
            )
        if cls._classifier is None:
            cls._classifier = hf_pipeline("zero-shot-classification", model=cls._model_name)
        return cls._classifier

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if text is None:
            return ""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        # keep head and tail to preserve context
        head = text[: max_chars // 2]
        tail = text[-(max_chars // 2) :]
        return head + "\n...\n" + tail

    @staticmethod
    def _extract_category_name(top_label: str) -> str:
        # Expect: "This Linux security log is an example of Privilege Escalation"
        try:
            return top_label.split(" of ", 1)[1].strip()
        except Exception:
            return top_label

    def _classify_top(self, text: str):
        clf = self._get_classifier()
        res = clf(text, candidate_labels=self._TOP_LEVEL_LABELS, multi_label=False)
        return res["labels"][0], float(res["scores"][0]), res

    def _refine(self, text: str, chosen_top_label: str):
        cat_name = self._extract_category_name(chosen_top_label)
        fine_labels = self._CATEGORY_TO_FINE.get(cat_name)
        if not fine_labels:
            return None, None, None
        clf = self._get_classifier()
        res = clf(text, candidate_labels=fine_labels, multi_label=False)
        return res["labels"][0], float(res["scores"][0]), res

    def use(self, tool_input: str) -> str:
        # Accept raw text or JSON envelope
        text = tool_input
        conf_thresh = self._DEFAULT_CONF_THRESH
        max_chars = self._DEFAULT_MAX_CHARS

        if tool_input and tool_input.strip().startswith("{"):
            try:
                payload = json.loads(tool_input)
                text = payload.get("text", "")
                conf_thresh = float(payload.get("conf_threshold", conf_thresh))
                max_chars = int(payload.get("max_chars", max_chars))
            except Exception:
                text = tool_input  # fall back to raw

        if not text:
            return json.dumps({"error": "No text provided to hierarchical analyzer."})

        text = self._truncate(text, max_chars)

        top_label, top_score, top_res = self._classify_top(text)
        refined = None

        if top_score >= conf_thresh:
            fine_label, fine_score, fine_res = self._refine(text, top_label)
            if fine_label is not None:
                refined = {
                    "label": fine_label,
                    "score": fine_score,
                    "full_result": fine_res
                }

        out = {
            "top_level": {
                "label": top_label,
                "score": top_score,
                "full_result": top_res
            },
            "refined": refined
        }
        return json.dumps(out, ensure_ascii=False, default=str)

# ---------------- Agent Setup & Loop ----------------

async def main():
    print("Initializing a single agent for demonstration...")

    # Brain (LLM)
    llm = HuggingFaceAdapter("mistral-7b")

    # Toolbelt
    tool_registry = ToolRegistry()
    reading_tool = logReader2ElectricBoogaloo()
    analyzer_tool = HierarchicalDebertaAnalyzer()
    tool_registry.register_tool(reading_tool)
    tool_registry.register_tool(analyzer_tool)

    print(f"Agent's tools: {[tool.name for tool in tool_registry.get_all_tools().values()]}")

    # Hands
    executor = ToolExecutor(tool_registry)

    # Memory
    memory = WorkingMemory()

    # Mind
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a system monitoring expert. Identify any errors, anomalies, or alerts in a system log file. "
        "Extract error codes, timestamps, and descriptions if present. "
        "When asked to categorize activity, call the 'deberta_hier' tool with the log text."
    )

    # Agent
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=10
    )
    print("Agent successfully created. You can now chat with the agent.")
    print("Provide it a tool to use (logReader), and a file path, after the file is loaded ask it a question. Type 'exit' to quit.")
    print("New: type `classify <path/to/log>` for hierarchical zero-shot classification.")

    # Helper: run the hierarchical pipeline on a file path
    async def classify_log_file(path: str, conf_threshold: float = 0.30, max_chars: int = 4000):
        # 1) Read ~500 lines using logReader
        text = reading_tool.use(path)
        # 2) Feed into hierarchical analyzer with your exact labels
        payload = {
            "text": text,
            "conf_threshold": conf_threshold,
            "max_chars": max_chars
        }
        out_json = analyzer_tool.use(json.dumps(payload))
        # 3) Pretty-print
        try:
            data = json.loads(out_json)
            tl = data.get("top_level", {}) or {}
            rf = data.get("refined")
            print("🔎 Hierarchical DeBERTa Classification")
            print(f"Top-level: {tl.get('label')} (score: {tl.get('score'):.4f})")
            if rf:
                print(f"Refined:   {rf.get('label')} (score: {rf.get('score'):.4f})")
            else:
                print("Refined:   <none> (top-level below threshold or no mapping)")
            print("\nCompact JSON:")
            compact = {
                "top_level_label": tl.get("label"),
                "top_level_score": tl.get("score"),
                "refined_label": rf.get("label") if rf else None,
                "refined_score": rf.get("score") if rf else None
            }
            print(json.dumps(compact, indent=2))
        except Exception:
            print(out_json)

    # --- Interaction Loop ---
    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Agent: Goodbye!")
                break

            # Quick command: classify <path> [threshold] [max_chars]
            # e.g., classify /var/log/auth.log 0.35 6000
            if user_input.lower().startswith("classify "):
                parts = user_input.split()
                file_path = parts[1]
                thr = float(parts[2]) if len(parts) > 2 else 0.30
                mx = int(parts[3]) if len(parts) > 3 else 4000
                await classify_log_file(file_path, thr, mx)
                continue

            # Normal agent turn
            agent_response = await agent.arun(user_input)
            print(f"LLM Raw Output:\n{agent_response}")
            print(f"🤖 Agent: {agent_response}")

        except KeyboardInterrupt:
            print("\n🤖 Agent: Exiting...")
            break

if __name__ == "__main__":
    asyncio.run(main())
