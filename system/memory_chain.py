"""SQLite 기반 메모리 체인.

기능:
- 대화 턴 저장(chat_turns)
- LLM 기반 장기기억 후보 추출(memory_candidates)
- 점수 기반 승격(memory_slots)
- 현재 user 발화와 연관된 슬롯 retrieval 후 system 메시지 주입
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re
import sqlite3
from datetime import datetime, timezone

try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.feature_extraction.text import HashingVectorizer
except Exception: 
    HashingVectorizer = None

try:
    import sqlite_vec
except Exception:
    sqlite_vec = None

from system.llm_engine import GenerationConfig, QwenEngine


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _tokenize_ko_en(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[가-힣A-Za-z0-9_]{2,}", text.lower()))


@dataclass
class SummaryMemoryConfig:
    """메모리 체인 설정."""

    enabled: bool = True
    update_every_turns: int = 1
    max_summary_chars: int = 900
    db_path: str | None = None
    session_id: str = "default"
    recent_turn_window: int = 6
    retrieval_limit: int = 8
    promote_threshold: float = 0.65
    vector_enabled: bool = True
    vector_dim: int = 1024


class SummaryMemoryChain:
    """장기기억 후보 추출 + 점수 승격 + retrieval."""

    def __init__(self, llm_engine: QwenEngine, config: SummaryMemoryConfig | None = None) -> None:
        self.llm_engine = llm_engine
        self.config = config or SummaryMemoryConfig()
        self.summary_text: str = ""
        self.turn_count: int = 0

        project_root = Path(__file__).resolve().parents[1]
        db_default = project_root / "outputs" / "memory" / "memory.sqlite3"
        self.db_path = Path(self.config.db_path) if self.config.db_path else db_default
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Gradio/FastAPI worker thread 간에 동일 RuntimeServices 인스턴스를 공유하므로
        # SQLite 연결은 thread check를 비활성화해 재사용 가능하게 둔다.
        # 실제 동시 접근은 상위(RuntimeServices._turn_lock)에서 직렬화한다.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.vector_dim = int(self.config.vector_dim)
        self._sqlite_vec_loaded = False
        if sqlite_vec is not None:
            try:
                sqlite_vec.load(self.conn)
                self._sqlite_vec_loaded = True
            except Exception:
                self._sqlite_vec_loaded = False
        self.vector_enabled = bool(
            self.config.vector_enabled
            and np is not None
            and HashingVectorizer is not None
            and self._sqlite_vec_loaded
        )
        self._vectorizer = None
        if self.vector_enabled:
            self._vectorizer = HashingVectorizer(
                n_features=self.vector_dim,
                alternate_sign=False,
                norm=None,
                ngram_range=(1, 2),
            )
            self._init_vec_table()
        print(
            f"[memory] db={self.db_path} vector_enabled={self.vector_enabled} "
            f"sqlite_vec_loaded={self._sqlite_vec_loaded} dim={self.vector_dim}"
        )

        self.extract_gen = GenerationConfig(
            max_new_tokens=420,
            temperature=0.2,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.02,
            no_repeat_ngram_size=3,
            do_sample=False,
            use_cache=True,
        )

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_idx INTEGER NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                ts TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_idx INTEGER NOT NULL,
                type TEXT NOT NULL,
                subject TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence REAL NOT NULL,
                future_impact REAL NOT NULL,
                emotion_intensity REAL NOT NULL,
                recurrence REAL NOT NULL,
                novelty REAL NOT NULL,
                score REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_slots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                type TEXT NOT NULL,
                subject TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                score REAL NOT NULL,
                last_evidence_turn INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, key)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_session_turn ON chat_turns(session_id, turn_idx)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidates_session_key ON memory_candidates(session_id, key)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_slots_session_score ON memory_slots(session_id, score DESC)"
        )
        self.conn.commit()

    def _init_vec_table(self) -> None:
        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_slot_vec
            USING vec0(embedding float[{self.vector_dim}])
            """
        )
        self.conn.commit()

    def _embed_text(self, text: str) -> bytes | None:
        if not self.vector_enabled or self._vectorizer is None or np is None:
            return None
        try:
            vec = self._vectorizer.transform([text]).toarray()[0].astype("float32")
        except Exception:
            return None
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        if sqlite_vec is not None:
            try:
                return sqlite_vec.serialize_float32(vec.tolist())
            except Exception:
                return None
        return None

    def _upsert_slot_vector(self, slot_id: int, text: str) -> None:
        blob = self._embed_text(text)
        if blob is None:
            return
        # memory_slots.id 를 vec rowid로 사용
        self.conn.execute(
            "DELETE FROM memory_slot_vec WHERE rowid = ?",
            (slot_id,),
        )
        self.conn.execute(
            "INSERT INTO memory_slot_vec(rowid, embedding) VALUES (?, ?)",
            (slot_id, blob),
        )

    def _next_turn_idx(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(turn_idx), 0) AS max_turn FROM chat_turns WHERE session_id = ?",
            (self.config.session_id,),
        ).fetchone()
        return int(row["max_turn"]) + 1

    def _insert_turn_pair(self, turn_idx: int, user_text: str, assistant_text: str) -> None:
        now = _utc_now()
        self.conn.executemany(
            """
            INSERT INTO chat_turns(session_id, turn_idx, role, text, ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (self.config.session_id, turn_idx, "user", user_text.strip(), now),
                (self.config.session_id, turn_idx, "assistant", assistant_text.strip(), now),
            ],
        )
        self.conn.commit()

    def _extract_candidates(self, user_text: str, assistant_text: str) -> list[dict[str, Any]]:
        prompt = f"""너는 대화 장기기억 추출기다.
이번 user/assistant 교환에서 장기 보존 가치가 있는 항목만 JSON으로 추출하라.
출력 스키마:
{{
  "candidates": [
    {{
      "type": "preference|fact|promise|relationship|emotion|plan|boundary",
      "subject": "user|assistant|pair",
      "key": "짧은_식별자",
      "value": "핵심 내용",
      "confidence": 0.0,
      "future_impact": 0.0,
      "emotion_intensity": 0.0
    }}
  ]
}}
규칙:
- 당장 다음 대화나 장기 관계에 영향이 없는 잡담은 제외.
- confidence/future_impact/emotion_intensity는 0~1 실수.
- JSON 객체만 출력.

[user]
{user_text.strip()}

[assistant]
{assistant_text.strip()}
"""
        msgs = [
            {"role": "system", "content": "출력은 JSON 객체 하나만."},
            {"role": "user", "content": prompt},
        ]
        out = self.llm_engine.generate(self.llm_engine.build_prompt(msgs), gen_config=self.extract_gen)
        if not out:
            return []
        m = re.search(r"\{[\s\S]*\}", out)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            return []
        arr = parsed.get("candidates")
        if not isinstance(arr, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for c in arr:
            if not isinstance(c, dict):
                continue
            t = str(c.get("type", "")).strip()
            s = str(c.get("subject", "")).strip()
            k = str(c.get("key", "")).strip()
            v = str(c.get("value", "")).strip()
            if not (t and s and k and v):
                continue
            cleaned.append(
                {
                    "type": t[:40],
                    "subject": s[:24],
                    "key": k[:120],
                    "value": v[:400],
                    "confidence": min(1.0, max(0.0, _safe_float(c.get("confidence"), 0.0))),
                    "future_impact": min(1.0, max(0.0, _safe_float(c.get("future_impact"), 0.0))),
                    "emotion_intensity": min(
                        1.0, max(0.0, _safe_float(c.get("emotion_intensity"), 0.0))
                    ),
                }
            )
        return cleaned

    def _type_importance(self, memory_type: str) -> float:
        mapping = {
            "promise": 1.0,
            "boundary": 0.95,
            "fact": 0.85,
            "relationship": 0.80,
            "plan": 0.75,
            "preference": 0.65,
            "emotion": 0.55,
        }
        return mapping.get(memory_type, 0.5)

    def _compute_recurrence(self, key: str) -> float:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM memory_candidates
            WHERE session_id = ? AND key = ?
            """,
            (self.config.session_id, key),
        ).fetchone()
        cnt = int(row["cnt"]) if row else 0
        return min(1.0, cnt / 3.0)

    def _compute_novelty(self, key: str, value: str) -> float:
        row = self.conn.execute(
            """
            SELECT value
            FROM memory_slots
            WHERE session_id = ? AND key = ?
            """,
            (self.config.session_id, key),
        ).fetchone()
        if row is None:
            return 1.0
        old_value = str(row["value"])
        if old_value == value:
            return 0.0
        return 0.4

    def _score_candidate(
        self,
        memory_type: str,
        confidence: float,
        future_impact: float,
        emotion_intensity: float,
        recurrence: float,
        novelty: float,
    ) -> float:
        importance = self._type_importance(memory_type)
        score = (
            0.30 * importance
            + 0.20 * confidence
            + 0.20 * future_impact
            + 0.15 * emotion_intensity
            + 0.10 * recurrence
            + 0.05 * novelty
        )
        return min(1.0, max(0.0, score))

    def _store_candidate_and_promote(
        self,
        turn_idx: int,
        user_text: str,
        assistant_text: str,
        c: dict[str, Any],
    ) -> None:
        recurrence = self._compute_recurrence(c["key"])
        novelty = self._compute_novelty(c["key"], c["value"])
        score = self._score_candidate(
            memory_type=c["type"],
            confidence=c["confidence"],
            future_impact=c["future_impact"],
            emotion_intensity=c["emotion_intensity"],
            recurrence=recurrence,
            novelty=novelty,
        )
        status = "promoted" if score >= self.config.promote_threshold else "candidate"
        evidence = f"user: {user_text.strip()}\nassistant: {assistant_text.strip()}"[:1200]

        self.conn.execute(
            """
            INSERT INTO memory_candidates(
                session_id, turn_idx, type, subject, key, value, evidence,
                confidence, future_impact, emotion_intensity, recurrence, novelty,
                score, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.config.session_id,
                turn_idx,
                c["type"],
                c["subject"],
                c["key"],
                c["value"],
                evidence,
                c["confidence"],
                c["future_impact"],
                c["emotion_intensity"],
                recurrence,
                novelty,
                score,
                status,
                _utc_now(),
            ),
        )

        if status == "promoted":
            self.conn.execute(
                """
                INSERT INTO memory_slots(session_id, type, subject, key, value, score, last_evidence_turn, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, key)
                DO UPDATE SET
                    type = excluded.type,
                    subject = excluded.subject,
                    value = excluded.value,
                    score = excluded.score,
                    last_evidence_turn = excluded.last_evidence_turn,
                    updated_at = excluded.updated_at
                """,
                (
                    self.config.session_id,
                    c["type"],
                    c["subject"],
                    c["key"],
                    c["value"],
                    score,
                    turn_idx,
                    _utc_now(),
                ),
            )
            row = self.conn.execute(
                """
                SELECT id
                FROM memory_slots
                WHERE session_id = ? AND key = ?
                """,
                (self.config.session_id, c["key"]),
            ).fetchone()
            if row is not None:
                slot_text = f"{c['key']} {c['value']} {c['type']} {c['subject']}"
                self._upsert_slot_vector(int(row["id"]), slot_text)
        self.conn.commit()

    def _recent_turn_digest(self) -> str:
        n = max(1, self.config.recent_turn_window)
        rows = self.conn.execute(
            """
            SELECT turn_idx, role, text
            FROM chat_turns
            WHERE session_id = ?
            ORDER BY turn_idx DESC, id DESC
            LIMIT ?
            """,
            (self.config.session_id, n * 2),
        ).fetchall()
        if not rows:
            return ""
        lines = []
        for r in reversed(rows):
            role = "user" if r["role"] == "user" else "assistant"
            lines.append(f"- ({r['turn_idx']}) {role}: {str(r['text'])[:220]}")
        return "\n".join(lines)

    def _retrieve_slots(self, current_user_text: str | None) -> list[sqlite3.Row]:
        # 1) Vector retrieval path
        if self.vector_enabled and current_user_text and current_user_text.strip():
            q_blob = self._embed_text(current_user_text)
            if q_blob is not None:
                rows = self.conn.execute(
                    """
                    SELECT
                        s.id,
                        s.key,
                        s.value,
                        s.type,
                        s.subject,
                        s.score,
                        s.last_evidence_turn,
                        s.updated_at,
                        mv.distance AS vec_distance
                    FROM memory_slots s
                    JOIN (
                        SELECT rowid, distance
                        FROM memory_slot_vec
                        WHERE embedding MATCH ?
                        AND k = ?
                    ) mv ON mv.rowid = s.id
                    WHERE s.session_id = ?
                    ORDER BY mv.distance ASC, s.updated_at DESC
                    """,
                    (q_blob, max(1, self.config.retrieval_limit), self.config.session_id),
                ).fetchall()
                if rows:
                    ranked: list[tuple[float, sqlite3.Row]] = []
                    for r in rows:
                        # sqlite-vec distance: 작을수록 유사
                        dist = _safe_float(r["vec_distance"], 1.0)
                        sim = max(0.0, 1.0 - dist)
                        final_score = float(r["score"]) * 0.35 + sim * 0.65
                        ranked.append((final_score, r))
                    if ranked:
                        ranked.sort(key=lambda x: x[0], reverse=True)
                        return [r for _, r in ranked[: max(1, self.config.retrieval_limit)]]

        # 2) Lexical fallback path
        rows = self.conn.execute(
            """
            SELECT key, value, type, subject, score, last_evidence_turn, updated_at
            FROM memory_slots
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT 100
            """,
            (self.config.session_id,),
        ).fetchall()
        if not rows:
            return []
        query_tokens = _tokenize_ko_en(current_user_text or "")
        ranked: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            text = f"{r['key']} {r['value']} {r['type']} {r['subject']}"
            slot_tokens = _tokenize_ko_en(text)
            overlap = len(query_tokens & slot_tokens)
            lexical = 0.0 if not query_tokens else min(1.0, overlap / max(1, len(query_tokens)))
            score = float(r["score"]) * 0.7 + lexical * 0.3
            ranked.append((score, r))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in ranked[: max(1, self.config.retrieval_limit)]]

    def build_memory_system_message(self, current_user_text: str | None = None) -> dict | None:
        """검색된 장기기억 + 최근 턴 요약을 system 메시지로 반환."""
        if not self.config.enabled:
            return None
        slots = self._retrieve_slots(current_user_text)
        recent_digest = self._recent_turn_digest()
        if not slots and not recent_digest:
            return None

        lines: list[str] = []
        if slots:
            lines.append("장기기억 슬롯(우선순위 순):")
            for s in slots:
                lines.append(
                    f"- [{s['type']}/{s['subject']}] {s['key']} = {s['value']} "
                    f"(score={float(s['score']):.2f}, turn={s['last_evidence_turn']})"
                )
        if recent_digest:
            lines.append("")
            lines.append("최근 대화 스냅샷:")
            lines.append(recent_digest)

        content = (
            "대화 메모리 컨텍스트:\n"
            + "\n".join(lines)
            + "\n\n"
            + "규칙: 위 메모리는 참고용이다. user의 마지막 발화에 직접 반응하라. "
            + "확실하지 않은 기억은 단정하지 마라."
        )
        if len(content) > self.config.max_summary_chars:
            content = content[: self.config.max_summary_chars].rstrip()
        return {"role": "system", "content": content}

    def update(self, user_text: str, assistant_text: str) -> None:
        """새 턴 저장 + 후보 추출 + 점수 승격."""
        if not self.config.enabled:
            return
        self.turn_count += 1
        turn_idx = self._next_turn_idx()
        self._insert_turn_pair(turn_idx, user_text, assistant_text)

        if self.turn_count % max(1, self.config.update_every_turns) != 0:
            return

        candidates = self._extract_candidates(user_text=user_text, assistant_text=assistant_text)
        if not candidates:
            return
        for c in candidates:
            self._store_candidate_and_promote(
                turn_idx=turn_idx, user_text=user_text, assistant_text=assistant_text, c=c
            )
