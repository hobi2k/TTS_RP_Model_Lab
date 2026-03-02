# SQLite 조회 가이드

대상 DB:
- `outputs/memory/memory.sqlite3`

## 1) 일반 테이블 (CLI로 바로 조회)

### 테이블 목록
```bash
sqlite3 outputs/memory/memory.sqlite3 ".tables"
```

### 스키마 확인
```bash
sqlite3 outputs/memory/memory.sqlite3 ".schema"
```

### chat_turns 조회
```bash
sqlite3 outputs/memory/memory.sqlite3 "SELECT id, session_id, turn_idx, role, text, ts FROM chat_turns ORDER BY id DESC LIMIT 20;"
```

### memory_candidates 조회
```bash
sqlite3 outputs/memory/memory.sqlite3 "SELECT id, session_id, turn_idx, type, subject, key, value, score, status, created_at FROM memory_candidates ORDER BY score DESC LIMIT 20;"
```

### memory_slots (장기 기억) 조회
```bash
sqlite3 outputs/memory/memory.sqlite3 "SELECT id, session_id, type, subject, key, value, score, last_evidence_turn, updated_at FROM memory_slots ORDER BY score DESC LIMIT 20;"
```

### 각 테이블 개수
```bash
sqlite3 outputs/memory/memory.sqlite3 "SELECT COUNT(*) FROM chat_turns;"
sqlite3 outputs/memory/memory.sqlite3 "SELECT COUNT(*) FROM memory_candidates;"
sqlite3 outputs/memory/memory.sqlite3 "SELECT COUNT(*) FROM memory_slots;"
```

## 2) 벡터 테이블 (sqlite-vec)

`memory_slot_vec`는 `vec0` 확장을 로드해야 조회 가능하다.

### 확장 경로 확인
```bash
uv run python - <<'PY'
import sqlite_vec
print(sqlite_vec.loadable_path())
PY
```

### CLI에서 vec 조회
```bash
sqlite3 outputs/memory/memory.sqlite3 \
  -cmd ".load /절대/경로/sqlite_vec/vec0" \
  "SELECT COUNT(*) FROM memory_slot_vec;"
```

### vec rowid + memory_slots 조인
```bash
sqlite3 outputs/memory/memory.sqlite3 \
  -cmd ".load /절대/경로/sqlite_vec/vec0" \
  "SELECT ms.id, ms.key, ms.value, ms.score FROM memory_slots ms JOIN memory_slot_vec mv ON mv.rowid = ms.id ORDER BY ms.score DESC LIMIT 20;"
```

## 3) Python으로 vec 조회

```bash
uv run python - <<'PY'
import sqlite3
import sqlite_vec

conn = sqlite3.connect("outputs/memory/memory.sqlite3")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.row_factory = sqlite3.Row

print("=== memory_slot_vec count ===")
print(conn.execute("SELECT COUNT(*) FROM memory_slot_vec").fetchone()[0])

print("=== memory_slots + vec join ===")
rows = conn.execute(
    "SELECT ms.id, ms.key, ms.value, ms.score FROM memory_slots ms JOIN memory_slot_vec mv ON mv.rowid = ms.id ORDER BY ms.score DESC LIMIT 20"
).fetchall()
for r in rows:
    print(dict(r))

conn.close()
PY
```

## 4) 참고

- `memory_slot_vec`는 `memory_slots.id`를 `rowid`로 사용한다.
- vec가 생성되지 않으면 `sqlite_vec.load()`가 실패한 경우가 대부분이다.
- 앱 로그에서 아래가 찍히면 정상:
```
[memory] db=... vector_enabled=True sqlite_vec_loaded=True dim=1024
```
