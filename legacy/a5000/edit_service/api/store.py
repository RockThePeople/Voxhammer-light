"""자산/버전 저장소 + 2단계 커밋 (HTTP_CONTRACT §10-1).

lab 과 모델이 다르다. lab 은 run 하나가 곧 자산이었지만(asset_id == run_id),
제품은 **3090이 UUID 로 asset_id 를 발급**해 보내고 한 자산 아래 v1..vN 이 쌓인다.

    {DATA_ROOT}/assets/{asset_id}/
    ├── v{n}/slat.safetensors      coords + feats + 메타
    ├── v{n}/chunks/{key}.cbin
    ├── committed.json             {"latest": 3}
    └── staging/{idempotency_key}/ 커밋 전
"""
import json
import os
import re
import shutil
import time

DATA_ROOT = os.environ.get("TRELLIS_DATA_ROOT", "<HOME>/ai-ar-data")
ASSETS = os.path.join(DATA_ROOT, "assets")
STAGING_TTL_S = 3600            # 계약 권장 1시간

ASSET_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
KEY_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# ⚠️ chunk_key 는 정규식으로 박지 않는다. 격자가 8³→16³ 로 바뀌자(D75) 옛 패턴
# `^[0-7]_[0-7]_[0-7]$` 이 v4 청크의 77.6%(509 중 395)를 조용히 422 로 거부했다.
# 상한을 16 으로 다시 박으면 같은 함정을 한 칸 옮기는 것이므로, **계약에서 유도한다.**
from deltacontract import parse_chunk_key, chunk_key as _canonical_chunk_key
from deltacontract.coords import CHUNK_GRID_RES


def _safe_chunk(v, what="chunk_key"):
    """계약 격자(CHUNK_GRID_RES)로 검증하고 **정규형**을 돌려준다.

    정규형을 쓰는 이유: parse 는 "+8" · " 8" 도 8 로 읽으므로, 원문을 그대로
    경로에 붙이면 같은 청크가 여러 문자열로 접근된다. 파싱 결과를 다시 조립해
    파일시스템에는 한 가지 표기만 닿게 한다.
    """
    try:
        cid = parse_chunk_key(v or "")
    except Exception:
        raise ValueError(f"{what} 형식이 아니다: {v!r}")
    if len(cid) != 3 or any(i < 0 or i >= CHUNK_GRID_RES for i in cid):
        raise ValueError(
            f"{what} 가 격자 범위를 벗어난다 (0..{CHUNK_GRID_RES - 1}): {v!r}")
    return _canonical_chunk_key(cid)


def _safe(pat, v, what):
    if not pat.match(v or ""):
        raise ValueError(f"{what} 형식이 아니다: {v!r}")
    return v


def asset_dir(asset_id):
    return os.path.join(ASSETS, _safe(ASSET_RE, asset_id, "asset_id"))


def version_dir(asset_id, n):
    return os.path.join(asset_dir(asset_id), f"v{int(n)}")


def staging_dir(asset_id, key):
    return os.path.join(asset_dir(asset_id), "staging",
                        _safe(KEY_RE, key, "idempotency_key"))


def committed(asset_id):
    """커밋된 최신 버전. 없으면 0."""
    p = os.path.join(asset_dir(asset_id), "committed.json")
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return int(json.load(f).get("latest", 0))


def has_committed_version(asset_id, n):
    """base_version 검증. **커밋 안 된 버전은 받지 않는다** (§10-1)."""
    try:
        return 1 <= int(n) <= committed(asset_id)
    except (ValueError, TypeError):
        return False


# ── 잡 범위 staging (계약 §13-8) ──────────────────────────────────────
#
# 결과는 **버전이 아니라 잡에 속한다.** ephemeral 편집은 promote 를 안 하므로
# to_version 이 여러 잡에서 똑같이 나오고(committed+1), 버전으로는 구분이 안 된다.
# 실측(sample-snowman, 2026-07-31): 슬롯 5개가 전부 to_version=2 였다.

def job_index_path(asset_id, job_id):
    return os.path.join(asset_dir(asset_id), "jobs",
                        _safe(KEY_RE, job_id, "job_id") + ".json")


def record_job(asset_id, job_id, key, to_version, mask=None):
    """job_id → idempotency_key 매핑을 **디스크에 영속화**한다.

    jobs.JOBS 는 프로세스 메모리라 재시작에 날아간다. 그런데 staging 은 살아
    있으므로, 재시작 후에도 잡 범위 URI 가 동작해야 한다.
    슬롯 안(job.json)과 색인(jobs/{job_id}.json) 양쪽에 적는다 — 색인이 유실돼도
    슬롯을 훑어 복구할 수 있게.
    """
    d = staging_dir(asset_id, key)
    os.makedirs(d, exist_ok=True)
    rec = {"job_id": job_id, "idempotency_key": key,
           "to_version": int(to_version), "created_at": time.time()}
    if mask is not None:
        # 마스크 지문. 슬롯에 남겨야 사후에 "두 편집의 마스크가 같았나"를
        # 즉답할 수 있다 — 없으면 출력 latent 로 역산해야 한다.
        rec["mask"] = mask
    for p in (os.path.join(d, "job.json"), job_index_path(asset_id, job_id)):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, p)


def job_id_for_key(asset_id, key):
    """idempotency_key 로 **원본 job_id** 를 찾는다. 없으면 None.

    재생이 새 job_id 를 발급하면 슬롯은 원본 job_id 아래 있어서 포인터가 어긋난다
    (바이트는 멀쩡한데 조회만 404). 그래서 재생은 원본 id 를 그대로 돌려준다.
    커밋되면 슬롯이 v{n} 으로 승격되므로 양쪽을 본다.
    """
    cand = [os.path.join(staging_dir(asset_id, key), "job.json")]
    p = idem_path(asset_id, key)
    if os.path.exists(p):
        try:
            with open(p) as f:
                n = int(json.load(f)["to_version"])
            cand.append(os.path.join(version_dir(asset_id, n), "job.json"))
        except (OSError, ValueError, KeyError):
            pass
    for c in cand:
        if os.path.exists(c):
            try:
                with open(c) as f:
                    j = json.load(f).get("job_id")
                if j:
                    return j
            except (OSError, ValueError):
                continue
    return None


def find_job(job_id):
    """job_id 로 (asset_id, 슬롯 디렉터리) 를 찾는다. 프로세스 재시작 대비.

    JOBS 는 메모리라 재시작에 날아가지만 슬롯은 디스크에 남는다. 자산 수가 적어
    (수십) 전수 조회로 충분하다.
    """
    if not os.path.isdir(ASSETS):
        return None, None
    for a in sorted(os.listdir(ASSETS)):
        try:
            if os.path.exists(job_index_path(a, job_id)):
                return a, job_slot(a, job_id)
        except ValueError:
            continue
    return None, None


def stored_response(asset_id, slot_dir):
    p = os.path.join(slot_dir or "", "response.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def job_slot(asset_id, job_id):
    """job_id 로 staging 슬롯 디렉터리를 찾는다. 없으면 None."""
    p = job_index_path(asset_id, job_id)
    if os.path.exists(p):
        with open(p) as f:
            rec = json.load(f)
        k = rec.get("idempotency_key")
        if k:
            d = staging_dir(asset_id, k)
            if os.path.isdir(d):
                return d
        cv = rec.get("committed_version")
        if cv:
            d = version_dir(asset_id, int(cv))
            if os.path.isdir(d):
                return d
    # 색인이 없거나 깨졌으면 슬롯을 훑어 복구한다.
    sroot = os.path.join(asset_dir(asset_id), "staging")
    if not os.path.isdir(sroot):
        return None
    for k in sorted(os.listdir(sroot)):
        jp = os.path.join(sroot, k, "job.json")
        if os.path.exists(jp):
            try:
                with open(jp) as f:
                    if json.load(f).get("job_id") == job_id:
                        return os.path.join(sroot, k)
            except (OSError, ValueError):
                continue
    return None


def slot_age(slot_dir):
    """슬롯 나이(초). 기준은 job.json 또는 디렉터리 mtime."""
    for c in (os.path.join(slot_dir, "job.json"), slot_dir):
        try:
            return time.time() - os.path.getmtime(c)
        except OSError:
            continue
    return float("inf")


def staging_chunk(asset_id, job_id, chunk_key):
    """잡 범위 청크 조회. (경로, 상태) 반환.

    상태: "ok" | "expired" | "no_job" | "no_chunk" | "escape"

    404 를 두 갈래로 가른다 (계약 §15):
      no_job    없는 job_id           → NOT_FOUND       (404)
      no_chunk  슬롯은 살아있는데 그 청크가 없다 → ASSET_NOT_FOUND (404)
      escape    경로 탈출 시도         → INVALID_REQUEST (422)  ← 없는 게 아니라 잘못된 것
    뒤엣것은 "부기가 지목했는데 청크가 없다" 이고 §4.2 전칭 검증이 깨졌다는
    신호다. 뭉치면 그 신호가 "없는 job_id" 와 같은 얼굴이 되어 사라진다.

    ⚠️ **청크 바이트에 TTL 을 여기서 건다.** 기존 chunk_path() 는 mtime 을 안 봐서
    만료 자체가 일어나지 않았고, 그러면 410 경로가 죽은 코드가 된다.
    """
    chunk_key = _safe_chunk(chunk_key)
    slot = job_slot(asset_id, job_id)
    if slot is None:
        return None, "no_job"
    root = os.path.realpath(asset_dir(asset_id))
    p = os.path.realpath(os.path.join(slot, "chunks", f"{chunk_key}.cbin"))
    if not p.startswith(root + os.sep):
        # 경로 탈출은 **없는 것이 아니라 잘못된 것**이다 (계약 3.14.0).
        # validate_job_id() 가 파일시스템에 닿기 전에 막으므로, 여기까지 왔다는 건
        # 그걸 안 거쳤거나 놓쳤다는 뜻 — 계약 위반 신호다. ASSET_NOT_FOUND 로 접으면
        # "부기가 지목한 청크가 없다"(§4.2 위반)와 같은 얼굴이 된다.
        # realpath 봉쇄는 다중 방어로 유지한다.
        return None, "escape"
    if slot_age(slot) > STAGING_TTL_S:
        return None, "expired"      # 존재했고 만료됐다 → 410
    if not os.path.isfile(p):
        return None, "no_chunk"     # 슬롯은 살아있는데 청크가 없다 → ASSET_NOT_FOUND
    return p, "ok"


def idem_path(asset_id, key):
    return os.path.join(asset_dir(asset_id), "idem",
                        _safe(KEY_RE, key, "idempotency_key") + ".json")


def record_idempotency(asset_id, key, to_version):
    """커밋된 연산을 키로 색인한다. staging 과 달리 **TTL 없이 남는다.**

    계약 3.7.0: "커밋된 결과가 있으면 그것을 반환. B 를 부르지 않는다."
    커밋하면 staging 이 v{n} 으로 승격돼 사라지므로, 그 이후의 재시도를 위해
    별도 색인이 필요하다. 이게 없으면 "커밋은 됐는데 응답만 유실" 재시도가
    재계산으로 떨어진다.
    """
    p = idem_path(asset_id, key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"to_version": int(to_version)}, f)
    os.replace(tmp, p)


def committed_result(asset_id, key):
    """커밋된 연산의 저장된 응답. 없으면 None."""
    p = idem_path(asset_id, key)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        n = int(json.load(f)["to_version"])
    rp = os.path.join(version_dir(asset_id, n), "response.json")
    if not os.path.exists(rp):
        return None
    with open(rp) as f:
        return json.load(f)


def replayable(asset_id, key):
    """멱등 재생 가능한 결과. **커밋 여부와 무관하게** 찾는다.

    계약 3.7.0: 이 조회가 base_version 검증보다 **먼저**다. 순서를 바꾸면
    "편집은 성공했는데 클라이언트는 409 로 실패를 배우는" 상태가 된다.

    ⚠️ `idempotency_key` 는 계약상 Optional 이다(BEditRequest). 키가 없으면
    **재생할 것이 없다** — 그런데 여기서 그냥 넘기면 `_safe()` 가 None 을 거부해
    ValueError 가 되고, 핸들러 밖으로 나가 **평문 500** 이 된다(3090 실측).
    "선택 필드를 생략했더니 서버가 500" 은 클라이언트가 원인을 못 읽는다.
    """
    if key is None:
        return None
    r = committed_result(asset_id, key)
    if r is not None:
        return r
    r, _ = staged_result(asset_id, key)
    return r


def staged_result(asset_id, key):
    """staged 결과가 살아 있으면 (응답 dict, 디렉터리). TTL 지나면 None.

    3090이 commit 을 못 보내고 죽어도 **같은 키로 재요청하면 재계산 없이**
    이걸 그대로 돌려준다. 그게 2단계 커밋의 요점이다.
    """
    d = staging_dir(asset_id, key)
    p = os.path.join(d, "response.json")
    if not os.path.exists(p):
        return None, d
    if time.time() - os.path.getmtime(p) > STAGING_TTL_S:
        return None, d          # TTL 초과 — GC 대상. 재계산한다.
    with open(p) as f:
        return json.load(f), d


def write_staging(asset_id, key, response):
    d = staging_dir(asset_id, key)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "response.json.tmp")
    with open(tmp, "w") as f:
        json.dump(response, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(d, "response.json"))
    return d


def commit(asset_id, to_version, key):
    """staging → committed 승격. 멱등이다 — 이미 승격됐으면 그대로 성공."""
    n = int(to_version)
    vd = version_dir(asset_id, n)
    sd = staging_dir(asset_id, key)
    if not os.path.isdir(vd):
        if not os.path.isdir(sd):
            raise FileNotFoundError(f"staging 도 버전도 없다: v{n} / {key}")
        os.makedirs(os.path.dirname(vd), exist_ok=True)
        shutil.move(sd, vd)
    p = os.path.join(asset_dir(asset_id), "committed.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"latest": max(n, committed(asset_id))}, f)
    os.replace(tmp, p)
    record_idempotency(asset_id, key, n)
    # 승격 후에도 job_id 로 찾히게 색인을 갱신한다 (슬롯이 v{n} 으로 옮겨갔다).
    jp = os.path.join(vd, "job.json")
    if os.path.exists(jp):
        try:
            with open(jp) as f:
                rec = json.load(f)
            ip = job_index_path(asset_id, rec["job_id"])
            os.makedirs(os.path.dirname(ip), exist_ok=True)
            with open(ip, "w") as f:
                json.dump({**rec, "committed_version": n}, f)
        except (OSError, ValueError, KeyError):
            pass
    return committed(asset_id)


def chunk_path(asset_id, chunk_key, version):
    """**커밋된** 버전의 청크 바이트 경로. 경로 검증은 realpath 봉쇄.

    ⚠️ staging 폴백을 제거했다 (계약 §15 4단계, 3090 전환 확인 후).

    이전엔 커밋 전 다운로드를 위해 staging 을 sorted() 로 훑어 첫 매치를 줬는데,
    슬롯이 둘 이상이면 **어느 잡의 바이트인지 임의**였다. 실측(sample-snowman)에서
    슬롯 5개 중 4개가 도달 불가였고 그게 Unity 의 CHUNK_HASH_MISMATCH 원인이었다.

    커밋 전 바이트는 이제 잡 범위 경로로만 받는다:
        GET /v2/assets/{asset}/staging/{job_id}/chunks/{chunk}.cbin
    """
    chunk_key = _safe_chunk(chunk_key)
    root = os.path.realpath(asset_dir(asset_id))
    p = os.path.realpath(
        os.path.join(version_dir(asset_id, version), "chunks", f"{chunk_key}.cbin"))
    if not p.startswith(root + os.sep):
        return None
    return p if os.path.isfile(p) else None


_LAST_GC = [0.0]
GC_MIN_INTERVAL_S = 300


def maybe_gc(now=None):
    """지연 GC. 요청 처리 중에 부르되 최대 GC_MIN_INTERVAL_S 마다 한 번만 실제로 돈다.

    별도 스레드·스케줄러를 두지 않는 이유는 단순함이다. 요청이 없으면 디스크가
    늘지도 않으므로 지연 GC 로 충분하다.
    """
    now = now or time.time()
    if now - _LAST_GC[0] < GC_MIN_INTERVAL_S:
        return 0
    _LAST_GC[0] = now
    return gc_staging(now)


def gc_staging(now=None):
    """TTL 지난 staging 슬롯 정리. 지워진 키로 재요청하면 재계산한다."""
    now = now or time.time()
    n = 0
    if not os.path.isdir(ASSETS):
        return 0
    for a in os.listdir(ASSETS):
        sroot = os.path.join(ASSETS, a, "staging")
        if not os.path.isdir(sroot):
            continue
        for k in os.listdir(sroot):
            d = os.path.join(sroot, k)
            try:
                age = now - os.path.getmtime(d)
            except OSError:
                continue
            if age > STAGING_TTL_S:
                # 슬롯을 지우기 전에 그 잡 색인도 지운다 — 안 그러면 jobs/ 에
                # 고아 색인이 남아 job_slot() 이 없는 디렉터리를 가리킨다.
                jp = os.path.join(d, "job.json")
                if os.path.exists(jp):
                    try:
                        with open(jp) as f:
                            jid = json.load(f).get("job_id")
                        if jid:
                            idx = os.path.join(ASSETS, a, "jobs", f"{jid}.json")
                            if os.path.exists(idx):
                                os.remove(idx)
                    except (OSError, ValueError):
                        pass
                shutil.rmtree(d, ignore_errors=True)
                n += 1
    return n
