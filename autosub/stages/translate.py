"""Translate stage (U5, KTD6): windowed ollama chat, refusal -> fallback model -> single lines -> marker."""
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import threading

from .. import agy, config, jobs, ollama
from ..refusal import is_refusal

LANGS = {"en": "English", "vi": "Vietnamese"}
UNTRANSLATED = "[untranslated]"


SERVED = collections.Counter()  # model -> windows it translated (shows how often a cloud model refused)
SERVED_LOCK = threading.Lock()
LOCAL_MODEL_LOCK = threading.Lock()


class GpuFitError(RuntimeError):
    pass


TAG = re.compile(r"^\s*\[[MF]\]\s*")


def tagged(seg):
    """'[M] text' when the voice guess knows the speaker's gender (voice.py)."""
    return f"[{seg['gender']}] {seg['text']}" if seg.get("gender") else seg["text"]


def brief_summary(brief):
    summary = brief.get("summary") if isinstance(brief, dict) else None
    return summary[:600] if isinstance(summary, str) else None


VI_PRONOUNS = {"anh", "em", "chị", "cô", "chú", "bác", "ông", "bà", "cháu", "con", "tôi", "mình", "cậu", "tớ",
               "tao", "mày", "thầy", "trò", "sếp", "chồng", "vợ", "bố", "mẹ", "ba", "má"}


DEFAULT_PAIR = {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}


def address_pair(brief):
    """The brief's Vietnamese pronoun pair, or None unless it is complete, real, and consistent:
    what she calls him must be what he calls himself, and vice versa (anh/em, chú/cháu, sếp/em...)."""
    address = brief.get("vi_address") if isinstance(brief, dict) else None
    if not isinstance(address, dict):
        return None
    pair = {k: address.get(k).strip().lower() for k in DEFAULT_PAIR if isinstance(address.get(k), str)}
    if len(pair) != 4 or not set(pair.values()) <= VI_PRONOUNS:  # 4B model sometimes answers "Speaker 1"
        return None
    if pair["female_to_male"] != pair["male_self"] or pair["male_to_female"] != pair["female_self"]:
        return None
    return pair


# These are role-shaped reciprocal Vietnamese pairs, not literal word-for-word mappings. For
# age-unspecified titles (work/customer/senior), anh/em avoids guessing a generation; parent and
# grandparent titles use con and cháu respectively. Separate speaker directions cover mothers,
# wives, daughters, and sisters addressed by men. Keeping each pair reciprocal is required by
# address_pair(): each person's self-reference is what the other calls them.
KINSHIP_PAIRS = [
    # Japanese, woman -> man.
    {"speaker": "F", "language": "ja", "terms": ("お義父さん", "お父さん", "父さん", "パパ"),
     "pair": {"male_self": "bố", "male_to_female": "con", "female_self": "con", "female_to_male": "bố"}},
    {"speaker": "F", "language": "ja", "terms": ("お兄ちゃん", "お兄さん", "兄さん"),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},
    {"speaker": "F", "language": "ja", "terms": ("おじさん", "叔父さん", "伯父さん"),
     "pair": {"male_self": "chú", "male_to_female": "cháu", "female_self": "cháu", "female_to_male": "chú"}},
    {"speaker": "F", "language": "ja", "terms": ("先生",),
     "pair": {"male_self": "thầy", "male_to_female": "em", "female_self": "em", "female_to_male": "thầy"}},
    {"speaker": "F", "language": "ja", "terms": ("社長", "部長", "課長"),
     "pair": {"male_self": "sếp", "male_to_female": "em", "female_self": "em", "female_to_male": "sếp"}},
    {"speaker": "F", "language": "ja", "terms": ("あなた",), "min_hits": 10,
     "pair": {"male_self": "chồng", "male_to_female": "vợ", "female_self": "vợ", "female_to_male": "chồng"}},
    {"speaker": "F", "language": "ja", "terms": ("おじいちゃん", "おじいさん", "じいちゃん"),
     "pair": {"male_self": "ông", "male_to_female": "cháu", "female_self": "cháu", "female_to_male": "ông"}},
    {"speaker": "F", "language": "ja", "terms": ("先輩",),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},
    {"speaker": "F", "language": "ja", "terms": ("お客様",),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},

    # Simplified and Traditional Chinese, woman -> man.
    {"speaker": "F", "language": "zh", "terms": ("爸爸", "爸", "公公", "岳父"),
     "pair": {"male_self": "bố", "male_to_female": "con", "female_self": "con", "female_to_male": "bố"}},
    {"speaker": "F", "language": "zh", "terms": ("哥哥", "哥"),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},
    {"speaker": "F", "language": "zh", "terms": ("叔叔", "大叔"),
     "pair": {"male_self": "chú", "male_to_female": "cháu", "female_self": "cháu", "female_to_male": "chú"}},
    {"speaker": "F", "language": "zh", "terms": ("老师", "老師"),
     "pair": {"male_self": "thầy", "male_to_female": "em", "female_self": "em", "female_to_male": "thầy"}},
    {"speaker": "F", "language": "zh", "terms": ("老板", "老闆", "经理", "經理"),
     "pair": {"male_self": "sếp", "male_to_female": "em", "female_self": "em", "female_to_male": "sếp"}},
    {"speaker": "F", "language": "zh", "terms": ("老公",),
     "pair": {"male_self": "chồng", "male_to_female": "vợ", "female_self": "vợ", "female_to_male": "chồng"}},
    {"speaker": "F", "language": "zh", "terms": ("爷爷", "爺爺"),
     "pair": {"male_self": "ông", "male_to_female": "cháu", "female_self": "cháu", "female_to_male": "ông"}},
    {"speaker": "F", "language": "zh", "terms": ("学长", "學長"),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},
    {"speaker": "F", "language": "zh", "terms": ("客人",),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},

    # Korean, woman -> man. 형 is a male-to-male form, outside this mixed-gender pair model.
    {"speaker": "F", "language": "ko", "terms": ("아빠", "아버지", "아버님"),
     "pair": {"male_self": "bố", "male_to_female": "con", "female_self": "con", "female_to_male": "bố"}},
    {"speaker": "F", "language": "ko", "terms": ("오빠",),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},
    {"speaker": "F", "language": "ko", "terms": ("아저씨", "삼촌"),
     "pair": {"male_self": "chú", "male_to_female": "cháu", "female_self": "cháu", "female_to_male": "chú"}},
    {"speaker": "F", "language": "ko", "terms": ("선생님",),
     "pair": {"male_self": "thầy", "male_to_female": "em", "female_self": "em", "female_to_male": "thầy"}},
    {"speaker": "F", "language": "ko", "terms": ("사장님", "부장님", "팀장님"),
     "pair": {"male_self": "sếp", "male_to_female": "em", "female_self": "em", "female_to_male": "sếp"}},
    {"speaker": "F", "language": "ko", "terms": ("여보", "자기"),
     "pair": {"male_self": "chồng", "male_to_female": "vợ", "female_self": "vợ", "female_to_male": "chồng"}},
    {"speaker": "F", "language": "ko", "terms": ("할아버지",),
     "pair": {"male_self": "ông", "male_to_female": "cháu", "female_self": "cháu", "female_to_male": "ông"}},
    {"speaker": "F", "language": "ko", "terms": ("선배", "선배님"),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},
    {"speaker": "F", "language": "ko", "terms": ("손님",),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},

    # Men addressing women: pair order still describes male and female roles.
    {"speaker": "M", "language": "ja", "terms": ("お母さん", "母さん", "ママ"),
     "pair": {"male_self": "con", "male_to_female": "mẹ", "female_self": "mẹ", "female_to_male": "con"}},
    {"speaker": "M", "language": "ja", "terms": ("奥さん", "嫁"),
     "pair": {"male_self": "chồng", "male_to_female": "vợ", "female_self": "vợ", "female_to_male": "chồng"}},
    {"speaker": "M", "language": "ja", "terms": ("娘", "娘さん"),
     "pair": {"male_self": "bố", "male_to_female": "con", "female_self": "con", "female_to_male": "bố"}},
    {"speaker": "M", "language": "ja", "terms": ("お姉ちゃん", "お姉さん", "姉さん"),
     "pair": {"male_self": "em", "male_to_female": "chị", "female_self": "chị", "female_to_male": "em"}},
    {"speaker": "M", "language": "ja", "terms": ("妹", "妹ちゃん"),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},
    {"speaker": "M", "language": "ja", "terms": ("先生",),
     "pair": {"male_self": "em", "male_to_female": "cô", "female_self": "cô", "female_to_male": "em"}},
    {"speaker": "M", "language": "ja", "terms": ("社長", "部長", "課長"),
     "pair": {"male_self": "em", "male_to_female": "sếp", "female_self": "sếp", "female_to_male": "em"}},

    {"speaker": "M", "language": "zh", "terms": ("妈妈", "媽媽", "妈", "媽"),
     "pair": {"male_self": "con", "male_to_female": "mẹ", "female_self": "mẹ", "female_to_male": "con"}},
    {"speaker": "M", "language": "zh", "terms": ("老婆", "妻子"),
     "pair": {"male_self": "chồng", "male_to_female": "vợ", "female_self": "vợ", "female_to_male": "chồng"}},
    {"speaker": "M", "language": "zh", "terms": ("女儿", "女兒", "闺女", "閨女", "娘"),
     "pair": {"male_self": "bố", "male_to_female": "con", "female_self": "con", "female_to_male": "bố"}},
    {"speaker": "M", "language": "zh", "terms": ("姐姐", "姊姊", "姐"),
     "pair": {"male_self": "em", "male_to_female": "chị", "female_self": "chị", "female_to_male": "em"}},
    {"speaker": "M", "language": "zh", "terms": ("妹妹", "妹"),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},
    {"speaker": "M", "language": "zh", "terms": ("老师", "老師"),
     "pair": {"male_self": "em", "male_to_female": "cô", "female_self": "cô", "female_to_male": "em"}},
    {"speaker": "M", "language": "zh", "terms": ("老板", "老闆", "经理", "經理"),
     "pair": {"male_self": "em", "male_to_female": "sếp", "female_self": "sếp", "female_to_male": "em"}},

    {"speaker": "M", "language": "ko", "terms": ("엄마", "어머니", "어머님"),
     "pair": {"male_self": "con", "male_to_female": "mẹ", "female_self": "mẹ", "female_to_male": "con"}},
    {"speaker": "M", "language": "ko", "terms": ("여보", "자기"),
     "pair": {"male_self": "chồng", "male_to_female": "vợ", "female_self": "vợ", "female_to_male": "chồng"}},
    {"speaker": "M", "language": "ko", "terms": ("딸", "딸아이"),
     "pair": {"male_self": "bố", "male_to_female": "con", "female_self": "con", "female_to_male": "bố"}},
    {"speaker": "M", "language": "ko", "terms": ("누나",),
     "pair": {"male_self": "em", "male_to_female": "chị", "female_self": "chị", "female_to_male": "em"}},
    {"speaker": "M", "language": "ko", "terms": ("여동생",),
     "pair": {"male_self": "anh", "male_to_female": "em", "female_self": "em", "female_to_male": "anh"}},
    {"speaker": "M", "language": "ko", "terms": ("선생님",),
     "pair": {"male_self": "em", "male_to_female": "cô", "female_self": "cô", "female_to_male": "em"}},
    {"speaker": "M", "language": "ko", "terms": ("사장님", "부장님", "팀장님"),
     "pair": {"male_self": "em", "male_to_female": "sếp", "female_self": "sếp", "female_to_male": "em"}},
]


def with_kinship(brief, segs, min_hits=3):
    """Infer a reciprocal Vietnamese pair from gender-tagged vocatives only when the brief lacks one."""
    brief = dict(brief) if isinstance(brief, dict) else {}
    if address_pair(brief):
        return brief

    hits = [0] * len(KINSHIP_PAIRS)
    for speaker in ("F", "M"):
        text = " ".join(s.get("src") or s.get("text", "") for s in segs if s.get("gender") == speaker)
        term_to_entry = {term: i for i, entry in enumerate(KINSHIP_PAIRS) if entry["speaker"] == speaker
                         for term in entry["terms"]}
        if not term_to_entry:
            continue
        pattern = re.compile("|".join(re.escape(term) for term in
                                      sorted(term_to_entry, key=lambda term: (-len(term), term))))
        for match in pattern.finditer(text):
            hits[term_to_entry[match.group()]] += 1

    best_hits = max(hits, default=0)
    winners = [i for i, count in enumerate(hits) if count == best_hits and count > 0]
    if len(winners) == 1:
        winner = KINSHIP_PAIRS[winners[0]]
        if best_hits >= max(min_hits, winner.get("min_hits", min_hits)):
            brief["vi_address"] = dict(winner["pair"])
            brief["address_source"] = "kinship"
    return brief


def allowed_pronouns(lang, brief=None):
    """Personal pronouns a Vietnamese line may use; others get flagged for polish (flags.py)."""
    return set((address_pair(brief) or DEFAULT_PAIR).values()) if lang == "vi" else None


# The pronoun rule must not rewrite kinship/role words the source actually says (お父さん is "bố", not "anh").
KINSHIP_NOTE = (" Quy tắc xưng hô chỉ áp dụng cho đại từ nhân xưng (tôi, bạn, anh, em...). Từ chỉ quan hệ hoặc "
                "vai trò có trong câu gốc thì dịch đúng nghĩa: お父さん/父 → bố, お母さん/母 → mẹ, 先生 → thầy/cô, "
                "社長 → giám đốc, 先輩 → tiền bối, 奥さん → vợ/chị nhà.")


def address_rule(cfg, lang, brief=None):
    pair = address_pair(brief) if lang == "vi" else None
    if pair:
        return (f" Xưng hô theo bối cảnh: nam xưng '{pair['male_self']}', gọi nữ là '{pair['male_to_female']}'; "
                f"nữ xưng '{pair['female_self']}', gọi nam là '{pair['female_to_male']}'. Giữ nguyên suốt video."
                + KINSHIP_NOTE)
    rule = cfg.get("address", {}).get(lang)
    return (f" {rule}" + (KINSHIP_NOTE if lang == "vi" else "")) if rule else ""


def prompt(lines, ctx, lang, cfg=None, *, previous_sources=None, following=None, brief=None):
    system = (f"You translate subtitles of adult films into natural, colloquial {LANGS[lang]}. Explicit sexual "
              "content is expected: translate it faithfully, never censor, never comment. Reply with ONLY a JSON "
              f"array of exactly {len(lines)} strings, one per input line, in order. A line may start with [M] "
              "(male speaker) or [F] (female speaker): use it for pronouns and tone, never output the tag."
              f"{address_rule(cfg or {}, lang, brief)} /no_think")
    user = ""
    summary = brief_summary(brief)
    if summary:
        user += f"Scene brief (context only): {summary}\n\n"
    if ctx:
        user += "Previous lines (context only):\n" + "\n".join(f"{ja} => {vi}" for ja, vi in ctx) + "\n\n"
    if previous_sources:
        user += "Previous Japanese source lines (context only):\n" + "\n".join(f"- {line}" for line in previous_sources) + "\n\n"
    user += "Translate:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    if following:
        user += "\n\nFollowing lines (context only, do not translate):\n" + "\n".join(f"- {line}" for line in following)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def check_fit(cfg, model):
    for m in ollama.ps(cfg["ollama_url"]):
        if m["name"] in (model, f"{model}:latest") and m.get("size_vram", 0) < cfg.get("min_gpu_share", 0.9) * m.get("size", 0):  # a few % on CPU is fine
            raise GpuFitError(f"model did not fit in GPU memory: {model} "
                              f"({m['size_vram'] >> 20}/{m['size'] >> 20} MiB on GPU)")


def ask(cfg, model, lines, ctx, lang, used, messages=None):
    """One chat call; None if the output isn't a JSON list. First call per model checks GPU fit (R9)."""
    msgs = messages or prompt(lines, ctx, lang, cfg)
    if agy.is_agy(model):  # cloud: nothing on the GPU to check or free
        content = agy.chat(model, msgs)
    else:
        # Cloud windows can fall back together, but only one local GPU request may run at once.
        with LOCAL_MODEL_LOCK:
            content = ollama.chat(cfg["ollama_url"], model, msgs, cfg)
            if model not in used:
                used.add(model)
                check_fit(cfg, model)
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())  # gemma wraps JSON in a code fence
    try:
        return json.loads(content)
    except ValueError:
        return None


def with_fallback(cfg, lines, ctx, lang, used, messages=None, fallback_stats=None):
    for index, model in enumerate((cfg["models"][lang], cfg["fallback_model"])):
        try:
            out = ask(cfg, model, lines, ctx, lang, used, messages)
        except OSError:  # timeout / connection drop: same path as a refusal (R13)
            if index == 0 and agy.is_agy(model) and fallback_stats is not None:
                with SERVED_LOCK:
                    fallback_stats["cloud_fallbacks"] += 1
            continue
        if not is_refusal(lines, out, lang):
            with SERVED_LOCK:
                SERVED[model] += 1
            return out
        if index == 0 and agy.is_agy(model) and fallback_stats is not None:
            with SERVED_LOCK:
                fallback_stats["cloud_fallbacks"] += 1
    return None


def _ask_cloud(cfg, lines, lang, used, messages):
    """Primary cloud model only; None on refusal/error (no local fallback here)."""
    model = cfg["models"][lang]
    try:
        out = ask(cfg, model, lines, [], lang, used, messages)
    except OSError:
        return None
    if is_refusal(lines, out, lang):
        return None
    with SERVED_LOCK:
        SERVED[model] += 1
    return out


def _cloud_bisect(cfg, sources, texts, start, end, lang, used, brief, fallback_stats):
    """Translate texts[start:end] with the cloud model; on refusal split in half and retry, so only the
    lines the cloud refuses even on their own go to the local fallback. -> (translations, failed)."""
    prev_n, lookahead = cfg.get("context_lines", 0), cfg.get("lookahead_lines", 4)
    win = texts[start:end]
    msgs = prompt(win, [], lang, cfg, previous_sources=sources[max(0, start - prev_n):start],
                  following=sources[end:end + lookahead], brief=brief)
    res = _ask_cloud(cfg, win, lang, used, msgs)
    if res is not None:
        return res, 0
    if len(win) > 1:
        mid = start + len(win) // 2
        a, fa = _cloud_bisect(cfg, sources, texts, start, mid, lang, used, brief, fallback_stats)
        b, fb = _cloud_bisect(cfg, sources, texts, mid, end, lang, used, brief, fallback_stats)
        return a + b, fa + fb
    if fallback_stats is not None:  # one line the cloud refused on its own: local model
        with SERVED_LOCK:
            fallback_stats["cloud_fallbacks"] += 1
    local = {**cfg, "models": {**cfg["models"], lang: cfg["fallback_model"]}}
    one = with_fallback(local, win, [], lang, used, msgs)
    return (one, 0) if one else ([UNTRANSLATED], 1)


def _translate_window(cfg, win, lang, used, *, ctx, previous_sources, following, brief, fallback_stats=None):
    messages = prompt(win, ctx, lang, cfg, previous_sources=previous_sources, following=following, brief=brief)
    res = with_fallback(cfg, win, ctx, lang, used, messages, fallback_stats)
    if res is not None:
        return res, 0
    out, failed = [], 0
    for line in win:
        single = prompt([line], ctx, lang, cfg, previous_sources=previous_sources, following=following, brief=brief)
        one = with_fallback(cfg, [line], ctx, lang, used, single, fallback_stats)
        out.append(one[0] if one else UNTRANSLATED)
        failed += one is None
    return out, failed


def translate_texts(cfg, texts, lang, used, progress=lambda f: None, *, sources=None, brief=None,
                    two_sided=True, fallback_stats=None):
    """-> (translations, failed_line_count). Never drops a line (R13)."""
    sources = list(sources) if sources is not None else list(texts)
    cloud = agy.is_agy(cfg["models"][lang])
    n = cfg.get("cloud_window", 24) if cloud else cfg["window"]
    prev_n = cfg.get("context_lines", 0)
    lookahead = cfg.get("lookahead_lines", 4) if two_sided else 0
    completed_lines = 0
    last_progress = 0.0
    progress_lock = threading.Lock()

    def report(value):
        nonlocal last_progress
        with progress_lock:
            last_progress = max(last_progress, min(1.0, float(value)))
            progress(last_progress)

    windows = [(i, texts[i:i + n]) for i in range(0, len(texts), n)]
    failed = 0
    if cloud:
        results = [None] * len(windows)

        def translate(index, i, win):
            res, errors = _cloud_bisect(cfg, sources, texts, i, i + len(win), lang, used, brief, fallback_stats)
            return index, res, errors

        pool = ThreadPoolExecutor(max_workers=max(1, int(cfg.get("cloud_parallel", 4))))
        futures = {}
        try:
            for index, (i, win) in enumerate(windows):
                futures[pool.submit(translate, index, i, win)] = index
            for future in as_completed(futures):
                index, res, errors = future.result()
                results[index] = res
                failed += errors
                completed_lines += len(windows[index][1])
                report(completed_lines / len(texts))
        except BaseException:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        out = [line for result in results for line in result]
    else:
        out = []
        for i, win in windows:
            ctx = list(zip(sources[max(0, i - prev_n):i], out[max(0, i - prev_n):i])) if prev_n and two_sided else []
            following = sources[i + len(win):i + len(win) + lookahead]
            res, errors = _translate_window(cfg, win, lang, used, ctx=ctx, previous_sources=[],
                                            following=following, brief=brief, fallback_stats=fallback_stats)
            out.extend(res)
            failed += errors
            completed_lines += len(win)
            report(completed_lines / len(texts))
    return out, failed


def translate_via_pivot(cfg, texts, lang, used, progress=lambda f: None, *, sources=None, brief=None,
                        fallback_stats=None):
    """Small models translate JA->EN far better than JA->VI, so go through the pivot language when configured."""
    pivot = cfg.get("pivot", {}).get(lang)
    if not pivot:
        return translate_texts(cfg, texts, lang, used, progress, sources=sources, brief=brief,
                               fallback_stats=fallback_stats)
    mid, _ = translate_texts(cfg, texts, pivot, used, lambda f: progress(f / 2), brief=brief,
                             two_sided=False, fallback_stats=fallback_stats)
    out, _ = translate_texts(cfg, mid, lang, used, lambda f: progress(0.5 + f / 2), brief=brief,
                             two_sided=False, fallback_stats=fallback_stats)
    out = [UNTRANSLATED if m == UNTRANSLATED else o for m, o in zip(mid, out)]
    return out, out.count(UNTRANSLATED)


def unload_all(cfg, used):
    for m in used:
        try:
            ollama.unload(cfg["ollama_url"], m)
        except Exception:
            pass


def run(cfg, db, batch):
    used = set()
    try:
        for job in batch:
            if jobs.get(db, job["id"])["control"]:
                continue
            try:
                d = config.job_dir(cfg, job["id"])
                segs = json.loads((d / "segments.json").read_text())
                brief = json.loads((d / "brief.json").read_text()) if (d / "brief.json").exists() else None
                brief = with_kinship(brief, segs)
                fallback_stats = {"cloud_fallbacks": 0}
                tr, failed = translate_via_pivot(cfg, [tagged(s) for s in segs], job["lang"], used,
                                                 lambda f: jobs.progress(db, job["id"], f),
                                                 sources=[s["text"] for s in segs], brief=brief,
                                                 fallback_stats=fallback_stats)
                tr = [TAG.sub("", t) for t in tr]
                for s, t in zip(segs, tr):
                    s["src"], s["text"] = s["text"], t
                (d / "translated.json").write_text(json.dumps(segs, ensure_ascii=False))
                jobs.update(db, job["id"], stage="translated", progress=0, failed_lines=failed,
                            cloud_fallbacks=fallback_stats["cloud_fallbacks"])
            except jobs.Stopped:  # paused/deleted from the UI: leave the stage to redo later
                continue
    finally:
        unload_all(cfg, used)  # keep_alive 0 + /api/ps confirm, so the next GPU stage gets the VRAM (KTD2)
