"""Shared code for the four steps, in the order it runs: settings, tidying up names and
postcodes, reading the two files, matching, the model, the breakdowns, and a final check that
nothing identifying leaves the machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from rapidfuzz import fuzz
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import getpass
import hashlib
import json
import numpy as np
import pandas as pd
import re
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# this file never uses the internet and never starts another program.


# ===== settings =====
"""The fixed numbers the rest of the code uses: which columns to expect, how close a name has to be to count as a match, the money bands, and a fixed number that makes each run repeatable."""


# Columns we must have. Each one is needed for the matching or the model, so if any is
# missing the run stops rather than guessing.
REQUIRED_HEADERS: tuple[str, ...] = (
    "ID",
    "Date Inserted",
    "JudgmentDate",
    "JudgmentStatus",
    "DefendantType",
    "Jurisdiction",
    "Defendant Company Name",
    "Defendant_Postcode",
)

# Columns we can manage without. None of them changes the model. If one is missing we fill it
# in and carry on (and note it): Amount only feeds the money totals, Trading Name is a backup
# way to match, and Address is not used at all.
OPTIONAL_HEADERS: dict[str, object] = {
    "Amount": None,
    "Defendant Trading Name": "",
    "Defendant Address": "",
}

HEADERS: tuple[str, ...] = (*REQUIRED_HEADERS, *OPTIONAL_HEADERS)

STATUS_VALUES: tuple[str, ...] = ("Satisfied", "Unsatisfied", "Cancelled")
DEFENDANT_TYPES: tuple[str, ...] = ("Corporate", "Non-Corporate", "Consumer", "Non-Consumer")
JURISDICTIONS: tuple[str, ...] = ("England and Wales", "Scotland")

CALIBRATION_DEFENDANT_TYPES: tuple[str, ...] = ("Corporate", "Non-Corporate")

MATCH_AUTO: float = 0.85
MATCH_REVIEW_LOW: float = 0.70

SEASONING_MONTHS: int = 12
VINTAGE_BANDS_MONTHS: tuple[tuple[int, int | None], ...] = ((0, 6), (6, 12), (12, 24), (24, None))
PRIMARY_FIT_VINTAGE_MONTHS: tuple[int, int] = (12, 36)

MIN_CELL_N: int = 10
MIN_TREE_LEAF: int = 50

AMOUNT_BANDS_GBP: tuple[int, ...] = (1_000, 5_000, 10_000, 50_000, 100_000)

DAYS_PER_MONTH: float = 30.44

SEED: int = 20260618

SNAPSHOT_DATE: str = "2026-06-01"


# ===== tidying up names and postcodes =====
"""Tidies company names and postcodes so they can be compared."""



_QUALIFIERS = (
    " T/A ",
    " TRADING AS ",
    " FORMERLY ",
    " IN LIQUIDATION",
    " IN ADMINISTRATION",
)
_LEGAL_SUFFIXES = frozenset({"LIMITED", "LTD", "PLC", "LLP", "LP", "CIC", "CO", "COMPANY"})


# tidy a company name so two versions of the same name look the same: make it upper-case,
# drop anything in brackets, drop endings like "in liquidation", and drop LTD/PLC.
def normalise_name(name: str) -> str:
    s = f" {str(name).upper()} "
    s = s.replace("&", " AND ")
    s = re.sub(r"\(.*?\)", " ", s)
    for q in _QUALIFIERS:
        idx = s.find(q)
        if idx != -1:
            s = s[:idx] + " "
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


# remove the spaces from a postcode and make it upper-case.
def normalise_postcode(postcode: str) -> str:
    return re.sub(r"\s+", "", str(postcode).upper())


# ===== reading the judgment file =====
"""Opens the judgment file and lines its columns up the way the rest of the code expects."""





_DATE_COLS = ("Date Inserted", "JudgmentDate")


# tidy a column heading so we can compare it (single spaces, lower-case).
def _canon(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


# rename the columns to our standard names, check the required ones are there, fill in missing optional ones.
def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    found = {_canon(c): c for c in df.columns}

    missing = [h for h in REQUIRED_HEADERS if _canon(h) not in found]
    if missing:
        raise ValueError(f"extract is missing required column(s): {missing}")

    canon_to_expected = {_canon(h): h for h in HEADERS}
    rename = {orig: canon_to_expected[c] for c, orig in found.items() if c in canon_to_expected}
    df = df.rename(columns=rename)

    absent_optional = [h for h in OPTIONAL_HEADERS if _canon(h) not in found]
    for h in absent_optional:
        df[h] = OPTIONAL_HEADERS[h]

    extra_headers = [c for c in df.columns if _canon(c) not in canon_to_expected]
    df.attrs["absent_optional"] = absent_optional
    df.attrs["extra_headers"] = extra_headers
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    # try the two common ways text is saved, so a file saved from Excel (with accented letters) still opens.
    problem: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError as exc:
            problem = exc
    raise ValueError(
        f"{path.name} is not readable as UTF-8 or Windows-1252. Re-save it from Excel as "
        f"'CSV UTF-8 (Comma delimited)' and try again. ({problem})"
    )


# open the judgment file (Excel or csv).
def read_extract(path: str | Path) -> pd.DataFrame:
    path = Path(str(path).strip())
    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xlsm"):
            df = pd.read_excel(path, engine="openpyxl")
        elif suffix == ".csv":
            df = _read_csv(path)
        else:
            raise ValueError(f"unsupported extract format: {suffix!r} (expected .xlsx or .csv)")
    except ValueError:
        raise
    except Exception as exc:  # if the file is broken or locked, stop with a plain message, not a scary error
        raise ValueError(f"could not read {path.name}: {type(exc).__name__}: {exc}") from exc

    df = normalise_schema(df)

    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df


# ===== reading the companies house file =====
"""Reads the big Companies House file and groups the companies by postcode, so matching only looks at companies at the same postcode.

The file is far too big to hold as a table, so it is read a piece at a time and only the columns we
need are kept. Headers are found by name because the real file's column order and spacing vary.
"""





_FIELDS = {
    "company_name": "companyname",
    "company_number": "companynumber",
    "postcode": "regaddress.postcode",
    "company_status": "companystatus",
    "company_category": "companycategory",
    "dissolution_date": "dissolutiondate",
    "incorporation_date": "incorporationdate",
    "accounts_next_due": "accounts.nextduedate",
    "accounts_category": "accounts.accountcategory",
    "n_mort_charges": "mortgages.nummortcharges",
    "n_mort_outstanding": "mortgages.nummortoutstanding",
    "n_mort_part_satisfied": "mortgages.nummortpartsatisfied",
    "n_mort_satisfied": "mortgages.nummortsatisfied",
    "sic_1": "siccode.sictext_1",
}
_PREV_NAME_CANON = [f"previousname_{i}.companyname" for i in range(1, 11)]

_INT_FIELDS = ("n_mort_charges", "n_mort_outstanding", "n_mort_part_satisfied", "n_mort_satisfied")
_STR_FIELDS = (
    "company_status",
    "company_category",
    "accounts_category",
    "sic_1",
    "dissolution_date",
)
_DATE_FIELDS = ("accounts_next_due",)

_WANTED = frozenset(_FIELDS.values()) | frozenset(_PREV_NAME_CANON)
assert set(_FIELDS.values()) <= _WANTED


# keep only the columns we actually use, matched by name.
def _wanted_col(col: object) -> bool:
    return _canon(col) in _WANTED


# tidy each different name once and reuse it, which is faster than tidying every row.
def _normalise_cached(series: pd.Series) -> np.ndarray:
    s = series.astype(str)
    lut = {u: normalise_name(u) for u in pd.unique(s.to_numpy())}
    return s.map(lut).to_numpy()


# one company: number, name, postcode, start date, and the facts the model uses.
@dataclass(slots=True)
class CompanyRecord:
    company_number: str
    company_name: str
    postcode: str
    incorporation_date: pd.Timestamp
    name_keys: tuple[str, ...]
    fields: dict[str, object] = field(default_factory=dict)


# the companies house file held two ways: by postcode (for matching) and by company number (for looking up facts).
@dataclass
class CHIndex:
    by_postcode: dict[str, list[CompanyRecord]]
    by_number: dict[str, CompanyRecord] = field(default_factory=dict)

    @property
    def n_companies(self) -> int:
        return sum(len(v) for v in self.by_postcode.values())

    def candidates(self, normalised_postcode: str) -> list[CompanyRecord]:
        return self.by_postcode.get(normalised_postcode, [])

    def get(self, company_number: str) -> CompanyRecord | None:
        return self.by_number.get(str(company_number).strip())


# read the big companies house file a piece at a time, so the whole file is never held at once.
# if it is a zip, read the csv straight from inside it; nothing is unzipped onto the disk.
def _open_rows(path: Path, chunksize: int):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)  # the csv inside the zip
            if inner is None:
                raise ValueError(f"{path.name} contains no .csv file inside it")
            with zf.open(inner) as fh:
                yield from pd.read_csv(
                    fh, dtype=str, chunksize=chunksize, keep_default_na=False,
                    usecols=_wanted_col, encoding="utf-8", encoding_errors="replace",
                )
    else:
        yield from pd.read_csv(
            path, dtype=str, chunksize=chunksize, keep_default_na=False,
            usecols=_wanted_col, encoding="utf-8", encoding_errors="replace",
        )


# read the companies house file, grouping it the two ways above.
def load_ch_index(path: str | Path, chunksize: int = 100_000) -> CHIndex:
    path = Path(str(path).strip())
    by_postcode: dict[str, list[CompanyRecord]] = {}
    by_number: dict[str, CompanyRecord] = {}

    for chunk in _open_rows(path, chunksize):
        canon_map = {_canon(c): c for c in chunk.columns}
        missing = [v for v in _FIELDS.values() if v not in canon_map]
        required = {"companyname", "companynumber", "regaddress.postcode"}
        if required & set(missing):
            missing_req = sorted(required & set(missing))
            raise ValueError(f"CH bulk missing required header(s): {missing_req}")
        prev_cols = [canon_map[c] for c in _PREV_NAME_CANON if c in canon_map]

        name_col = canon_map["companyname"]
        num_col = canon_map["companynumber"]
        pc_col = canon_map["regaddress.postcode"]
        incorp_col = canon_map.get("incorporationdate")

        raw_names = chunk[name_col].astype(str).to_numpy()
        norm_current = _normalise_cached(chunk[name_col])
        norm_prev = {c: _normalise_cached(chunk[c]) for c in prev_cols}
        nums = chunk[num_col].astype(str).str.strip().to_numpy()
        pcs = chunk[pc_col].astype(str).to_numpy()
        if incorp_col:
            incorps = pd.to_datetime(chunk[incorp_col], dayfirst=True, errors="coerce").to_numpy()
        else:
            incorps = np.full(len(chunk), np.datetime64("NaT"))

        int_arrays = {
            f: (chunk[canon_map[_FIELDS[f]]].to_numpy() if _FIELDS[f] in canon_map else None)
            for f in _INT_FIELDS
        }
        str_arrays = {
            f: (chunk[canon_map[_FIELDS[f]]].to_numpy() if _FIELDS[f] in canon_map else None)
            for f in _STR_FIELDS
        }
        parsed_dates: dict[str, np.ndarray] = {}
        for f in _DATE_FIELDS:
            col = canon_map.get(_FIELDS[f])
            parsed_dates[f] = (
                pd.to_datetime(chunk[col], dayfirst=True, errors="coerce").to_numpy()
                if col
                else np.full(len(chunk), np.datetime64("NaT"))
            )

        for i in range(len(chunk)):
            keys = [norm_current[i]]
            for c in prev_cols:
                k = norm_prev[c][i]
                if k:
                    keys.append(k)

            rec_fields: dict[str, object] = {}
            for f in _INT_FIELDS:
                arr = int_arrays[f]
                raw = arr[i] if arr is not None else ""
                rec_fields[f] = int(raw) if str(raw).strip().isdigit() else 0
            for f in _STR_FIELDS:
                arr = str_arrays[f]
                rec_fields[f] = str(arr[i]) if arr is not None else ""
            for f in _DATE_FIELDS:
                rec_fields[f] = pd.Timestamp(parsed_dates[f][i])

            rec = CompanyRecord(
                company_number=nums[i],
                company_name=raw_names[i],
                postcode=pcs[i],
                incorporation_date=pd.Timestamp(incorps[i]),
                name_keys=tuple(dict.fromkeys(k for k in keys if k)),
                fields=rec_fields,
            )
            by_postcode.setdefault(normalise_postcode(pcs[i]), []).append(rec)
            by_number[rec.company_number] = rec

    return CHIndex(by_postcode=by_postcode, by_number=by_number)


# ===== matching judgments to companies =====
"""Matches judgments to companies.

Only companies with the same postcode are considered, then names are scored on how closely they match.
A company incorporated after the judgment date cannot be the defendant, so it is rejected and the
next best candidate is tried.
"""





# turn a name-match score into: good enough (auto), needs a look (review), or no match.
def tier_for(score: float) -> str:
    if score >= MATCH_AUTO:
        return "auto"
    if score >= MATCH_REVIEW_LOW:
        return "review"
    return "unmatched"


# best name-match score against any of a company's names (current or former).
def _best_score(name_norm: str, candidate: CompanyRecord) -> float:
    best = 0.0
    for key in candidate.name_keys:
        best = max(best, fuzz.token_sort_ratio(name_norm, key) / 100.0)
    return best


# match each judgment to a company: look only at companies at the same postcode, then compare names.
def match_judgments(fold: pd.DataFrame, index: CHIndex) -> pd.DataFrame:
    ids = fold["ID"].to_numpy()
    dtypes = fold["DefendantType"].to_numpy()
    postcodes = fold["Defendant_Postcode"].to_numpy()
    names = fold["Defendant Company Name"].to_numpy()
    tradings = fold["Defendant Trading Name"].to_numpy()
    judgment_dates = list(pd.to_datetime(fold["JudgmentDate"], dayfirst=True, errors="coerce"))

    rows = []
    for i in range(len(fold)):
        in_pop = dtypes[i] in CALIBRATION_DEFENDANT_TYPES
        block = index.candidates(normalise_postcode(postcodes[i]))
        judgment_date = judgment_dates[i]

        scored: list[tuple[float, str, CompanyRecord]] = []
        if in_pop:
            name_norm = normalise_name(names[i])
            trading_raw = str(tradings[i])
            trading_norm = normalise_name(trading_raw) if trading_raw.strip() else ""
            for cand in block:
                s_name = _best_score(name_norm, cand)
                s_trade = _best_score(trading_norm, cand) if trading_norm else 0.0
                s, on = (s_name, "name") if s_name >= s_trade else (s_trade, "trading")
                scored.append((s, on, cand))
            scored.sort(key=lambda x: x[0], reverse=True)

        matched_num, matched_on, tier = "", "", "unmatched"
        matched_score = 0.0
        wrong_match_rejected = False
        age_unverifiable = False
        for s, on, cand in scored:
            if s < MATCH_REVIEW_LOW:
                break
            if pd.notna(cand.incorporation_date) and pd.notna(judgment_date):
                if judgment_date < cand.incorporation_date:  # this company started after the judgment, so it can't be the one; try the next
                    wrong_match_rejected = True
                    continue
                matched_num, matched_on, tier, matched_score = (
                    cand.company_number,
                    on,
                    tier_for(s),
                    s,
                )
                wrong_match_rejected = False
                break
            matched_num, matched_on, tier, matched_score = cand.company_number, on, tier_for(s), s
            age_unverifiable = True
            break

        best_raw = scored[0][0] if scored else 0.0
        rows.append(
            {
                "ID": ids[i],
                "in_population": in_pop,
                "matched_company_number": matched_num,
                "match_score": round(float(matched_score if tier != "unmatched" else best_raw), 4),
                "match_tier": tier,
                "matched_on": matched_on,
                "wrong_match_rejected": wrong_match_rejected,
                "age_unverifiable": age_unverifiable,
                "block_size": len(block),
            }
        )
    if not rows:
        # if there were no judgments, still hand back an empty table with the right columns
        # so the later steps don't error.
        return pd.DataFrame(
            {
                "ID": pd.Series(dtype=object),
                "in_population": pd.Series(dtype=bool),
                "matched_company_number": pd.Series(dtype=object),
                "match_score": pd.Series(dtype=float),
                "match_tier": pd.Series(dtype=object),
                "matched_on": pd.Series(dtype=object),
                "wrong_match_rejected": pd.Series(dtype=bool),
                "age_unverifiable": pd.Series(dtype=bool),
                "block_size": pd.Series(dtype=int),
            }
        )
    return pd.DataFrame(rows)


# the match report: how many matched well, how many need a look, and why the rest didn't match. counts only, no names.
def match_report(fold: pd.DataFrame, matched: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = matched.merge(fold[["ID", "DefendantType", "JudgmentDate"]], on="ID", how="left")
    work = _coerce_coded(work[work["in_population"]].copy())
    work["vintage_year"] = pd.to_datetime(
        work["JudgmentDate"], dayfirst=True, errors="coerce"
    ).dt.year

    tier_counts = (
        work.groupby(["match_tier", "DefendantType", "vintage_year"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    unmatched = work[work["match_tier"] == "unmatched"].copy()

    def _reason(row: pd.Series) -> str:
        if row["block_size"] == 0:
            return "no_postcode_candidate"
        if row["wrong_match_rejected"]:
            return "wrong_match_rejected"
        return "name_noise"

    unmatched["reason"] = unmatched.apply(_reason, axis=1)
    unmatched_diagnosis = unmatched.groupby("reason", dropna=False).size().reset_index(name="n")

    return {"tier_counts": tier_counts, "unmatched_diagnosis": unmatched_diagnosis}


# self-test only: check how many fake judgments matched back to the right company. not used on real data.
def recall_by_corruption_class(matched_with_gt: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    pop = matched_with_gt[matched_with_gt["in_population"]]
    for cls, grp in pop.groupby("corruption_class"):
        correct = (grp["matched_company_number"] == grp["company_number"]) & (
            grp["match_tier"].isin(["auto", "review"])
        )
        out[str(cls)] = float(correct.mean()) if len(grp) else 0.0
    return out


# ===== saving the match so we don't do it twice =====
"""Saves the match result so the later steps do not have to repeat it."""






# your login name, used to give you your own saved-match folder; uses "user" if it can't find it.
def _user_tag() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "user"


_CACHE_DIR = Path(tempfile.gettempdir()) / f"recovery_cache_{_user_tag()}"  # the saved-match folder, in the computer's temp area
# saved as plain text, not a format that could hide a program, and the numbers are written
# exactly so a re-run matches. the "2" means any older saved file is ignored.
_CACHE_V = 2


# a unique code worked out from the two input files. if either file changes, the code changes,
# so we never reuse a saved match that belongs to a different file.
def _fingerprint(fold_path: str | Path, ch_path: str | Path) -> str:
    h = hashlib.sha256()
    fp, cp = Path(fold_path), Path(ch_path)
    h.update(b"fold\0")
    h.update(fp.read_bytes())
    cs = cp.stat()
    h.update(b"\0ch\0")
    h.update(str(cp.resolve()).encode())
    h.update(f"\0{cs.st_size}\0{cs.st_mtime_ns}".encode())
    return h.hexdigest()[:32]


# turn the match results into plain text.
def _encode(df: pd.DataFrame) -> str:
    return json.dumps(
        {
            "v": _CACHE_V,
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "data": {c: df[c].tolist() for c in df.columns},
        }
    )


# read the match results back from that text, but only if it is the current version; otherwise redo the match.
def _decode(text: str) -> pd.DataFrame | None:
    obj = json.loads(text)
    if not (
        isinstance(obj, dict)
        and obj.get("v") == _CACHE_V
        and isinstance(obj.get("columns"), list)
        and isinstance(obj.get("dtypes"), dict)
        and isinstance(obj.get("data"), dict)
    ):
        return None
    df = pd.DataFrame({c: obj["data"][c] for c in obj["columns"]})
    return df.astype(obj["dtypes"])


# give back the match, reusing the saved one when the two input files have not changed.
def get_matched(
    fold: pd.DataFrame, index: CHIndex, fold_path: str | Path, ch_path: str | Path
) -> pd.DataFrame:
    """Give back the match of judgments to companies, reusing the saved copy when the two
    input files have not changed.

    Matching is the slow step, and steps 2, 3 and 4 each need it, so it is done once and saved.
    The saved copy is a plain-text file in the computer's temp area, tied to a code worked out
    from the two input files; if either file changes, the code changes and the match is redone.
    Any problem reading the saved copy just means it is redone, never used stale.
    """
    try:
        key = _fingerprint(fold_path, ch_path)
        cache_file = _CACHE_DIR / f"matched_{key}.json"
        if cache_file.exists():
            cached = _decode(cache_file.read_text(encoding="utf-8"))
            if cached is not None:
                return cached
    except Exception:
        cache_file = None

    matched = match_judgments(fold, index)

    try:
        if cache_file is not None:
            # lock this folder so only your login can open it (it holds company numbers).
            _CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            cache_file.write_text(_encode(matched), encoding="utf-8")
    except Exception:
        pass
    return matched


# ===== working out a few facts about each company =====
"""Builds the few company facts the model uses: charges, accounts overdue, age, active status."""




CH_SNAPSHOT = pd.Timestamp(SNAPSHOT_DATE)
FEATURE_COLUMNS = (
    "any_charges",
    "n_charges",
    "pct_charges_satisfied",
    "accounts_overdue",
    "company_age_years",
    "company_status_active",
)


# build the few company facts the model uses (charges, accounts overdue, age, active), matched rows only.
def build_features(matched: pd.DataFrame, fold: pd.DataFrame, index: CHIndex) -> pd.DataFrame:
    accepted = matched[matched["match_tier"].isin(["auto", "review"])]
    rows = []
    for _, m in accepted.iterrows():
        rec = index.get(m["matched_company_number"])
        if rec is None:
            continue
        f = rec.fields
        n_charges = int(f.get("n_mort_charges") or 0)  # type: ignore[call-overload]
        n_sat = int(f.get("n_mort_satisfied") or 0)  # type: ignore[call-overload]
        due = f.get("accounts_next_due", pd.NaT)
        incorp = rec.incorporation_date
        rows.append(
            {
                "ID": m["ID"],
                "any_charges": int(n_charges > 0),
                "n_charges": n_charges,
                "pct_charges_satisfied": min(n_sat / n_charges, 1.0) if n_charges > 0 else 0.0,
                "accounts_overdue": int(pd.notna(due) and pd.Timestamp(due) < CH_SNAPSHOT),
                "company_age_years": (
                    max((CH_SNAPSHOT - incorp).days, 0) / 365.0 if pd.notna(incorp) else 0.0
                ),
                "company_status_active": int(
                    str(f.get("company_status", "")).strip().lower() == "active"
                ),
            }
        )
    return pd.DataFrame(rows, columns=["ID", *FEATURE_COLUMNS])


# ===== choosing what to learn from =====
"""Decides which judgments the model learns from, and what counts as paid in full.

Only England and Wales company judgments at least 12 months old train the model. Scotland and cancelled
judgments are counted separately and held back.
"""





# age of each judgment in months (date inserted minus judgment date).
def _age_months(fold: pd.DataFrame) -> pd.Series:
    delta = pd.to_datetime(fold["Date Inserted"]) - pd.to_datetime(fold["JudgmentDate"])
    return delta.dt.days / DAYS_PER_MONTH


# pick which judgments the model learns from, and the yes/no it learns. only England &
# Wales company judgments at least 12 months old are used; Scotland and cancelled ones are counted separately and set aside.
def build_labelled(fold: pd.DataFrame) -> dict[str, object]:
    work = fold.copy()
    work["age_months"] = _age_months(work)
    in_pop = work["DefendantType"].isin(CALIBRATION_DEFENDANT_TYPES)
    ew = work["Jurisdiction"] == "England and Wales"
    seasoned = work["age_months"] >= SEASONING_MONTHS

    base = work[in_pop & ew & seasoned]
    primary = base[base["JudgmentStatus"].isin(["Satisfied", "Unsatisfied"])].copy()
    primary["p_full"] = (primary["JudgmentStatus"] == "Satisfied").astype(int)
    primary = primary[["ID", "p_full", "JudgmentDate"]].reset_index(drop=True)

    cancelled = base[base["JudgmentStatus"] == "Cancelled"][["ID"]].reset_index(drop=True)

    counts = {
        "rows_total": int(len(work)),
        "in_population": int(in_pop.sum()),
        "ew": int((in_pop & ew).sum()),
        "seasoned_primary": int(len(primary)),
        "cancelled_stratum": int(len(cancelled)),
        "scotland_stratum": int((in_pop & (work["Jurisdiction"] == "Scotland") & seasoned).sum()),
    }
    return {"primary": primary, "cancelled": cancelled, "counts": counts}


# ===== training the model and checking how good it is =====
"""Fits the model and measures how well it does.

Uses LightGBM where it is installed and scikit-learn otherwise. It trains on older judgments and
tests on newer ones, so the score is not flattered by hindsight.
"""






# use the LightGBM model if it is available, otherwise use scikit-learn's version.
try:
    from lightgbm import LGBMClassifier

    _HAVE_LIGHTGBM = True
except (ImportError, OSError):
    _HAVE_LIGHTGBM = False


# what the training produced: the AUC score, which model ran, and the predictions on the test judgments.
@dataclass
class FitResult:
    auc_oot: float
    backend: str
    n_train: int
    n_holdout: int
    mean_pred_pfull: float
    feature_importance: dict[str, float]
    calibration: pd.DataFrame
    evaluable: bool = field(default=True)
    below_floor: bool = field(default=False)
    holdout: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["ID", "JudgmentDate", "y_true", "proba"])
    )


# split by date: learn from the older judgments, test on the newer ones (not a random split).
def _oot_split(primary: pd.DataFrame, holdout_frac: float = 0.3) -> tuple[pd.Index, pd.Index]:
    order = pd.to_datetime(primary["JudgmentDate"], dayfirst=True).sort_values().index
    cut = int(len(order) * (1 - holdout_frac))
    return order[:cut], order[cut:]


# train the "chance it was paid in full" model, then score it on the newer judgments it never saw,
# so the score is not flattered by testing on what it already learned.
def fit_pfull(feats: pd.DataFrame, primary: pd.DataFrame) -> FitResult:
    data = primary.merge(feats, on="ID", how="inner")
    train_idx, hold_idx = _oot_split(data)
    feat_cols = list(FEATURE_COLUMNS)
    y_train = data.loc[train_idx, "p_full"]
    y_hold = data.loc[hold_idx, "p_full"]
    backend = "lightgbm" if _HAVE_LIGHTGBM else "sklearn-gbm"

    degenerate = (
        len(train_idx) == 0
        or len(hold_idx) == 0
        or int(y_train.nunique()) < 2
        or int(y_hold.nunique()) < 2
    )
    if degenerate:
        return FitResult(
            auc_oot=float("nan"),
            backend=backend,
            n_train=len(train_idx),
            n_holdout=len(hold_idx),
            mean_pred_pfull=float("nan"),
            feature_importance={},
            calibration=pd.DataFrame(columns=["bin_mean_pred", "bin_frac_pos"]),
            evaluable=False,
            below_floor=False,
            holdout=pd.DataFrame(columns=["ID", "JudgmentDate", "y_true", "proba"]),
        )

    x_train = data.loc[train_idx, feat_cols]
    x_hold = data.loc[hold_idx, feat_cols]

    if _HAVE_LIGHTGBM:
        model = LGBMClassifier(
            min_child_samples=MIN_TREE_LEAF, n_estimators=200, random_state=SEED, verbose=-1
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier

        # a fixed starting number so a re-run gives the same result.
        model = GradientBoostingClassifier(min_samples_leaf=MIN_TREE_LEAF, random_state=SEED)

    model.fit(x_train, y_train)
    pos_col = list(model.classes_).index(1)
    proba = model.predict_proba(x_hold)[:, pos_col]
    auc = float(roc_auc_score(y_hold, proba))

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.zeros(len(feat_cols))
    importance = {c: float(v) for c, v in zip(feat_cols, importances, strict=False)}

    frac_pos, mean_pred = calibration_curve(y_hold, proba, n_bins=10, strategy="quantile")
    calibration = pd.DataFrame({"bin_mean_pred": mean_pred, "bin_frac_pos": frac_pos})

    holdout = (
        data.loc[hold_idx, ["ID", "JudgmentDate", "p_full"]]
        .rename(columns={"p_full": "y_true"})
        .reset_index(drop=True)
    )
    holdout["proba"] = proba

    return FitResult(
        auc_oot=auc,
        backend=backend,
        n_train=len(train_idx),
        n_holdout=len(hold_idx),
        mean_pred_pfull=float(np.mean(proba)),
        feature_importance=importance,
        calibration=calibration,
        evaluable=True,
        below_floor=auc < 0.70,
        holdout=holdout,
    )


# write the model files (the AUC score, which facts mattered most, and calibration.png, a chart of how well its predictions matched what really happened) into the outputs folder.
def write_model_files(result: FitResult, labelled: dict, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    auc_str = f"{result.auc_oot:.4f}" if result.evaluable else "not computable (degenerate holdout)"
    (outdir / "model_score.txt").write_text(
        f"AUC (out-of-time holdout): {auc_str}\n"
        f"training rows: {result.n_train}\nholdout rows: {result.n_holdout}\n"
        f"mean predicted probability of full payment: {result.mean_pred_pfull:.4f}\n"
    )
    pd.DataFrame(
        sorted(result.feature_importance.items(), key=lambda kv: -kv[1]),
        columns=["feature", "importance"],
    ).to_csv(outdir / "feature_weights.csv", index=False)
    result.calibration.to_csv(outdir / "calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    if not result.calibration.empty:
        ax.plot(result.calibration["bin_mean_pred"], result.calibration["bin_frac_pos"], marker="o")
    else:
        ax.text(0.5, 0.5, "not evaluable", ha="center", va="center")
    ax.set_xlabel("mean predicted P(full)")
    ax.set_ylabel("observed satisfied fraction")
    ax.set_title("Probability-of-payment calibration (out-of-time)")
    fig.tight_layout()
    fig.savefig(outdir / "calibration.png", dpi=90)
    plt.close(fig)

    counts = labelled["counts"]
    if not result.evaluable:
        floor_line = "AUC: not computable (holdout had a single class or was empty)"
    else:
        floor_line = f"AUC (out-of-time holdout): {result.auc_oot:.4f}"
    (outdir / "model_summary.txt").write_text(
        "Probability-of-full-payment model\n"
        f"{floor_line}\n"
        f"mean predicted probability of full payment: {result.mean_pred_pfull:.4f}\n"
        f"fit rows: {counts['seasoned_primary']}\n"
        f"cancelled group: {counts['cancelled_stratum']}\n"
        f"scotland group: {counts['scotland_stratum']}\n"
    )


# ===== breaking the results down =====
"""Breaks the results down by age, money band and jurisdiction, and writes the run log."""





# which age band a judgment falls in.
def vintage_band(age_months: float) -> str:
    for lo, hi in VINTAGE_BANDS_MONTHS:
        if age_months >= lo and (hi is None or age_months < hi):
            return f"{lo}-{hi if hi is not None else '+'}m"
    return "unknown"


# only work out an AUC score when there are enough judgments and both outcomes appear; otherwise leave it blank.
def _safe_auc(y: pd.Series, p: pd.Series) -> float:
    if len(y) < MIN_CELL_N or y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# for each group: the real rate, the predicted rate, and the AUC; left blank if the group is too small.
def _slice_rows(dimension: str, h: pd.DataFrame) -> list[dict]:  # type: ignore[type-arg]
    out = []
    for value, sub in h.groupby(dimension, dropna=False):
        n = int(len(sub))
        small = n < MIN_CELL_N
        out.append(
            {
                "dimension": dimension,
                "value": str(value),
                "n": n,
                "mean_actual": float("nan") if small else float(sub["y_true"].mean()),
                "mean_pred": float("nan") if small else float(sub["proba"].mean()),
                "auc": _safe_auc(sub["y_true"], sub["proba"]),
            }
        )
    return out


# the breakdown by age, defendant type, charges and accounts overdue, plus how close the predictions were to what really happened and a note on the limits.
def breakdown_report(
    holdout: pd.DataFrame,
    feats: pd.DataFrame,
    fold: pd.DataFrame,
    labelled: dict,  # type: ignore[type-arg]
) -> dict[str, object]:
    h = holdout.merge(feats, on="ID", how="left").merge(
        fold[["ID", "DefendantType"]], on="ID", how="left"
    )
    h = _coerce_coded(h)
    age_months = (CH_SNAPSHOT - pd.to_datetime(h["JudgmentDate"])).dt.days / DAYS_PER_MONTH
    h = h.assign(vintage_band=age_months.apply(vintage_band), defendant_type=h["DefendantType"])

    rows: list[dict] = []  # type: ignore[type-arg]
    for dim in ("vintage_band", "defendant_type", "any_charges", "accounts_overdue"):
        rows.extend(_slice_rows(dim, h))
    performance_by_slice = pd.DataFrame(
        rows, columns=["dimension", "value", "n", "mean_actual", "mean_pred", "auc"]
    )

    if h["y_true"].nunique() > 1 and len(h) >= MIN_CELL_N:
        frac_pos, mean_pred = calibration_curve(
            h["y_true"], h["proba"], n_bins=10, strategy="quantile"
        )
        reliability = pd.DataFrame({"bin_mean_pred": mean_pred, "bin_frac_pos": frac_pos})
    else:
        reliability = pd.DataFrame(columns=["bin_mean_pred", "bin_frac_pos"])

    counts = labelled["counts"]
    strata = pd.DataFrame(
        [
            {"stratum": "seasoned_primary_EW", "n": counts.get("seasoned_primary", 0)},
            {"stratum": "cancelled_mixed_class", "n": counts.get("cancelled_stratum", 0)},
            {"stratum": "scotland_separate_regime", "n": counts.get("scotland_stratum", 0)},
        ]
    )

    revision_list = (
        "Limitations:\n"
        "  - has_all_assets_floating not in the Companies House bulk file used here\n"
        "  - features measured at the snapshot date, not the judgment date\n"
        "  - label is status-at-snapshot (no payment date), seasoned >= 12 months\n"
        "  - Cancelled group and Scotland held separate\n"
    )

    return {
        "performance_by_slice": performance_by_slice,
        "reliability": reliability,
        "strata": strata,
        "revision_list": revision_list,
    }


# write the breakdown files and the limitations note into the outputs folder.
def write_breakdown_files(report: dict, outdir: str | Path) -> None:  # type: ignore[type-arg]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report["performance_by_slice"].to_csv(outdir / "breakdown_by_group.csv", index=False)
    report["reliability"].to_csv(outdir / "reliability.csv", index=False)
    report["strata"].to_csv(outdir / "group_sizes.csv", index=False)
    (outdir / "limitations.txt").write_text(str(report["revision_list"]))


# write the run log (row counts per step) into the outputs folder.
def write_run_log(stages: list[dict], outdir: str | Path) -> None:  # type: ignore[type-arg]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = ["Run log, stage row counts and outcomes:"]
    for s in stages:
        lines.append(f"  {s.get('stage', '?')}: rows={s.get('rows', '?')}  {s.get('note', '')}")
    (outdir / "run_log.txt").write_text("\n".join(lines) + "\n")


# ===== safety check before anything leaves the machine =====
"""Checks the output files before any of them leave the machine.

Two rules. Anything that looks like it could identify a person or company is a hard stop. Any
figure resting on fewer than the cut-off number of records is blanked out. A file with no recognised count
column is treated as having nothing to blank out.
"""





_IDENTIFIER_TOKENS = (
    "postcode",
    "address",
    "company name",
    "companyname",
    "trading name",
    "tradingname",
    "name",
    "true_name",
    "company_number",
    "companynumber",
)
_COUNT_COLS = ("n", "count", "freq", "frequency")

_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}\b")  # the shape of a UK postcode, to catch any stray postcode in the outputs
_COMPANY_NUMBER_RE = re.compile(r"\b[A-Z]{2}[0-9]{6}\b")  # the shape of a lettered company number, like SC123456
_DISCLOSURE_REPORT = "safety_check.txt"


# count any values that look like a postcode or a company number.
def count_value_identifiers(text: str) -> int:
    up = str(text).upper()
    return len(_POSTCODE_RE.findall(up)) + len(_COMPANY_NUMBER_RE.findall(up))


# flag any column whose heading looks like it holds identifying info (postcode, address, company name or number).
def scan_identifiers(df: pd.DataFrame) -> list[str]:
    flagged = []
    for col in df.columns:
        key = str(col).strip().lower()
        if any(tok in key for tok in _IDENTIFIER_TOKENS):
            flagged.append(col)
    return flagged


# drop any group smaller than the cut-off, so no figure rests on too few records.
def suppress_small_cells(
    df: pd.DataFrame, *, count_col: str = "n", min_n: int = MIN_CELL_N
) -> tuple[pd.DataFrame, int]:
    if count_col not in df.columns:
        return df, 0
    counts = pd.to_numeric(df[count_col], errors="coerce")
    keep = counts >= min_n
    return df[keep].reset_index(drop=True), int((~keep).sum())


# the safety check: look at every output file before anything leaves. two rules.
# 1: anything that could identify someone (an ID-like column, or a postcode in a cell) stops the run.
# 2: any figure based on fewer than the cut-off number of records is blanked out.
# a file with no count column has nothing to blank, so it passes.
def apply_disclosure(outdir: str | Path, *, min_n: int = MIN_CELL_N) -> dict[str, object]:
    outdir = Path(outdir)
    violations: list[str] = []
    suppressed: dict[str, int] = {}
    no_count: list[str] = []

    # if a file fails the check, move it aside so it can't be sent by mistake.
    withheld = outdir / "_withheld_do_not_send"

    def _quarantine(p: Path) -> None:
        withheld.mkdir(exist_ok=True)
        p.rename(withheld / p.name)

    for csv in sorted(outdir.glob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            continue
        ident = scan_identifiers(df)
        val_hits = count_value_identifiers(df.astype(str).to_csv(index=False))
        if ident or val_hits:
            why = []
            if ident:
                why.append(f"row-level identifier column(s) {ident}")
            if val_hits:
                why.append(f"{val_hits} cell value(s) match a postcode pattern")
            violations.append(f"{csv.name}: " + "; ".join(why))
            _quarantine(csv)
            continue
        count_col = next((c for c in _COUNT_COLS if c in df.columns), None)
        if count_col:
            cleaned, n_supp = suppress_small_cells(df, count_col=count_col, min_n=min_n)
            if n_supp:
                cleaned.to_csv(csv, index=False)
                suppressed[csv.name] = n_supp
        else:
            no_count.append(csv.name)

    for txt in sorted(outdir.glob("*.txt")):
        if txt.name == _DISCLOSURE_REPORT:
            continue
        val_hits = count_value_identifiers(txt.read_text(errors="ignore"))
        if val_hits:
            violations.append(
                f"{txt.name}: {val_hits} value(s) match a row-level identifier pattern"
            )
            _quarantine(txt)

    report: dict[str, object] = {
        "violations": violations,
        "suppressed": suppressed,
        "no_count_column": no_count,
        "min_cell_n": min_n,
    }
    lines = [
        "Disclosure-control pass, min cell n >= "
        f"{min_n} (the data owner's threshold governs if stricter):",
        f"  identifier violations: {len(violations)}",
    ]
    lines += [f"    VIOLATION {v}" for v in violations]
    lines += [f"  suppressed {n} small cell(s) in {name}" for name, n in suppressed.items()]
    lines += [f"  no recognized count column (assumed count-free, not suppressed): {no_count}"]
    lines.append(
        "  RESULT: "
        + ("ZERO violations: egress-clear" if not violations else "VIOLATIONS: DO NOT EGRESS")
    )
    (outdir / _DISCLOSURE_REPORT).write_text("\n".join(lines) + "\n")
    return report


# ===== checking the judgment file =====
"""Checks the judgment file and counts what is in it."""






# check the required column headings are there, and match them to our standard names.
def validate_headers(columns: list[str] | pd.Index) -> dict[str, str]:
    expected = {_canon(h): h for h in HEADERS}
    required = {_canon(h) for h in REQUIRED_HEADERS}
    found = {_canon(c): c for c in columns}

    missing = [expected[k] for k in required if k not in found]
    if missing:
        raise ValueError(f"extract is missing required header(s): {missing}")

    return {found[k]: expected[k] for k in expected if k in found}


# true if Date Inserted is the same on every row, which is worth flagging.
def is_date_inserted_constant(df: pd.DataFrame) -> bool:
    return int(df["Date Inserted"].nunique(dropna=False)) <= 1


_KNOWN_VALUES = {
    "JudgmentStatus": STATUS_VALUES,
    "DefendantType": DEFENDANT_TYPES,
    "Jurisdiction": JURISDICTIONS,
}


# in the fixed-value columns (status, type, jurisdiction), replace any unexpected value with "(other)", so that if an identifier
# lands in one of those columns by mistake it can never show up in a count file.
def _coerce_coded(df: pd.DataFrame) -> pd.DataFrame:
    for col, known in _KNOWN_VALUES.items():
        if col in df.columns:
            df[col] = df[col].where(df[col].isin(known), "(other)")
    return df


# for the fixed-value columns, which known values appear and which unexpected ones do.
def value_sets(df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for col, known in _KNOWN_VALUES.items():
        if col not in df.columns:
            continue
        present = [str(v) for v in pd.unique(df[col].dropna())]
        out[col] = {
            "seen": [v for v in known if v in present],
            "unseen": [v for v in present if v not in known],
        }
    return out


# judgment age in months, for the audit summary.
def judgment_age_months(df: pd.DataFrame) -> pd.Series:
    delta = pd.to_datetime(df["Date Inserted"]) - pd.to_datetime(df["JudgmentDate"])
    return delta.dt.days / DAYS_PER_MONTH


# the year of each judgment.
def _vintage_year(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["JudgmentDate"]).dt.year


# counts by status, type, jurisdiction and year. counts only, no identifiers.
def row_counts(df: pd.DataFrame) -> pd.DataFrame:
    work = _coerce_coded(df.copy())
    work["vintage_year"] = _vintage_year(work)
    grouped = (
        work.groupby(
            ["JudgmentStatus", "DefendantType", "Jurisdiction", "vintage_year"],
            dropna=False,
        )
        .size()
        .reset_index(name="n")
    )
    return grouped


# count judgments per £ band.
def amount_band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    edges = [float("-inf"), *AMOUNT_BANDS_GBP, float("inf")]
    labels = (
        [f"<£{AMOUNT_BANDS_GBP[0]:,}"]
        + [
            f"£{AMOUNT_BANDS_GBP[i]:,}-{AMOUNT_BANDS_GBP[i + 1]:,}"
            for i in range(len(AMOUNT_BANDS_GBP) - 1)
        ]
        + [f"£{AMOUNT_BANDS_GBP[-1]:,}+"]
    )
    amounts = pd.to_numeric(df["Amount"], errors="coerce")
    bands = pd.cut(amounts, bins=edges, labels=labels, right=False)
    counts = bands.value_counts(sort=False).reindex(labels, fill_value=0)
    return counts.rename_axis("band").reset_index(name="n")


# what the audit found about the input file.
@dataclass
class AuditResult:
    n_rows: int
    date_inserted_constant: bool
    header_mapping: dict[str, str]
    value_sets: dict[str, dict[str, list[str]]]
    extra_headers: list[str]
    absent_optional: list[str]


# the input check: open the file, check the columns, and write the count files into the outputs folder.
def run_audit(input_path: str | Path, outdir: str | Path) -> AuditResult:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_extract(input_path)
    mapping = validate_headers(df.columns)
    extra_headers = df.attrs.get("extra_headers", [])
    absent_optional = df.attrs.get("absent_optional", [])

    constant = is_date_inserted_constant(df)
    vs = value_sets(df)

    row_counts(df).to_csv(outdir / "row_counts.csv", index=False)
    amount_band_distribution(df).to_csv(outdir / "amount_bands.csv", index=False)

    # expected values are safe to list; for unexpected ones write only how many, never the value
    # itself (a mis-aligned file could drop an identifier here).
    vs_rows = [
        {"column": col, "kind": "seen", "value": v}
        for col, d in vs.items()
        for v in d["seen"]
    ]
    vs_rows += [
        {"column": col, "kind": "unexpected_count", "value": len(d["unseen"])}
        for col, d in vs.items()
        if d["unseen"]
    ]
    pd.DataFrame(vs_rows).to_csv(outdir / "column_values.csv", index=False)

    ages = judgment_age_months(df)
    summary = [
        f"rows: {len(df)}",
        f"Date Inserted constant: {constant}",
        f"Date Inserted distinct values: {df['Date Inserted'].nunique(dropna=False)}",
        (
            f"judgment age months: min {ages.min():.1f}"
            f" / median {ages.median():.1f}"
            f" / max {ages.max():.1f}"
        ),
        f"non-null judgment ages: {int(ages.notna().sum())} / {len(df)}",
        f"seasoned >= {SEASONING_MONTHS}m: {(ages >= SEASONING_MONTHS).sum()}",
    ]
    for col, d in vs.items():
        if d["unseen"]:
            # just the count; the unexpected values themselves are kept out of the output
            summary.append(f"unexpected {col} values: {len(d['unseen'])} (values withheld)")
    if absent_optional:
        summary.append(f"OPTIONAL columns not provided (filled with defaults): {absent_optional}")
    if extra_headers:
        summary.append(f"EXTRA columns ignored: {extra_headers}")
    (outdir / "input_summary.txt").write_text("\n".join(summary) + "\n")

    return AuditResult(
        n_rows=len(df),
        date_inserted_constant=constant,
        header_mapping=mapping,
        value_sets=vs,
        extra_headers=extra_headers,
        absent_optional=absent_optional,
    )


# ===== making fake judgments for the self-test =====
"""Makes fake judgments for the self-test.

It starts from real-looking company names and messes them up in known ways, so the matcher
can be checked against a known right answer. Used only by the self-test, never on real data.
"""




CORRUPTION_CLASSES: tuple[str, ...] = (
    "clean",
    "suffix_variant",
    "punctuation",
    "typo",
    "trading_name_sub",
    "postcode_drift",
)

_SUFFIX_VARIANTS: tuple[str, ...] = (" LTD", " LTD.", " LIMITED", " CO LTD", " (UK) LTD")

_CLASS_WEIGHTS: tuple[float, ...] = (0.45, 0.15, 0.10, 0.10, 0.10, 0.10)  # one weight per way of messing up a name, same order as the list above  # noqa: E501
_STATUS_WEIGHTS: tuple[float, ...] = (0.30, 0.62, 0.08)  # one weight per status, same order
_DTYPE_WEIGHTS: tuple[float, ...] = (0.55, 0.20, 0.15, 0.10)  # one weight per defendant type, same order


# drop an ending like LTD from a name, for one of the fake test cases.
def _strip_known_suffix(name: str) -> str:
    for suffix in (" LIMITED", " LTD.", " LTD", " PLC", " CO LTD"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# deliberately mess up a name one way for a test case (typo, punctuation, different ending). test data only.
def corrupt_name(name: str, kind: str, rng: np.random.Generator) -> str:
    if kind in ("clean", "postcode_drift"):
        return name
    if kind == "suffix_variant":
        base = _strip_known_suffix(name)
        # leave out the current ending so the changed name is always different from the original.
        current_suffix = name[len(base):]
        candidates = [s for s in _SUFFIX_VARIANTS if s != current_suffix]
        if not candidates:
            candidates = list(_SUFFIX_VARIANTS)
        return base + candidates[int(rng.integers(0, len(candidates)))]
    if kind == "punctuation":
        # add a comma and full stop; the matcher's tidy-up should remove them.
        return name.replace(" ", ", ", 1) + "."
    if kind == "typo":
        chars = list(name)
        # pick a letter and change it, to make a typo.
        idxs = [i for i, c in enumerate(chars) if c.isalpha()]
        if idxs:
            pos = int(rng.choice(idxs))
            chars[pos] = "X" if chars[pos] != "X" else "Z"
        return "".join(chars)
    raise ValueError(f"unknown corruption kind: {kind!r}")


# change one character of a postcode, for the wrong-postcode test case. test data only.
def corrupt_postcode(
    name: str, postcode: str, kind: str, rng: np.random.Generator
) -> tuple[str, str]:
    """Change one character of the postcode for the 'postcode drift' test case; otherwise give the name and postcode back unchanged."""
    if kind != "postcode_drift":
        return name, postcode
    chars = list(postcode)
    idxs = [i for i, c in enumerate(chars) if c.isalnum()]
    pos = int(rng.choice(idxs))
    if chars[pos].isdigit():
        chars[pos] = "9" if chars[pos] != "9" else "0"
    else:
        chars[pos] = "Z" if chars[pos] != "Z" else "Y"
    return name, "".join(chars)


# random £ amounts for the fake judgments. test data only.
def _heavy_tailed_amounts(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.round(rng.lognormal(mean=8.5, sigma=1.3, size=n), 2)


# a made-up postcode. not real, test data only.
def _synthetic_postcode(rng: np.random.Generator) -> str:
    letters = "ABCDEFGHIJKLMNOPRSTUWYZ"
    a = letters[int(rng.integers(0, len(letters)))]
    b = letters[int(rng.integers(0, len(letters)))]
    return f"{a}{b}{int(rng.integers(1, 10))} {int(rng.integers(1, 10))}{a}{b}"


# make a fake judgment file plus the right answers to check against, for the self-test. never used on real data.
def generate_fold(
    *,
    n_rows: int,
    ch_names: list[str] | tuple[str, ...],
    seed: int = SEED,
    snapshot: str = SNAPSHOT_DATE,
    inject_unseen_status: bool = True,
    date_inserted_constant: bool = True,
    ch_bulk: pd.DataFrame | None = None,
    age_negative_fraction: float = 0.0,
    plant_signal: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make the fake judgment file plus the right answers to check it against.

    The judgment file has the same columns as a real one. The answer key has, for each judgment,
    the real company name and postcode and how the name was messed up, so the matcher can be
    scored. If a fake companies house table is passed in, each judgment is tied to a real company
    from it (same postcode, and a judgment date after the company started).
    """
    assert len(_CLASS_WEIGHTS) == len(CORRUPTION_CLASSES)
    assert len(_STATUS_WEIGHTS) == len(STATUS_VALUES)
    assert len(_DTYPE_WEIGHTS) == len(DEFENDANT_TYPES)

    rng = np.random.default_rng(seed)
    snap = pd.Timestamp(snapshot)

    if ch_bulk is not None:
        ch = ch_bulk.reset_index(drop=True)
        names = ch["CompanyName"].to_numpy()
        company_numbers_all = ch["CompanyNumber"].to_numpy()
        company_pcs_all = ch["RegAddress.PostCode"].to_numpy()
        incorp_all = pd.to_datetime(ch["IncorporationDate"], dayfirst=True)
        base_idx = rng.integers(0, len(ch), size=n_rows)
        is_age_neg = rng.random(n_rows) < age_negative_fraction
        charges_all = (
            pd.to_numeric(ch["Mortgages.NumMortCharges"], errors="coerce").fillna(0).to_numpy()
        )
        due_all = pd.to_datetime(
            ch["Accounts.NextDueDate"], dayfirst=True, errors="coerce"
        ).to_numpy()
    else:
        names = np.asarray(ch_names)
        base_idx = rng.integers(0, len(names), size=n_rows)
        is_age_neg = np.zeros(n_rows, dtype=bool)

    classes = rng.choice(
        CORRUPTION_CLASSES,
        size=n_rows,
        p=_CLASS_WEIGHTS,  # mostly clean, the rest spread across noise
    )
    # mostly business rows, with some consumer rows mixed in (those are left out of the model later).
    dtypes = rng.choice(DEFENDANT_TYPES, size=n_rows, p=_DTYPE_WEIGHTS)
    juris = rng.choice(JURISDICTIONS, size=n_rows, p=[0.9, 0.1])
    if plant_signal:
        if ch_bulk is None:
            raise ValueError("plant_signal requires ch_bulk")
        charges = charges_all[base_idx]
        overdue = (pd.to_datetime(pd.Series(due_all[base_idx])) < snap).to_numpy().astype(float)
        age_years = (
            snap - pd.to_datetime(pd.Series(incorp_all.to_numpy()[base_idx]))
        ).dt.days.to_numpy() / 365.0
        logit = 1.0 - 0.6 * charges - 1.6 * overdue + 0.04 * age_years
        p_full = 1.0 / (1.0 + np.exp(-logit))
        u = rng.random(n_rows)
        statuses = np.where(u < p_full, "Satisfied", "Unsatisfied").astype(object)
        cancel = rng.random(n_rows) < 0.05
        statuses[cancel] = "Cancelled"
    else:
        statuses = rng.choice(STATUS_VALUES, size=n_rows, p=_STATUS_WEIGHTS).astype(object)
    if inject_unseen_status:
        n_unseen = max(1, n_rows // 500)
        statuses[rng.integers(0, n_rows, size=n_unseen)] = "Set Aside"  # deliberately unseen

    if ch_bulk is not None:
        incorp = pd.to_datetime(pd.Series(incorp_all.to_numpy()[base_idx]))
        span_days = np.maximum((snap - incorp).dt.days.to_numpy(), 1)
        pos_age = (rng.random(n_rows) * span_days).astype(int)
        jd = incorp + pd.to_timedelta(pos_age, unit="D")
        neg_age = rng.integers(1, 400, size=n_rows)
        jd = jd.where(~pd.Series(is_age_neg), incorp - pd.to_timedelta(neg_age, unit="D"))
        judgment_dates = jd.reset_index(drop=True)
    else:
        age_days = rng.integers(0, 6 * 365, size=n_rows)
        judgment_dates = pd.Series(snap - pd.to_timedelta(age_days, unit="D"))

    if date_inserted_constant:
        date_inserted = pd.Series([snap] * n_rows)
    else:
        # its own randomness so only Date Inserted changes and nothing else does.
        date_rng = np.random.default_rng(seed + 1)
        gap = np.maximum((snap - judgment_dates).dt.days.to_numpy(), 1)
        extra = date_rng.integers(0, gap)
        date_inserted = judgment_dates + pd.to_timedelta(extra, unit="D")

    company_names: list[str] = []
    trading_names: list[str] = []
    postcodes: list[str] = []
    true_names: list[str] = []
    true_postcodes: list[str] = []

    for i in range(n_rows):
        real = str(names[base_idx[i]])
        kind = str(classes[i])
        if ch_bulk is not None:
            pc = str(company_pcs_all[base_idx[i]])
        else:
            pc = _synthetic_postcode(rng)
        true_names.append(real)
        true_postcodes.append(pc)
        if kind == "trading_name_sub":
            # put a trading style in the company-name column and the real name in the trading-name column.
            company_names.append(real.split()[0] + " DIRECT")
            trading_names.append(real)
            postcodes.append(pc)
        else:
            corrupted = corrupt_name(real, kind, rng)
            _, drifted_pc = corrupt_postcode(real, pc, kind, rng)
            company_names.append(corrupted)
            trading_names.append("")
            postcodes.append(drifted_pc)

    ids = np.arange(1, n_rows + 1, dtype=np.int64)
    fold = pd.DataFrame(
        {
            "Date Inserted": date_inserted.to_numpy(),
            "JudgmentStatus": statuses,
            "DefendantType": dtypes,
            "ID": ids,
            "Amount": _heavy_tailed_amounts(n_rows, rng),
            "JudgmentDate": judgment_dates.to_numpy(),
            "Defendant Company Name": company_names,
            "Defendant Trading Name": trading_names,
            "Defendant Address": [
                f"{int(rng.integers(1, 250))} HIGH STREET" for _ in range(n_rows)
            ],
            "Defendant_Postcode": postcodes,
            "Jurisdiction": juris,
        }
    )
    fold = fold[list(HEADERS)]  # put the columns in the exact expected order

    gt_data: dict[str, object] = {
        "ID": ids,
        "true_name": true_names,
        "true_postcode": true_postcodes,
        "corruption_class": classes.astype(str),
        "defendant_type": dtypes,
    }
    if ch_bulk is not None:
        gt_data["company_number"] = [str(company_numbers_all[base_idx[i]]) for i in range(n_rows)]
        gt_data["age_negative"] = is_age_neg
    ground_truth = pd.DataFrame(gt_data)
    return fold, ground_truth


# ===== making a fake companies house file for the self-test =====
"""Makes a fake Companies House file for the self-test.

One row per company, with the same style of column headings as the real file, so the reader is
tested on realistic input. Each company has a steady postcode and start date so the fake
judgments can be linked to companies. Used only by the self-test.
"""



_STATUSES = ("Active", "Active", "Active", "Liquidation")  # mostly active
_CATEGORIES = ("Private Limited Company", "PLC", "Private Limited Company")
_SIC = (
    "43999 - Other construction",
    "62012 - Business software",
    "70229 - Management consultancy",
)


# a made-up postcode for the fake CH rows. test data only.
def _postcode(rng: np.random.Generator) -> str:
    letters = "ABCDEFGHIJKLMNOPRSTUWYZ"
    a = letters[int(rng.integers(0, len(letters)))]
    b = letters[int(rng.integers(0, len(letters)))]
    return f"{a}{b}{int(rng.integers(1, 10))} {int(rng.integers(1, 10))}{a}{b}"


# make a fake Companies House bulk table for the tests. fake companies and numbers, test data only.
def generate_ch_bulk(
    ch_names: list[str] | tuple[str, ...],
    *,
    seed: int = 7,
    prev_name_fraction: float = 0.25,
) -> pd.DataFrame:
    """One row per company. Each company keeps the same postcode and start date.

    Some companies are given a former name (a small change to the current name) so the part
    of the matching that checks former names gets used.
    """
    rng = np.random.default_rng(seed)
    n = len(ch_names)
    base = pd.Timestamp("2026-06-01")

    # give each fake company a different 8-digit number (a repeat would spoil the check).
    # if the same number comes up twice, just pick another; that is lighter than building a giant list of every possible number.
    seen: set[str] = set()
    company_numbers: list[str] = []
    while len(company_numbers) < n:
        candidate = f"{int(rng.integers(1000, 99999999)):08d}"
        if candidate not in seen:
            seen.add(candidate)
            company_numbers.append(candidate)
    postcodes = [_postcode(rng) for _ in range(n)]
    incorp_offsets = rng.integers(366, 30 * 365, size=n)  # 1-30 years old
    incorp_dates = [
        (base - pd.Timedelta(days=int(d))).strftime("%d/%m/%Y") for d in incorp_offsets
    ]

    # vary the accounts due-date around the snapshot so "overdue" actually means something.
    # a due-date before the snapshot counts as overdue.
    due_offsets = rng.integers(-400, 400, size=n)
    accounts_next_due = [
        (base + pd.Timedelta(days=int(d))).strftime("%d/%m/%Y") for d in due_offsets
    ]

    prev_names = []
    prev_condates = []
    for i in range(n):
        if rng.random() < prev_name_fraction:
            stem = str(ch_names[i]).rsplit(" ", 1)[0]
            prev_names.append(stem + " HOLDINGS LIMITED")
            prev_condates.append("01/01/2020")
        else:
            prev_names.append("")
            prev_condates.append("")

    df = pd.DataFrame(
        {
            "CompanyName": list(ch_names),
            "CompanyNumber": company_numbers,
            "RegAddress.PostCode": postcodes,
            "CompanyCategory": rng.choice(_CATEGORIES, size=n),
            "CompanyStatus": rng.choice(_STATUSES, size=n),
            "DissolutionDate": ["" for _ in range(n)],
            "IncorporationDate": incorp_dates,
            "Accounts.NextDueDate": accounts_next_due,
            "Accounts.AccountCategory": rng.choice(
                ("MICRO ENTITY", "SMALL", "DORMANT"), size=n
            ),
            "Mortgages.NumMortCharges": rng.integers(0, 5, size=n),
            "Mortgages.NumMortOutstanding": rng.integers(0, 3, size=n),
            "Mortgages.NumMortPartSatisfied": np.zeros(n, dtype=np.int64),
            "Mortgages.NumMortSatisfied": rng.integers(0, 3, size=n),
            "SICCode.SicText_1": rng.choice(_SIC, size=n),
            "PreviousName_1.CONDATE": prev_condates,
            "PreviousName_1.CompanyName": prev_names,
        }
    )
    return df
