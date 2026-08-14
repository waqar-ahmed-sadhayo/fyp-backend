# Security Review

Aakhri review: 2026-08-13, Phase 4 ke testing/hardening pass ke saath. Ye ek student FYP ke liye point-in-time review hai, koi professional security audit nahi — kisi bhi asli deployment se pehle dependency scan dobara chalayein aur ye document dobara parh lein.

## Dependency vulnerability scan

[`pip-audit`](https://pypi.org/project/pip-audit/) ke saath chalayein:

```bash
pip install pip-audit
pip-audit
```

**Is review tak ka result: koi bhi known vulnerability nahi mili** kisi bhi installed package mein (runtime ho ya dev). `pip` khud version 26.0.1 par 3 CVEs (PYSEC-2026-196, PYSEC-2026-2875, PYSEC-2026-2876) ke liye flag hua tha — 26.1.2+ par upgrade karne se theek ho gaya; yahan wala venv ab 26.2.1 par hai. Ye scan waqtan fawqtan dobara chalate rahein; aaj ka saaf result hamesha ke liye saaf nahi rehta, kyunke pinned versions ke khilaf nayi CVEs disclose hoti rehti hain.

## Kya protect hai, aur kaise

| Concern | Mitigation | Kis test se verify hota hai |
|---|---|---|
| Password storage | bcrypt hashing, kabhi plaintext nahi | `test_auth.py` (login sirf sahi password se hi kamyab hota hai) |
| SQL injection | Har jagah sirf SQLAlchemy ORM istemal hota hai — codebase mein kahin bhi raw ya string-built query nahi hai | Code review (raw `execute(`/string-formatted SQL dhoondi gayi — koi nahi mili) |
| Session forgery | JWT `JWT_SECRET_KEY` se sign hota hai, access tokens short-lived hain (1 ghanta) | `test_auth.py` |
| Churaye gaye refresh-token ka dobara istemal | Refresh tokens individually revoke ho sakte hain (`jti` ke zariye blocklist); logout par revoke ho jate hain | `test_auth.py::test_logout_revokes_refresh_token` |
| Login/password-reset par brute-force hamla | Register/login/forgot-password/reset-password par rate limiting | `test_rate_limit.py` |
| Forgot-password ke zariye email enumeration | Chahe account maujood ho ya na ho, response bilkul waisa hi rehta hai | `test_auth.py::test_forgot_password_unknown_email_is_generic` |
| Nuqsaandeh file upload (CSV) | Extension check, null-byte/binary detection, strict UTF-8 decode, column-relevance check | `test_predict.py` |
| Nuqsaandeh file upload (PDF) | Extension check, PDF magic-byte check, size har jagah 5MB tak mehdood | `test_pdf.py` |
| Cross-origin ka ghalat istemal | CORS sirf `/api/*` surface tak mehdood | Code review (`app/__init__.py`) |
| Admin tak privilege escalation | `is_admin` sirf server-side `ADMIN_EMAILS` env var ke zariye register/login ke waqt set hota hai — koi bhi API raasta aisa nahi jahan user khud apna `is_admin` set kar sake (`user.py` ka profile update ye accept nahi karta) | `test_admin.py` |
| Doosre user ka data access karna | Har `history`/`feedback`/`profile` query sirf JWT identity ke mutabiq mehdood hai, client ke bheje hue user id par kabhi bharosa nahi kiya jata | `test_predict.py::test_history_is_scoped_to_the_user`, `test_delete_history_not_owned_by_user` |
| Source code mein secrets | `SECRET_KEY`/`JWT_SECRET_KEY`/`DATABASE_URL`/`ADMIN_EMAILS` sab env vars se parhe jate hain, saath mein saaf tor par fake dev defaults hain (`dev-secret-change-me`); `.gitignore` mein `*.db`, `venv/`, `.env.local` exclude hain | Code review (`app/config.py`) |

## Maujooda kamiyan (is pass mein theek nahi ki gayin)

- **Access tokens individually revoke nahi ho sakte.** Logout sirf refresh token ko revoke karta hai; pehle se jaari access token apni tabiyat ke mutabiq expire hone tak valid rehta hai. Isay access tokens short-lived (1 ghanta) rakh kar kam kiya gaya hai, lekin agar koi access token chori ho jaye to wo logout ke baad bhi utni hi der tak istemal ho sakta hai. Poori tarah revoke karne ke liye har request par blocklist check chahiye hoga (cost/latency ka trade-off) — ye ek soch samajh kar liya gaya scope decision hai, koi bhool nahi.
- **Rate limiter in-memory storage istemal karta hai.** Ek akela dev process ke liye theek hai; lekin multi-worker production deployment ko shared storage (Redis) chahiye hoga, warna limit global ke bajaye per-worker ho jati hai. Ye deployment phase ke liye note kar diya gaya hai.
- **Is codebase mein koi HTTPS termination nahi hai** — ye reverse-proxy ka kaam hai, jaan boojh kar yahan se bahar rakha gaya hai aur `phases/deployment/` mein track hota hai.
- **PDF/CSV parsing mein koi antivirus/malware scanning nahi hai.** Yahan sirf *format* ki sehat check hoti hai (kya ye waqai CSV/PDF hai, kya is mein pehchani jaane wali fields hain), embedded malicious payloads scan nahi kiye jate. Local/educational deployment ke liye qabil-e-qabool hai; agar bade paimane par ajnabi logon se uploads accept karne hon to isay dobara sochna hoga.
- **CI mein koi automated dependency scanning nahi hai** — is review ke liye `pip-audit` manually chalaya gaya. Is repo mein abhi koi CI pipeline nahi hai jo har change par ise chalaye; deployment phase ke saath ye ek fitri (natural) izafa hoga.
