from __future__ import annotations

INDEPENDENT_TERMS = (
    "specific independent study",
    "independent study",
    "studi independent",
    "studi indepedent",
)
RESEARCH_TERMS = ("research", "fellowship", "certified research")
INTERNSHIP_TERMS = ("internship", "magang", "certified internship", "company internship")
BINUS_BANDUNG_TERMS = (
    "bina nusantara university school of computer science bandung",
    "binus bandung",
    "school of computer science bandung",
)
BINUS_INTERNAL_TERMS = (
    "bina nusantara university",
    "binus",
    "school of computer science bandung",
    "binus incubator",
    "apple developer academy binus",
    "apple developer academy",
)
NETWORK_TERMS = ("network", "jaringan", "cloud", "infrastructure", "it operation")
ENTRE_TERMS = ("entrepreneurship", "startup", "business", "venture")
DRONE_TERMS = ("drone", "uav")
GOVERNMENT_TERMS = ("government", "pemerintah", "kementerian", "dinas", "public sector")
HEALTH_TERMS = ("health", "kesehatan", "hospital", "medis", "klinik")
HOSPITAL_NICHE_TERMS = (
    "hospital",
    "rumah sakit",
    "siloam",
    "klinik",
    "medical center",
    "rsud",
    "rsia",
)
GAME_TERMS = ("game", "games", "gaming", "unity", "unreal")
BANKING_TERMS = ("bank", "banking", "perbankan", "financial")
APPLE_TERMS = ("apple academy", "apple developer academy", "swift", "ios")
AGIT_TERMS = ("agit", "astra graphia information technology")

WEB_TERMS = (
    "web",
    "frontend",
    "front end",
    "backend",
    "back end",
    "full stack",
    "website",
    "react",
    "javascript",
    "api",
)
SOFTWARE_ENGINEERING_TERMS = (
    "software",
    "application",
    "program",
    "developer",
    "engineering",
    "implementation",
    "system",
)
DATA_AI_TERMS = (
    "data science",
    "machine learning",
    "analytics",
    "analyst",
    "ai",
    "model",
    "visualization",
)
SECURITY_TERMS = ("security", "cyber", "infosec", "pentest")
IOT_TERMS = ("iot", "embedded", "microcontroller", "sensor", "drone", "uav", "robot")
EDUCATION_TERMS = ("academy", "instructor", "sekolah", "learning", "kampus", "universitas")

LABEL_TERMS: dict[str, tuple[str, ...]] = {
    "independent_study": INDEPENDENT_TERMS,
    "internship": INTERNSHIP_TERMS,
    "research": RESEARCH_TERMS,
    "binus_bandung": BINUS_BANDUNG_TERMS,
    "binus_internal_internship": BINUS_INTERNAL_TERMS,
    "network_cloud": NETWORK_TERMS,
    "entrepreneurship": ENTRE_TERMS,
    "iot_embedded": IOT_TERMS,
    "government_public": GOVERNMENT_TERMS,
    "health_medical": HEALTH_TERMS,
    "hospital_niche": HOSPITAL_NICHE_TERMS,
    "game_interactive": GAME_TERMS,
    "finance_banking": BANKING_TERMS,
    "apple_mobile": APPLE_TERMS,
    "web_fullstack": WEB_TERMS,
    "software_engineering": SOFTWARE_ENGINEERING_TERMS,
    "data_ai": DATA_AI_TERMS,
    "cyber_security": SECURITY_TERMS,
    "education": EDUCATION_TERMS,
}
