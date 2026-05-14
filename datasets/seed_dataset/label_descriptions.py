from __future__ import annotations

# (label_name, description, threshold, is_niche)
DEFAULT_LABEL_DESCRIPTIONS: list[tuple[str, str, float, bool]] = [
    ("government_public",        "student doing internship at a government ministry or public sector institution",          0.50, True),
    ("hospital_niche",           "student working at a hospital clinic or medical center providing health services",        0.50, True),
    ("health_medical",           "student working in a healthcare or medical technology context",                           0.42, False),
    ("game_interactive",         "student building a game or interactive media application using Unity or Unreal",          0.45, False),
    ("finance_banking",          "student doing internship at a bank or financial technology or perbankan company",         0.45, False),
    ("apple_mobile",             "student in Apple Developer Academy building iOS applications with Swift",                 0.50, False),
    ("independent_study",        "student enrolled in a specific independent study or studi independent program",           0.45, False),
    ("internship",               "student doing a company internship or magang program",                                   0.40, False),
    ("research",                 "student doing research fellowship or certified research project",                         0.45, False),
    ("binus_bandung",            "student at BINUS University School of Computer Science Bandung campus",                  0.45, False),
    ("binus_internal_internship","student doing internship internally within BINUS University or Apple Developer Academy",  0.45, False),
    ("network_cloud",            "student working on network infrastructure cloud computing or IT operations",              0.42, False),
    ("entrepreneurship",         "student involved in startup business venture or entrepreneurship program",                0.42, False),
    ("iot_embedded",             "student building IoT embedded system microcontroller sensor or drone project",            0.45, False),
    ("data_ai",                  "student doing data science machine learning analytics or AI project",                    0.42, False),
    ("web_fullstack",            "student building web application frontend backend or full stack system",                  0.40, False),
    ("software_engineering",     "student developing software application or engineering system",                           0.38, False),
    ("education",                "student working in an education technology or academic learning platform",                0.45, False),
    ("cyber_security",           "student doing cybersecurity penetration testing or information security work",            0.50, False),
]