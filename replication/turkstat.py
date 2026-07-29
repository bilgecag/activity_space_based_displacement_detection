"""Readers for the official Turkish statistics used in the validation.

TURKSTAT publishes annual inter-province migration counts disaggregated by
reason; earthquake-induced moves fall under the "other" reason. Province
populations come from TURKSTAT (Turkish citizens, 2023) and the Presidency of
Migration Management (Syrians under temporary protection, April 2023).
"""
from __future__ import annotations

import pandas as pd

REASON_TRANSLATIONS = {
    "Aile Yanına/Memlekete Geri Dönme": "return_to_family",
    "Bilinmeyen": "unknown",
    "Daha İyi Konut Ve Yaşam Koşulları": "better_housing",
    "Diğer": "other",
    "Emeklilik": "retirement",
    "Ev Alınması": "house_purchase",
    "Eğitim": "education",
    "Hane/Aile Fertlerinden Birine Bağımlı Göç": "dependent_migration",
    "Medeni Durum Değişikliği/Ailevi Nedenler": "marital_family_reasons",
    "Sağlık/Bakım": "health_care",
    "Tayin/İş Değişikliği": "job_transfer",
    "İşe Başlamak/İş Bulmak": "job_seeking",
}


def turkish_to_english(text: str) -> str:
    table = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocIGUSOC")
    return text.translate(table)


def read_migration_by_reason(file_path: str, direction: str) -> pd.DataFrame:
    """City x reason migration counts ('inflow' or 'outflow')."""
    df = pd.read_csv(file_path, sep="|", skiprows=2)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    marker = "Göç Alan:" if direction == "inflow" else "Göç Veren:"

    rows = []
    for col in df.columns:
        if marker not in col:
            continue
        city, reason = col.split(marker)[1].split(" ve Göç etme nedeni:")
        rows.append({"city": city.strip(), "reason": reason.strip(),
                     "value": pd.to_numeric(df[col].iloc[2], errors="coerce")})
    long = pd.DataFrame(rows)
    wide = long.pivot(index="city", columns="reason", values="value").reset_index()
    wide[f"total_{direction}"] = wide.drop(columns="city").sum(axis=1)
    wide = wide.rename(columns=REASON_TRANSLATIONS)
    wide["city"] = wide["city"].map(lambda x: turkish_to_english(x.upper()))
    return wide


def read_turkish_population(file_path: str) -> pd.DataFrame:
    """Province-level Turkish population, 2023 ([city, turkish_population])."""
    df = pd.read_csv(file_path, sep="|", dtype=str, skiprows=4)
    df = df[["Unnamed: 1", "Unnamed: 2"]].dropna(subset=["Unnamed: 1"]).reset_index(drop=True)

    def _city(text):
        return turkish_to_english(text.split("-")[0].split("(")[0].upper())

    df["city"] = df["Unnamed: 1"].map(_city)
    df["turkish_population"] = df["Unnamed: 2"].astype(float).astype(int)
    return df.groupby("city", as_index=False)["turkish_population"].sum()


def read_syrian_population(file_path: str) -> pd.DataFrame:
    """Province-level Syrian population, April 2023 ([city, syrian_population])."""
    df = pd.read_csv(file_path, header=None).rename(
        columns={1: "city", 2: "syrian_population"})[["city", "syrian_population"]]
    df["city"] = df["city"].str.strip()
    df["syrian_population"] = df["syrian_population"].astype(int)
    return df
