"""
Category definitions for the datasets we can run detection on.

The categories the app needs at request time live here. `key` identifies the
dataset in requests; data_subdir and file_prefix locate its files under
settings.data_dir (see app/services/memory.py).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    normal_categories: tuple[str, ...]
    anomaly_category: str
    data_subdir: str = ""  # folder under settings.data_dir
    file_prefix: str = ""  # NLP-ADBench file prefix, e.g. "N24News"


DATASETS: dict[str, DatasetSpec] = {
    "n24_news": DatasetSpec(
        key="n24_news",
        normal_categories=(
            "Television", "Your Money", "Automobiles", "Science", "Economy",
            "Dance", "Travel", "Technology", "Sports", "Movies", "Music",
            "Real Estate", "Books", "Education", "Art & Design", "Theater",
            "Media", "Style", "Global Business", "Well", "Health",
            "Fashion & Style", "Opinion",
        ),
        anomaly_category="Food",
        data_subdir="n24",
        file_prefix="N24News",
    ),
}
