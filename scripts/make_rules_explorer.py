"""Regenerate docs/assets/rules-explorer.html from a German Credit fit.

Run from the repo root: `uv run python scripts/make_rules_explorer.py`.
Uses the same column names as docs/notebooks/german_credit.ipynb and fits on
the full dataset so every discovered rule appears in the explorer.
"""

from flaggam import FlagGAMClassifier, export_rules_html
from flaggam.datasets import CLASSIFICATION

RENAME = {
    "Attribute1": "checking_status", "Attribute2": "duration_months",
    "Attribute3": "credit_history", "Attribute4": "purpose",
    "Attribute5": "credit_amount", "Attribute6": "savings_status",
    "Attribute7": "employment_since", "Attribute8": "installment_rate_pct",
    "Attribute9": "personal_status_sex", "Attribute10": "other_debtors",
    "Attribute11": "present_residence_since", "Attribute12": "property",
    "Attribute13": "age", "Attribute14": "other_installment_plans",
    "Attribute15": "housing", "Attribute16": "existing_credits",
    "Attribute17": "job", "Attribute18": "num_dependents",
    "Attribute19": "telephone", "Attribute20": "foreign_worker",
}

if __name__ == "__main__":
    X, y = CLASSIFICATION["german_credit"].loader()
    X = X.rename(columns=RENAME)
    clf = FlagGAMClassifier(random_state=0).fit(X, y)
    export_rules_html(
        clf,
        path="docs/assets/rules-explorer.html",
        title="FlagGAM rules — German Credit",
    )
    print(f"wrote docs/assets/rules-explorer.html ({len(clf.core_.bases_)} rules)")
