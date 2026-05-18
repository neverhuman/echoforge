from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "ai-service"))

from echoforge_core import (  # noqa: E402
    ComplexScalar,
    DatasetCard,
    LicenseInfo,
    MaterialCard,
    NumericRange,
    ObjectCard,
    Provenance,
    SplitCounts,
    ValidationCheck,
    ValidationInfo,
    Vector3,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


class CoreSliceTests(unittest.TestCase):
    def sample_common(self) -> dict:
        return {
            "provenance": Provenance(
                source_kind="synthetic",
                source_refs=["tests/schemas"],
                generated_by="echoforge-core-test",
                generated_at="2026-05-18T00:00:00Z",
            ),
            "license": LicenseInfo(spdx_id="CC0-1.0"),
            "validation": ValidationInfo(
                tier="basic",
                status="pass",
                uncertainty_score=0.25,
                checks=[ValidationCheck(name="shape", status="pass", message="ok")],
            ),
        }

    def test_schema_catalog_and_common_schema_parse(self) -> None:
        catalog = load_json(REPO_ROOT / "contracts" / "schema_catalog.json")
        self.assertEqual(len(catalog), 12)
        for entry in catalog:
            schema = load_json(REPO_ROOT / entry["schema_file"])
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(schema["type"], "object")
            self.assertIn("kind", schema["properties"])
            self.assertIn("provenance", schema["properties"])

    def test_object_and_dataset_finalize(self) -> None:
        card = ObjectCard(
            public_proxy_id="proxy-1",
            display_name="Proxy 1",
            object_family="airframe",
            geometry_variant="baseline",
            material_variant="default",
            dimensions_m=Vector3(1.0, 2.0, 3.0),
            tags=["proxy"],
            **self.sample_common(),
        ).finalize()
        self.assertTrue(card.id.startswith("ef:object_card:proxy-1:"))
        self.assertEqual(card.kind, "object_card")

        dataset = DatasetCard(
            public_proxy_id="dataset-1",
            dataset_name="dataset-1",
            source_campaign_ids=["camp-1"],
            splits=SplitCounts(train=10, validation=2, test=2),
            **self.sample_common(),
        ).finalize()
        self.assertTrue(dataset.id.startswith("ef:dataset_card:dataset-1:"))
        self.assertEqual(dataset.splits.train, 10)

    def test_material_round_trip(self) -> None:
        material = MaterialCard(
            public_proxy_id="mat-1",
            material_name="mat-1",
            material_family="composite",
            frequency_range_hz=NumericRange(1.0, 10.0),
            permittivity=ComplexScalar(2.0, 0.1),
            conductivity_s_per_m=0.01,
            roughness_m=0.001,
            **self.sample_common(),
        ).finalize()
        self.assertEqual(material.kind, "material_card")
        data = material.to_dict()
        self.assertEqual(data["id"], material.id)
        self.assertEqual(data["provenance"]["fingerprint_sha256"], material.provenance.fingerprint_sha256)


if __name__ == "__main__":
    unittest.main()
