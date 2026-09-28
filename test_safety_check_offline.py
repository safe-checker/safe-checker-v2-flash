import unittest

from safety_check import (
    build_peer_vision_system_prompt,
    issues_are_related,
    merge_model_issues,
)


def make_issue(
    risk_key: str,
    item: str,
    target: str,
    evidence: str,
    *,
    status: str = "SUSPECTED",
    confidence: float = 0.8,
) -> dict:
    return {
        "status": status,
        "item": item,
        "risk_key": risk_key,
        "target": target,
        "position": target,
        "evidence": evidence,
        "citation_key": "UNKNOWN",
        "confidence": confidence,
        "target_visibility": "complete",
        "evidence_level": "direct",
        "citation_confidence": "unknown",
    }


class SafetyFusionOfflineTests(unittest.TestCase):
    def test_peer_prompt_is_coverage_oriented(self) -> None:
        prompt = build_peer_vision_system_prompt("AUTO")
        self.assertNotIn("Top2", prompt)
        self.assertIn("最多4个", prompt)
        self.assertIn('"target"', prompt)
        self.assertIn('"position"', prompt)

    def test_same_regulation_different_risks_are_not_merged(self) -> None:
        left = make_issue(
            "电缆拖地",
            "电缆沿地面敷设",
            "箱底电缆",
            "箱底电缆连续接触地面",
        )
        right = make_issue(
            "配电箱防护不足",
            "箱体防护不足",
            "配电箱箱体",
            "箱体侧面未见完整防雨防尘措施",
        )
        left["citation_key"] = "SAME_ARTICLE"
        right["citation_key"] = "SAME_ARTICLE"
        self.assertFalse(issues_are_related(left, right))

    def test_same_risk_and_target_are_merged(self) -> None:
        left = make_issue(
            "电缆拖地",
            "电缆沿地面敷设",
            "箱底电缆",
            "电缆连续接触地面",
        )
        right = make_issue(
            "电缆拖地",
            "电缆拖地敷设",
            "箱底电缆",
            "电缆散落在地面",
        )
        self.assertTrue(issues_are_related(left, right))

    def test_minority_candidate_is_retained_and_confidence_is_discounted(self) -> None:
        issue = make_issue(
            "箱体防护不足",
            "箱体防护措施不足",
            "画面右侧箱体",
            "箱体侧面未见完整防护结构",
            confidence=0.95,
        )
        model_sets = [
            ("deepseek", [issue]),
            ("qwen", []),
            ("glm", []),
            ("kimi", []),
            ("doubao", []),
        ]
        roster = [
            {"source": source, "model": source, "status": "completed"}
            for source, _ in model_sets
        ]
        merged = merge_model_issues(model_sets, total_models=5, model_roster=roster)
        self.assertEqual(len(merged), 1)
        result = merged[0]
        self.assertEqual(result["vote_support"], 1)
        self.assertEqual(result["vote_total"], 5)
        self.assertEqual(result["consensus_score"], 0.2)
        self.assertLess(result["vote_confidence"], result["model_evidence_confidence"])
        self.assertEqual(result["vote_label"], "单模型发现，待人工复核")
        self.assertEqual(sum(vote["vote"] == "支持" for vote in result["named_votes"]), 1)
        self.assertEqual(sum(vote["vote"] == "未支持" for vote in result["named_votes"]), 4)

    def test_majority_and_minority_lanes_are_both_visible(self) -> None:
        majority = make_issue(
            "电缆拖地",
            "电缆沿地面敷设",
            "箱底电缆",
            "电缆连续接触地面",
            status="CLEAR",
            confidence=0.9,
        )
        minority = make_issue(
            "箱体防护不足",
            "箱体防护措施不足",
            "画面右侧箱体",
            "箱体侧面未见完整防护结构",
            confidence=0.8,
        )
        model_sets = [
            ("deepseek", [majority]),
            ("qwen", [majority]),
            ("glm", [majority]),
            ("kimi", [minority]),
            ("doubao", []),
        ]
        roster = [
            {"source": source, "model": source, "status": "completed"}
            for source, _ in model_sets
        ]
        merged = merge_model_issues(model_sets, total_models=5, model_roster=roster)
        self.assertEqual(len(merged), 2)
        self.assertEqual({item["risk_key"] for item in merged}, {"电缆拖地", "箱体防护不足"})
        minority_result = next(item for item in merged if item["risk_key"] == "箱体防护不足")
        self.assertFalse(minority_result["vote_majority"])
        self.assertEqual(minority_result["vote_label"], "单模型发现，待人工复核")

    def test_incomplete_roster_discounts_consensus_confidence(self) -> None:
        issue = make_issue(
            "电缆拖地",
            "电缆沿地面敷设",
            "箱底电缆",
            "电缆连续接触地面",
            status="CLEAR",
            confidence=0.95,
        )
        model_sets = [
            ("deepseek", [issue]),
            ("qwen", [issue]),
        ]
        roster = [
            {"source": "deepseek", "model": "deepseek-flash", "status": "completed"},
            {"source": "qwen", "model": "qwen3.8-flash", "status": "completed"},
            {"source": "glm", "model": "glm-5.3-flash", "status": "failed"},
            {"source": "kimi", "model": "kimi-k2.6", "status": "failed"},
            {"source": "doubao", "model": "doubao", "status": "failed"},
        ]
        merged = merge_model_issues(model_sets, total_models=5, model_roster=roster)
        result = merged[0]
        self.assertEqual(result["vote_support"], 2)
        self.assertEqual(result["vote_completed"], 2)
        self.assertEqual(result["model_availability_rate"], 0.4)
        self.assertLess(result["vote_confidence"], 0.6)
        self.assertEqual(result["vote_label"], "少数模型发现，待人工复核")


if __name__ == "__main__":
    unittest.main()
