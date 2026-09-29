import unittest

from safety_check import (
    bbox_1000_to_original,
    build_single_vision_system_prompt,
    build_peer_vision_system_prompt,
    issues_are_related,
    merge_model_issues,
    model_judged_report,
    normalize_fact_result,
    normalize_bbox_1000,
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


class SingleModelCoordinateTests(unittest.TestCase):
    def test_single_model_prompt_requires_full_image_normalized_boxes(self) -> None:
        prompt = build_single_vision_system_prompt(None)
        self.assertIn("bbox_2d_1000", prompt)
        self.assertIn("0到1000归一化整数", prompt)
        self.assertIn("禁止猜坐标", prompt)

    def test_normalized_box_maps_to_original_pixel_xywh(self) -> None:
        bbox = bbox_1000_to_original([100, 200, 500, 700], 2000, 1000)
        self.assertEqual(
            bbox,
            {
                "x": 200,
                "y": 200,
                "width": 800,
                "height": 500,
                "image_width": 2000,
                "image_height": 1000,
            },
        )

    def test_invalid_or_unlocalizable_box_returns_none(self) -> None:
        self.assertIsNone(bbox_1000_to_original(None, 1000, 500))
        self.assertIsNone(bbox_1000_to_original([400, 300, 200, 600], 1000, 500))
        self.assertIsNone(normalize_bbox_1000([400, 300, 200, 600]))
        self.assertEqual(normalize_bbox_1000([-5, 10, 1005, 900]), [0, 10, 1000, 900])

    def test_clear_issue_requires_complete_visibility_and_direct_evidence(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "issues": [
                    {
                        "status": "CLEAR",
                        "item": "局部可见的风险线索",
                        "target": "局部目标",
                        "evidence": "只看见目标的一部分",
                        "citation_key": "UNKNOWN",
                        "confidence": 0.9,
                        "target_visibility": "partial",
                        "evidence_level": "direct",
                        "bbox_2d_1000": [100, 100, 300, 300],
                    }
                ],
            },
            original_size=(1000, 1000),
        )
        self.assertEqual(result["issues"][0]["status"], "SUSPECTED")
        self.assertTrue(result["issues"][0]["needs_review"])

    def test_clear_issue_with_uncertain_wording_is_not_assessable(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "issues": [
                    {
                        "status": "CLEAR",
                        "item": "设备状态异常",
                        "target": "设备开关",
                        "evidence": "开关位置可见，但无法确认是否处于断开状态。",
                        "citation_key": "UNKNOWN",
                        "confidence": 0.8,
                        "target_visibility": "complete",
                        "evidence_level": "direct",
                        "bbox_2d_1000": [100, 100, 300, 300],
                    }
                ],
            },
            original_size=(1000, 1000),
        )
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")
        self.assertTrue(result["issues"][0]["needs_review"])

    def test_normalized_result_and_report_include_standard_bbox_fields(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "issues": [
                    {
                        "status": "CLEAR",
                        "item": "通道被占用",
                        "target": "通道材料",
                        "position": "画面下方",
                        "evidence": "材料堆在通道范围内",
                        "citation_key": "UNKNOWN",
                        "confidence": 0.8,
                        "target_visibility": "complete",
                        "evidence_level": "direct",
                        "bbox_2d_1000": [100, 200, 500, 700],
                    }
                ],
            },
            original_size=(2000, 1000),
        )
        issue = result["issues"][0]
        self.assertEqual(issue["bbox_original_px"]["x"], 200)
        self.assertEqual(issue["bbox_original_px"]["width"], 800)
        report = model_judged_report(
            {**result, "image_width": 2000, "image_height": 1000},
            {"width": 2000, "height": 1000, "note": "图片质量基本可用"},
            elapsed=1.2,
        )
        self.assertIn("报告格式版本：1.0", report)
        self.assertIn("x=200, y=200, width=800, height=500", report)

    def test_no_risk_report_does_not_create_a_fake_issue(self) -> None:
        result = normalize_fact_result(
            {"scene_type": "GENERAL_SITE", "issues": []},
            original_size=(1280, 720),
        )
        report = model_judged_report(
            result,
            {"width": 1280, "height": 720, "note": "图片质量基本可用"},
        )
        self.assertEqual(result["issues"], [])
        self.assertIn("未发现可直接确认的明显违规", report)
        self.assertNotIn("1. 不可判断", report)


if __name__ == "__main__":
    unittest.main()
