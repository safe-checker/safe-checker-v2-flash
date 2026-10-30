import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import asyncio
import base64
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from PIL import Image

from safety_check import (
    _backfill_regulation_from_catalog,
    apply_static_image_scope_gate,
    bbox_1000_to_original,
    build_single_vision_system_prompt,
    build_vision_payload,
    call_deepseek,
    is_transient_model_error,
    ModelBudgetExhausted,
    ModelQuotaError,
    emit_cli_error,
    expand_compact_result,
    extract_json_object,
    model_judged_report,
    normalize_fact_result,
    normalize_bbox_1000,
    original_image_size,
    parse_detail_crop,
    prepare_detail_crop_data_url,
    prepare_image_data_url,
    select_auto_crop_from_image,
)
from verified_regulations import verify_regulation


class SingleModelCoordinateTests(unittest.TestCase):
    def test_cli_error_is_standard_json_by_default(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            emit_cli_error(
                SimpleNamespace(json_output=True),
                "timeout",
                "request timed out",
                8.01,
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["format_version"], "1.3")
        self.assertEqual(payload["error"]["type"], "timeout")

    def test_reflective_vest_claim_requires_visible_task_hazard(self) -> None:
        status = apply_static_image_scope_gate(
            {
                "assessment_type": "PPE",
                "ppe_item": "reflective_vest",
                "risk_key": "NO_REFLECTIVE_VEST",
                "item": "高处作业未穿反光背心",
                "target": "作业人员",
                "evidence": "人员躯干未见反光背心",
            },
            "CLEAR",
        )
        self.assertEqual(status, "SUSPECTED")

    def test_reflective_vest_claim_cannot_self_prove_low_visibility(self) -> None:
        status = apply_static_image_scope_gate(
            {
                "assessment_type": "PPE",
                "ppe_item": "reflective_vest",
                "risk_key": "REFLECTIVE_VEST_NOT_STANDARD",
                "item": "反光背心穿戴不规范",
                "target": "作业人员",
                "evidence": "施工现场可见度较低，背心没有明显反光",
                "person_ids": ["P1"],
                "object_ids": ["O1"],
            },
            "CLEAR",
            people_by_id={"P1": {"action": "搬运钢筋"}},
            objects_by_id={"O1": {"object_type": "静止货车", "observed_state": "停放"}},
        )
        self.assertEqual(status, "SUSPECTED")

    def test_reflective_strip_performance_is_not_assessable_from_daylight_photo(self) -> None:
        status = apply_static_image_scope_gate(
            {
                "assessment_type": "PPE",
                "ppe_item": "reflective_vest",
                "risk_key": "REFLECTIVE_STRIP_MISSING",
                "item": "背心未见反光条",
                "target": "作业人员",
                "evidence": "蓝色背心未出现强反光",
            },
            "CLEAR",
        )
        self.assertEqual(status, "NOT_ASSESSABLE")

    def test_vision_payload_enforces_json_object_response(self) -> None:
        payload = build_vision_payload(
            image_data_url="data:image/jpeg;base64,AA==",
            system_prompt="system",
            user_text="user",
            model="deepseek-flash",
            image_detail="low",
            max_tokens=1000,
            temperature=0.0,
            reasoning_effort="none",
        )
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_vision_payload_can_add_one_detail_crop_in_same_call(self) -> None:
        payload = build_vision_payload(
            image_data_url="data:image/jpeg;base64,FULL",
            detail_image_data_url="data:image/jpeg;base64,CROP",
            detail_crop_box_1000=[100, 200, 700, 800],
            system_prompt="system",
            user_text="user",
            model="deepseek-flash",
            image_detail="low",
            max_tokens=1000,
            temperature=0.0,
            reasoning_effort="none",
        )
        content = payload["messages"][1]["content"]
        self.assertEqual(
            [item["type"] for item in content],
            ["text", "image_url", "text", "image_url"],
        )
        self.assertIn("空间关系", content[2]["text"])
        self.assertIn("[100, 200, 700, 800]", content[2]["text"])

    def test_detail_crop_uses_original_image_normalized_grid(self) -> None:
        self.assertEqual(
            parse_detail_crop("100,200,700,800", (2000, 1000)),
            (200, 200, 1400, 800),
        )

    def test_detail_crop_rejects_invalid_or_outside_coordinates(self) -> None:
        for value in ("nan,0,100,100", "0,0,inf,100", "-1,0,500,500",
                      "0,0,1001,500", "500,0,100,100", "0,0,0,100", "1,2,3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_detail_crop(value, (2000, 1000))

    def test_tiny_images_never_crop_outside_original_bounds(self) -> None:
        self.assertEqual(parse_detail_crop("0,0,1000,1000", (1, 1)), (0, 0, 1, 1))
        self.assertEqual(select_auto_crop_from_image(Image.new("RGB", (1, 1))), (0, 0, 1, 1))

    def test_detail_crop_uses_exif_oriented_image_and_keeps_real_detail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (200, 100), "red")
            image.paste("blue", (100, 0, 200, 100))
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif, quality=100)
            size = original_image_size(path)
            self.assertEqual(size, (100, 200))
            box = parse_detail_crop("0,0,1000,500", size)
            self.assertEqual(box, (0, 0, 100, 100))
            full = prepare_image_data_url(path, max_edge=160, quality=90)
            crop = prepare_detail_crop_data_url(path, box, max_edge=80, quality=90)
            with Image.open(io.BytesIO(base64.b64decode(full.split(",", 1)[1]))) as encoded:
                self.assertEqual(encoded.size, (80, 160))
            with Image.open(io.BytesIO(base64.b64decode(crop.split(",", 1)[1]))) as encoded:
                self.assertEqual(encoded.size, (80, 80))
                red, green, blue = encoded.getpixel((40, 40))
                self.assertGreater(red, 200)
                self.assertLess(blue, 30)
            with self.assertRaises(ValueError):
                prepare_detail_crop_data_url(path, (0, 0, 101, 100), max_edge=80, quality=90)

    def test_crop_main_decodes_original_only_once(self) -> None:
        from safety_check import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (200, 100)).save(path)
            output = io.StringIO()
            with patch("sys.argv", ["safety_check.py", str(path), "--dry-run", "--detail-crop", "0,0,500,1000"]), \
                    patch.dict(os.environ, {"MAX_IMAGE_EDGE": "100", "DETAIL_CROP_MAX_EDGE": "100"}), \
                    patch("safety_check.Image.open", wraps=Image.open) as opened, redirect_stdout(output):
                self.assertEqual(main(), 0)
            self.assertEqual(opened.call_count, 1)
            result = json.loads(output.getvalue())
            self.assertEqual(result["image_inputs"], 2)
            self.assertEqual(result["detail_crop"]["bbox_original_px"], {
                "x": 0, "y": 0, "width": 100, "height": 100,
            })

    def test_request_deadline_cancels_slow_provider_without_retry(self) -> None:
        client = AsyncMock()
        client.__aenter__.return_value = client
        cancelled = []

        async def delayed_response(*args, **kwargs):
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        client.post.side_effect = delayed_response
        with patch("safety_check.httpx.AsyncClient", return_value=client), self.assertRaises(httpx.TimeoutException):
            call_deepseek({}, "test-key", "https://example.invalid", timeout_seconds=0.01)
        self.assertEqual(cancelled, [True])
        self.assertEqual(client.post.await_count, 1)

    def test_single_model_prompt_requires_full_image_normalized_boxes(self) -> None:
        prompt = build_single_vision_system_prompt(None, compact=False)
        self.assertIn("bbox_2d_1000", prompt)
        self.assertIn("0到1000归一化整数", prompt)
        self.assertIn("禁止猜坐标", prompt)
        self.assertIn('"regulation"', prompt)
        self.assertIn('"visual_cues"', prompt)
        self.assertIn('"persons"', prompt)
        self.assertIn('"objects"', prompt)
        self.assertIn('"claim_validation"', prompt)
        self.assertIn('"body_visibility"', prompt)
        self.assertIn('"ppe_states"', prompt)
        self.assertIn('"spatial_relation"', prompt)
        self.assertIn("抬腿、跨越材料", prompt)
        self.assertIn("靠近钢丝绳、车辆或静置材料不等于处于吊物下方", prompt)
        self.assertIn("至少2条相互独立的直接视觉线索", prompt)
        self.assertIn("法规匹配范围不得局限于临时用电", prompt)
        self.assertIn("禁止编造法规", prompt)
        self.assertIn("只有目录核验成功的条款才能作为正式引用", prompt)
        self.assertIn("箱门关闭但未上锁", prompt)
        self.assertIn("单一深色区域、土壤色差、阴影", prompt)
        self.assertIn("已有横杆不得", prompt)
        self.assertIn("说明页属于设备文件", prompt)
        self.assertIn("仅因人员在工地、高处或车辆附近不得判定", prompt)
        self.assertIn("未出现强反光不能证明", prompt)
        self.assertIn("禁止使用“未佩戴或佩戴不规范”", prompt)
        self.assertIn("最多8个", prompt)
        self.assertIn("最多输出3条", prompt)
        self.assertNotIn("citation_key", prompt)

    def make_state_claim(
        self,
        claim_type: str,
        *,
        validation: dict,
        item: str = "对象状态异常",
        risk_key: str = "OBJECT_STATE_RISK",
    ) -> dict:
        return {
            "scene_type": "GENERAL_SITE",
            "objects": [{
                "object_id": "O1",
                "object_type": "测试对象",
                "bbox_2d_1000": [100, 100, 500, 700],
                "visibility": "complete",
                "identity_cues": ["对象轮廓", "用途标识"],
                "observed_state": "测试状态",
            }],
            "issues": [{
                "status": "CLEAR",
                "risk_category": "GENERAL_SITE",
                "risk_key": risk_key,
                "assessment_type": "OBJECT_STATE",
                "claim_type": claim_type,
                "object_ids": ["O1"],
                "item": item,
                "target": "O1",
                "evidence": "对象状态看似异常",
                "visual_cues": ["对象轮廓可见", "局部状态可见"],
                "claim_validation": validation,
                "confidence": 0.9,
                "target_visibility": "complete",
                "evidence_level": "direct",
                "regulation": {},
                "bbox_2d_1000": [100, 100, 500, 700],
            }],
        }

    def test_open_door_claim_requires_complete_component_boundary(self) -> None:
        data = self.make_state_claim(
            "OPEN_CLOSED",
            item="配电箱主箱门打开",
            risk_key="DISTRIBUTION_BOX_MAIN_DOOR_OPEN",
            validation={
                "target_identity_confirmed": True,
                "component_boundary_complete": False,
                "inspection_region_complete": True,
                "positive_cues": ["可见门缝", "可见圆形插口"],
                "alternative_explanation": "看到的是插座盖或箱体侧面",
                "alternative_excluded": True,
            },
        )
        result = normalize_fact_result(data, original_size=(1000, 1000))
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")

    def test_shadow_cannot_support_clear_water_claim(self) -> None:
        data = self.make_state_claim(
            "MATERIAL_SURFACE",
            item="基坑底部积水",
            risk_key="PIT_WATER_ACCUMULATION",
            validation={
                "target_identity_confirmed": True,
                "component_boundary_complete": True,
                "inspection_region_complete": True,
                "positive_cues": ["坑底存在深色区域"],
                "alternative_explanation": "可能是土壤色差或阴影",
                "alternative_excluded": False,
            },
        )
        result = normalize_fact_result(data, original_size=(1000, 1000))
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")

    def test_missing_guardrail_requires_complete_inspection_region(self) -> None:
        data = self.make_state_claim(
            "PRESENCE_ABSENCE",
            item="移动式操作平台缺少护栏",
            risk_key="MOBILE_PLATFORM_GUARDRAIL_MISSING",
            validation={
                "target_identity_confirmed": True,
                "component_boundary_complete": True,
                "inspection_region_complete": False,
                "positive_cues": ["平台边界局部可见", "一侧立杆可见"],
                "alternative_explanation": "护栏可能被人员或拍摄角度遮挡",
                "alternative_excluded": True,
            },
        )
        result = normalize_fact_result(data, original_size=(1000, 1000))
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")

    def test_fixed_equipment_document_is_not_clear_foreign_object(self) -> None:
        data = self.make_state_claim(
            "FOREIGN_OBJECT",
            item="配电箱内放置杂物",
            risk_key="FOREIGN_OBJECT_IN_DISTRIBUTION_BOX",
            validation={
                "target_identity_confirmed": False,
                "component_boundary_complete": True,
                "inspection_region_complete": True,
                "positive_cues": ["可见纸张", "纸张位于箱门内侧"],
                "alternative_explanation": "纸张可能是固定接线图或说明页",
                "alternative_excluded": False,
            },
        )
        result = normalize_fact_result(data, original_size=(1000, 1000))
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")

    def test_adjacent_object_does_not_prove_spatial_occupancy(self) -> None:
        data = self.make_state_claim(
            "SPATIAL_OCCUPANCY",
            item="配电箱操作空间被占用",
            risk_key="DISTRIBUTION_BOX_SPACE_BLOCKED",
            validation={
                "target_identity_confirmed": True,
                "component_boundary_complete": True,
                "inspection_region_complete": True,
                "positive_cues": ["配电箱可见", "纸箱位于附近"],
                "alternative_explanation": "纸箱可能仅相邻且未进入操作区",
                "alternative_excluded": True,
                "subject_anchor": "纸箱边界",
                "reference_anchor": "配电箱操作区边界",
                "relation": "ADJACENT",
            },
        )
        result = normalize_fact_result(data, original_size=(1000, 1000))
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")

    def test_report_scene_follows_the_hint_because_the_hint_selects_the_checklist(self) -> None:
        """Since the scene hint picks the checklist, it also defines what was inspected.

        The model's own guess is kept separately so a mismatch stays visible
        instead of silently relabelling the report.
        """
        result = normalize_fact_result(
            {
                "scene_type": "CONSTRUCTION_HOIST",
                "scene": "施工升降机",
                "scene_confidence": 0.92,
                "scene_evidence": ["可见吊笼门", "可见自动开门标识"],
                "issues": [],
            },
            scene_override="施工现场临时用电",
            original_size=(1000, 1000),
        )
        self.assertEqual(result["scene_type"], "TEMPORARY_ELECTRICITY")
        self.assertEqual(result["scene"], "临时用电/配电箱")
        self.assertEqual(result["scene_hint"], "施工现场临时用电")
        self.assertEqual(result["model_detected_scene"], "CONSTRUCTION_HOIST")
        self.assertEqual(result["model_scene_confidence"], 0.92)

    def test_old_electric_standard_is_replaced_by_current_verified_clause(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "TEMPORARY_ELECTRICITY",
                "issues": [
                    {
                        "status": "CLEAR",
                        "risk_category": "TEMPORARY_ELECTRICITY",
                        "risk_key": "CABLE_ON_GROUND",
                        "item": "电缆沿地面明设",
                        "target": "地面电缆",
                        "evidence": "电缆连续贴地敷设",
                        "confidence": 0.9,
                        "target_visibility": "complete",
                        "evidence_level": "direct",
                        "citation_confidence": "high",
                        "regulation": {
                            "standard_name": "施工现场临时用电安全技术规范",
                            "standard_code": "JGJ 46-2005",
                            "article": "第7.2.3条",
                            "clause_summary": "电缆不得沿地面明设。",
                            "match_reason": "电缆贴地。",
                            "match_status": "MATCHED",
                        },
                        "bbox_2d_1000": [100, 100, 300, 500],
                    }
                ],
            },
            original_size=(1000, 1000),
        )
        issue = result["issues"][0]
        self.assertEqual(issue["model_regulation_candidate"]["standard_code"], "JGJ 46-2005")
        self.assertEqual(issue["verified_regulation"]["standard_code"], "JGJ/T 46-2024")
        self.assertEqual(issue["verified_regulation"]["article"], "第6.2.3条")
        self.assertEqual(issue["regulation_match"]["method"], "risk_key")
        self.assertFalse(issue["verified_regulation"]["verification_required"])

    def test_exact_current_clause_match(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "LIFTING_OPERATIONS",
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "LIFTING_OPERATIONS",
                    "risk_key": "UNRECOGNIZED_KEY",
                    "item": "吊物下方人员停留",
                    "target": "吊物下方人员",
                    "evidence": "人员位于悬吊物正下方",
                    "confidence": 0.9,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "citation_confidence": "high",
                    "regulation": {
                        "standard_name": "建筑与市政施工现场安全卫生与职业健康通用规范",
                        "standard_code": "GB 55034-2022",
                        "article": "3.4.1",
                        "clause_summary": "严禁任何人在吊物下停留或通过。",
                        "match_status": "MATCHED",
                    },
                    "bbox_2d_1000": [100, 100, 300, 500],
                }],
            },
            original_size=(1000, 1000),
        )
        issue = result["issues"][0]
        self.assertEqual(issue["regulation_match"]["method"], "exact_code_article")
        self.assertEqual(issue["verified_regulation"]["regulation_id"], "LIFTING_EXCLUSION_ZONE")

    def test_mobile_platform_guardrail_matches_cross_scene_clause(self) -> None:
        verified, match = verify_regulation(
            candidate={},
            risk_key="MOBILE_PLATFORM_GUARDRAIL_MISSING",
            item_text="移动式操作平台缺少临边防护",
            target="移动式操作平台",
            evidence="平台边缘缺少防护栏杆",
            scene_type="WORK_AT_HEIGHT",
        )
        self.assertEqual(verified["regulation_id"], "HEIGHT_PLATFORM_EDGE")
        self.assertEqual(verified["article"], "第3.2.5条")
        self.assertEqual(match["method"], "risk_key")

    def test_lifting_sling_defect_matches_verified_clause(self) -> None:
        verified, match = verify_regulation(
            candidate={},
            risk_key="SLING_OR_SHACKLE_DEFECT",
            item_text="吊索具存在明显缺陷",
            target="吊带和卸扣",
            evidence="吊带存在清晰破损",
            scene_type="LIFTING_OPERATIONS",
        )
        self.assertEqual(verified["regulation_id"], "LIFTING_SLING_CONDITION")
        self.assertEqual(verified["article"], "第3.4.2条")
        self.assertEqual(match["method"], "risk_key")

    def test_construction_hoist_guard_matches_machine_clause(self) -> None:
        verified, match = verify_regulation(
            candidate={},
            risk_key="CONSTRUCTION_HOIST_GUARD_MISSING",
            item_text="施工升降机防护装置缺失",
            target="施工升降机围栏",
            evidence="围栏防护存在明确缺口",
            scene_type="CONSTRUCTION_HOIST",
        )
        self.assertEqual(verified["regulation_id"], "MACHINE_GUARD")
        self.assertEqual(verified["article"], "第3.6.3条")
        self.assertEqual(match["method"], "risk_key")

    def test_unrelated_scene_does_not_match_risk_key(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "WORK_AT_HEIGHT",
                "issues": [{
                    "status": "SUSPECTED",
                    "risk_category": "WORK_AT_HEIGHT",
                    "risk_key": "CABLE_ON_GROUND",
                    "item": "不相关风险",
                    "target": "目标",
                    "evidence": "证据不足",
                    "confidence": 0.5,
                    "target_visibility": "partial",
                    "evidence_level": "partial",
                    "regulation": {},
                    "bbox_2d_1000": [100, 100, 300, 500],
                }],
            },
            original_size=(1000, 1000),
        )
        issue = result["issues"][0]
        self.assertIsNone(issue["verified_regulation"])
        self.assertFalse(issue["regulation_match"]["verified"])

    def test_generic_keywords_do_not_force_an_unrelated_clause(self) -> None:
        verified, match = verify_regulation(
            candidate={"standard_code": "JGJ 46-2005", "article": "第8.1.5条"},
            risk_key="NO_ISOLATION_AROUND_DISTRIBUTION_BOX",
            item_text="配电箱周边未设置围栏或警示隔离",
            target="配电箱周围区域",
            evidence="周围未见防护围栏、警示带或隔离设施",
            scene_type="TEMPORARY_ELECTRICITY",
        )
        self.assertIsNone(verified)
        self.assertEqual(match["method"], "none")

    def test_keyword_match_is_not_allowed_to_remain_clear(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "TEMPORARY_ELECTRICITY",
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "TEMPORARY_ELECTRICITY",
                    "risk_key": "UNKNOWN_CABLE_RISK",
                    "item": "现场情况",
                    "target": "对象",
                    "evidence": "配电箱 操作空间 通道 堆放",
                    "visual_cues": ["配电箱附近有物品", "操作空间和通道出现堆放"],
                    "confidence": 0.9,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "regulation": {},
                    "bbox_2d_1000": [100, 100, 300, 500],
                }],
            },
            original_size=(1000, 1000),
        )
        issue = result["issues"][0]
        self.assertEqual(issue["regulation_match"]["method"], "keyword")
        self.assertTrue(issue["has_verified_regulation"])
        self.assertEqual(issue["status"], "SUSPECTED")
        self.assertIn("关键词命中", issue["review_reason"])

    def test_verified_regulation_exposes_audit_metadata(self) -> None:
        regulation, match = verify_regulation(
            candidate={},
            risk_key="CABLE_ON_GROUND",
            item_text="电缆沿地面明设",
            target="施工现场电缆",
            evidence="电缆与地面连续接触",
            scene_type="TEMPORARY_ELECTRICITY",
        )
        self.assertTrue(match["verified"])
        self.assertEqual(regulation["version_status"], "current")
        self.assertTrue(regulation["citation_ready"])
        self.assertTrue(regulation["catalog_verified_as_of"])
        self.assertIn("clause_text", regulation)

    def test_clear_claim_without_two_visual_cues_is_downgraded(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "GENERAL_SITE",
                    "risk_key": "UNKNOWN_VISIBLE_RISK",
                    "item": "设备状态异常",
                    "target": "设备",
                    "evidence": "设备状态直接可见",
                    "confidence": 0.9,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "visual_cues": ["仅有一条视觉线索"],
                    "regulation": {},
                    "bbox_2d_1000": [100, 100, 300, 500],
                }],
            },
            original_size=(1000, 1000),
        )
        self.assertEqual(result["issues"][0]["status"], "SUSPECTED")

    def test_glove_violation_requires_both_hands_completely_visible(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "persons": [{
                    "person_id": "P1",
                    "bbox_2d_1000": [100, 100, 400, 900],
                    "body_visibility": {
                        "head": "complete",
                        "left_hand": "complete",
                        "right_hand": "occluded",
                        "torso": "complete",
                        "waist": "complete",
                        "left_foot": "complete",
                        "right_foot": "complete",
                    },
                    "ppe_states": {"gloves": "not_worn"},
                    "support_surface": "地面",
                    "elevation_state": "GROUND_LEVEL",
                    "spatial_cues": ["左脚接触地面", "右脚接触地面"],
                }],
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "GENERAL_SITE",
                    "risk_key": "GLOVES_NOT_WORN",
                    "assessment_type": "PPE",
                    "person_ids": ["P1"],
                    "ppe_item": "gloves",
                    "focus_body_parts": ["left_hand", "right_hand"],
                    "item": "人员未佩戴防护手套",
                    "target": "P1双手",
                    "evidence": "手部位于钢筋附近",
                    "visual_cues": ["左手可见", "右手局部被遮挡"],
                    "confidence": 0.9,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "regulation": {},
                    "bbox_2d_1000": [100, 100, 400, 900],
                }],
            },
            original_size=(1000, 1000),
        )
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")
        self.assertEqual(result["persons"][0]["body_visibility"]["right_hand"], "occluded")

    def test_left_and_right_gloves_are_normalized_separately(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "persons": [{
                    "person_id": "P1",
                    "bbox_2d_1000": [100, 100, 400, 900],
                    "body_visibility": {
                        "left_hand": "complete",
                        "right_hand": "complete",
                    },
                    "hand_ppe_states": {
                        "left_glove": "worn",
                        "right_glove": "not_worn",
                    },
                    "ppe_states": {},
                }],
                "issues": [],
            },
            original_size=(1000, 1000),
        )
        person = result["persons"][0]
        self.assertEqual(person["hand_ppe_states"]["left_glove"], "worn")
        self.assertEqual(person["hand_ppe_states"]["right_glove"], "not_worn")
        self.assertEqual(person["ppe_states"]["gloves"], "not_worn")

    def test_suspected_missing_ppe_with_occluded_body_part_is_not_assessable(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "persons": [{
                    "person_id": "P1",
                    "body_visibility": {
                        "left_hand": "partial",
                        "right_hand": "partial",
                    },
                    "hand_ppe_states": {
                        "left_glove": "uncertain",
                        "right_glove": "uncertain",
                    },
                    "ppe_states": {"gloves": "uncertain"},
                }],
                "issues": [{
                    "status": "SUSPECTED",
                    "risk_category": "GENERAL_SITE",
                    "risk_key": "MISSING_HAND_PROTECTION",
                    "assessment_type": "PPE",
                    "person_ids": ["P1"],
                    "ppe_item": "gloves",
                    "focus_body_parts": ["left_hand", "right_hand"],
                    "item": "疑似未佩戴防护手套",
                    "target": "P1双手",
                    "evidence": "手部被材料部分遮挡",
                    "visual_cues": ["左手局部可见", "右手局部可见"],
                    "confidence": 0.6,
                    "target_visibility": "partial",
                    "evidence_level": "partial",
                    "regulation": {},
                    "bbox_2d_1000": [100, 100, 400, 900],
                }],
            },
            original_size=(1000, 1000),
        )
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")

    def test_clear_spatial_claim_requires_visible_body_anchor(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "persons": [{
                    "person_id": "P1",
                    "body_visibility": {"left_foot": "partial", "right_foot": "partial"},
                    "ppe_states": {},
                    "elevation_state": "LOW_OBSTACLE",
                }],
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "GENERAL_SITE",
                    "risk_key": "PERSON_ON_UNSTABLE_MATERIAL_STACK",
                    "assessment_type": "SPATIAL",
                    "person_ids": ["P1"],
                    "focus_body_parts": ["left_foot", "right_foot"],
                    "spatial_relation": {
                        "subject_anchor": "双脚",
                        "reference_object": "材料堆",
                        "reference_anchor": "材料堆顶部",
                        "both_anchors_visible": True,
                        "elevation_state": "LOW_OBSTACLE",
                        "reference_state": "STATIC",
                        "person_zone": "ADJACENT",
                        "cues": ["脚部接近材料", "人员位于材料旁"],
                    },
                    "item": "人员站在材料堆上",
                    "target": "P1",
                    "evidence": "脚部接近材料堆",
                    "visual_cues": ["脚部局部可见", "材料堆可见"],
                    "confidence": 0.9,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "regulation": {},
                    "bbox_2d_1000": [100, 100, 400, 900],
                }],
            },
            original_size=(1000, 1000),
        )
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")

    def test_ground_level_step_is_not_clear_high_place_climbing(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "persons": [{
                    "person_id": "P1",
                    "bbox_2d_1000": [100, 100, 400, 900],
                    "body_visibility": {part: "complete" for part in (
                        "head", "left_hand", "right_hand", "torso", "waist", "left_foot", "right_foot"
                    )},
                    "ppe_states": {},
                    "support_surface": "地面及低矮钢筋材料",
                    "elevation_state": "LOW_OBSTACLE",
                    "spatial_cues": ["右脚接触地面", "左脚跨过低矮材料"],
                }],
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "WORK_AT_HEIGHT",
                    "risk_key": "UNSAFE_HIGH_PLACE_CLIMBING",
                    "assessment_type": "SPATIAL",
                    "person_ids": ["P1"],
                    "ppe_item": "not_applicable",
                    "focus_body_parts": ["left_foot", "right_foot"],
                    "spatial_relation": {
                        "subject_anchor": "双脚",
                        "reference_object": "地面材料",
                        "reference_anchor": "材料顶部",
                        "both_anchors_visible": True,
                        "elevation_state": "LOW_OBSTACLE",
                        "cues": ["一脚在地面", "一脚跨过材料"],
                    },
                    "item": "人员高处攀爬",
                    "target": "P1",
                    "evidence": "人员抬腿跨越材料",
                    "visual_cues": ["人员抬起左腿", "右脚接近地面"],
                    "confidence": 0.9,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "regulation": {},
                    "bbox_2d_1000": [100, 100, 400, 900],
                }],
            },
            original_size=(1000, 1000),
        )
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")

    def test_clear_spatial_issue_requires_existing_person_reference(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "LIFTING_OPERATIONS",
                "persons": [],
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "LIFTING_OPERATIONS",
                    "risk_key": "PERSON_IN_LIFTING_DANGER_ZONE",
                    "assessment_type": "SPATIAL",
                    "person_ids": ["P9"],
                    "spatial_relation": {
                        "both_anchors_visible": True,
                        "elevation_state": "GROUND_LEVEL",
                        "cues": ["人员锚点", "吊物锚点"],
                    },
                    "item": "人员进入吊装危险区",
                    "target": "P9",
                    "evidence": "人员与吊物重叠",
                    "visual_cues": ["人员轮廓", "吊物轮廓"],
                    "confidence": 0.9,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "regulation": {},
                    "bbox_2d_1000": [100, 100, 400, 900],
                }],
            },
            original_size=(1000, 1000),
        )
        self.assertEqual(result["issues"][0]["status"], "NOT_ASSESSABLE")

    def test_adjacent_static_material_is_not_person_under_suspended_load(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "LIFTING_OPERATIONS",
                "persons": [{
                    "person_id": "P1",
                    "bbox_2d_1000": [100, 100, 400, 900],
                    "body_visibility": {part: "complete" for part in (
                        "head", "left_hand", "right_hand", "torso", "waist", "left_foot", "right_foot"
                    )},
                    "ppe_states": {},
                    "support_surface": "地面",
                    "elevation_state": "GROUND_LEVEL",
                    "spatial_cues": ["双脚接触地面", "人员靠近静置钢筋"],
                }],
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "LIFTING_OPERATIONS",
                    "risk_key": "PERSON_UNDER_SUSPENDED_LOAD",
                    "assessment_type": "SPATIAL",
                    "person_ids": ["P1"],
                    "spatial_relation": {
                        "subject_anchor": "身体中心",
                        "reference_object": "钢筋",
                        "reference_anchor": "钢筋中心",
                        "both_anchors_visible": True,
                        "elevation_state": "GROUND_LEVEL",
                        "reference_state": "STATIC",
                        "person_zone": "ADJACENT",
                        "cues": ["人员靠近钢筋", "钢丝绳在附近"],
                    },
                    "item": "人员处于吊物下方",
                    "target": "P1",
                    "evidence": "人员靠近钢筋和钢丝绳",
                    "visual_cues": ["人员轮廓可见", "钢筋轮廓可见"],
                    "confidence": 0.9,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "regulation": {},
                    "bbox_2d_1000": [100, 100, 400, 900],
                }],
            },
            original_size=(1000, 1000),
        )
        issue = result["issues"][0]
        self.assertEqual(issue["status"], "NOT_ASSESSABLE")
        self.assertEqual(issue["spatial_relation"]["reference_state"], "STATIC")
        self.assertEqual(issue["spatial_relation"]["person_zone"], "ADJACENT")

    def test_unknown_risk_is_retained_without_verified_clause(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "GENERAL_SITE",
                    "risk_key": "UNKNOWN_NEW_SITE_RISK",
                    "item": "新的现场风险",
                    "target": "未知设备",
                    "evidence": "设备出现清晰异常状态",
                    "confidence": 0.8,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "regulation": {
                        "standard_name": "待确认标准",
                        "standard_code": "UNKNOWN",
                        "article": "",
                        "clause_summary": "",
                        "match_status": "NEEDS_VERIFICATION",
                    },
                    "bbox_2d_1000": [100, 100, 300, 500],
                }],
            },
            original_size=(1000, 1000),
        )
        issue = result["issues"][0]
        self.assertEqual(issue["item"], "新的现场风险")
        self.assertEqual(issue["status"], "SUSPECTED")
        self.assertIsNone(issue["regulation"])
        self.assertIn("人工查阅", issue["rule"])
        self.assertTrue(issue["needs_review"])

    def test_unmatched_complete_model_candidate_is_retained_but_not_verified(self) -> None:
        result = normalize_fact_result(
            {
                "scene_type": "GENERAL_SITE",
                "issues": [{
                    "status": "CLEAR",
                    "risk_category": "GENERAL_SITE",
                    "risk_key": "UNKNOWN_VISUAL_RISK",
                    "item": "未知视觉风险",
                    "target": "设备",
                    "evidence": "设备出现清晰异常",
                    "visual_cues": ["异常线索一", "异常线索二"],
                    "confidence": 0.9,
                    "target_visibility": "complete",
                    "evidence_level": "direct",
                    "regulation": {
                        "standard_name": "候选标准",
                        "standard_code": "GB 00000-2026",
                        "article": "第1.2.3条",
                        "clause_summary": "候选条文内容",
                        "match_status": "NEEDS_VERIFICATION",
                    },
                    "bbox_2d_1000": [100, 100, 300, 500],
                }],
            },
            original_size=(1000, 1000),
        )
        issue = result["issues"][0]
        self.assertEqual(issue["status"], "SUSPECTED")
        self.assertEqual(issue["regulation_status"], "MODEL_CANDIDATE_UNVERIFIED")
        self.assertIn("不可作为正式引用", issue["model_regulation_candidate_text"])
        self.assertIsNone(issue["verified_regulation"])

    def test_pit_edge_risk_key_matches_verified_general_clause(self) -> None:
        regulation, match = verify_regulation(
            candidate={},
            risk_key="NO_EDGE_PROTECTION_FOUNDATION_PIT",
            item_text="基坑临边无防护栏杆",
            target="基坑临边",
            evidence="边缘完整可见且无围护设施",
            scene_type="FOUNDATION_PIT",
        )
        self.assertEqual(regulation["standard_code"], "GB 55034-2022")
        self.assertEqual(regulation["article"], "第3.2.3条")
        self.assertTrue(match["verified"])

    def test_exposed_live_parts_risk_key_matches_current_electrical_clause(self) -> None:
        regulation, match = verify_regulation(
            candidate={},
            risk_key="OPEN_PANEL_LIVE_PARTS_EXPOSED",
            item_text="配电箱内带电部分外露",
            target="配电箱",
            evidence="箱内导体和端子清晰外露",
            scene_type="TEMPORARY_ELECTRICITY",
        )
        self.assertEqual(regulation["standard_code"], "JGJ/T 46-2024")
        self.assertEqual(regulation["article"], "第4.1.11条")
        self.assertTrue(match["verified"])

    def test_pit_support_claim_is_not_clear_without_design_context(self) -> None:
        result = normalize_fact_result(
            self.make_state_claim(
                "MATERIAL_SURFACE",
                validation={
                    "target_identity_confirmed": True,
                    "component_boundary_complete": True,
                    "inspection_region_complete": True,
                    "positive_cues": ["坑壁裸露", "未见支护构件"],
                    "alternative_explanation": "设计可能允许放坡开挖",
                    "alternative_excluded": True,
                    "relation": "SEPARATE",
                },
                item="基坑坑壁未支护",
                risk_key="UNSUPPORTED_PIT_WALL",
            ),
            original_size=(1000, 1000),
        )
        self.assertEqual(result["issues"][0]["status"], "SUSPECTED")

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
        self.assertIn("报告格式版本：1.3", report)
        self.assertIn("本地目录匹配分", report)
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


class SceneFamilyChecklistTests(unittest.TestCase):
    """The merged family checklist and the subject anchors it carries."""

    def test_family_checklist_merges_every_scaffold_subtype(self) -> None:
        from catalogs.scene_checklists import SCENE_CHECKLISTS, SCENE_FAMILY_CHECKLISTS

        family = SCENE_FAMILY_CHECKLISTS["SCAFFOLD_FAMILY"]
        single = SCENE_CHECKLISTS["SCAFFOLD_COUPLER_TYPE"]
        self.assertGreater(family["dimension_count"], single["dimension_count"])
        self.assertEqual(
            set(family["scenes"]),
            {"SCAFFOLD_CANTILEVER", "SCAFFOLD_COUPLER_TYPE",
             "SCAFFOLD_DISC_BUCKLE", "SCAFFOLD_ATTACHED_LIFTING"},
        )
        # 子类特有维度必须写明适用条件，避免在图片里不是该型式时误报
        scoped = [line for line in family["lines"] if "仅" in line and "适用" in line]
        self.assertTrue(scoped)
        for line in scoped:
            self.assertTrue(any(word in line for word in ("悬挑式", "扣件式", "盘扣式", "附着升降式")))

    def test_family_core_skips_dimensions_the_person_protocol_already_covers(self) -> None:
        from catalogs.scene_checklists import GLOBAL_DIMENSIONS, SCENE_FAMILY_CHECKLISTS

        for data in SCENE_FAMILY_CHECKLISTS.values():
            for line in data["core_lines"]:
                for dimension in GLOBAL_DIMENSIONS:
                    self.assertNotIn(dimension, line)

    def test_subject_anchor_is_off_by_default_but_switchable(self) -> None:
        """94 张配对实测：锚点开 81.7%、关 84.0%，方向为负，所以默认关闭。"""
        import os

        from catalogs.scene_checklists import SCENE_CHECKLISTS

        lines = SCENE_CHECKLISTS["TOWER_CRANE"]["lines"]
        self.assertFalse([line for line in lines if line.startswith("- [")])
        previous = os.environ.get("CHECKLIST_SUBJECT_ANCHOR")
        os.environ["CHECKLIST_SUBJECT_ANCHOR"] = "1"
        try:
            from safety_check import build_scene_checklist_section

            section = build_scene_checklist_section("TOWER_CRANE")
        finally:
            if previous is None:
                os.environ.pop("CHECKLIST_SUBJECT_ANCHOR", None)
            else:
                os.environ["CHECKLIST_SUBJECT_ANCHOR"] = previous
        self.assertIn("[塔机吊钩与钩口]", section)

    def test_auto_path_defaults_to_the_subject_list(self) -> None:
        """94 张配对实测：主体清单 72.1% > 场景预览 67.7% > 家族清单 62.9%。"""
        from safety_check import build_scene_checklist_section

        section = build_scene_checklist_section("AUTO")
        self.assertIn("本场景主体清单", section)
        self.assertIn("主体[", section)
        self.assertIn("违规:", section)

    def test_auto_path_family_mode_still_works(self) -> None:
        import os

        from safety_check import build_scene_checklist_section

        previous = os.environ.get("CHECKLIST_AUTO_MODE")
        os.environ["CHECKLIST_AUTO_MODE"] = "family"
        try:
            section = build_scene_checklist_section("AUTO")
        finally:
            if previous is None:
                os.environ.pop("CHECKLIST_AUTO_MODE", None)
            else:
                os.environ["CHECKLIST_AUTO_MODE"] = previous
        self.assertIn("【脚手架】", section)
        self.assertIn("【起重与升降设备】", section)
        self.assertIn("违规:", section)
        # 不逐个场景罗列，避免模型去猜子类
        self.assertNotIn("【扣件式钢管脚手架】", section)

    def test_subject_checklist_covers_every_rule_with_a_subject(self) -> None:
        from catalogs.scene_checklists import SUBJECT_CHECKLIST, SUBJECT_CHECKLIST_LINES
        from catalogs.rulepack_catalog import RULEPACK_ROWS

        self.assertEqual(len(SUBJECT_CHECKLIST), len(SUBJECT_CHECKLIST_LINES))
        # 规则包里的每一条规则都要能挂到一个主体上
        self.assertTrue(all(row.get("triggers") for row in RULEPACK_ROWS))
        dims = {d["dimension"] for group in SUBJECT_CHECKLIST for d in group["dimensions"]}
        expected = {row["judge_dimension"] for row in RULEPACK_ROWS}
        # 规则包里人工整理的维度必须一条不少地留在清单里；
        # 主系统补充的维度会按每主体条数上限截断，所以只要求它们来自补充来源
        extra = {d["dimension"] for group in SUBJECT_CHECKLIST for d in group["dimensions"]
                 if d.get("source") != "rulepack"}
        self.assertTrue(expected.issubset(dims))
        self.assertTrue((dims - expected).issubset(extra))
        # 同一个维度可以挂在多个主体上，所以主体-维度对会多于规则数
        bound = sum(len(group["dimensions"]) for group in SUBJECT_CHECKLIST)
        self.assertGreater(bound, len(expected))
        for line in SUBJECT_CHECKLIST_LINES:
            self.assertTrue(line.startswith("- 主体["))

    def test_subject_mode_does_not_require_a_scene_label(self) -> None:
        import os
        from safety_check import build_scene_checklist_section

        previous = os.environ.get("CHECKLIST_AUTO_MODE")
        os.environ["CHECKLIST_AUTO_MODE"] = "subject"
        try:
            section = build_scene_checklist_section("AUTO")
        finally:
            if previous is None:
                os.environ.pop("CHECKLIST_AUTO_MODE", None)
            else:
                os.environ["CHECKLIST_AUTO_MODE"] = previous
        self.assertIn("主体[", section)
        self.assertIn("主体不存在直接跳过", section)

    def test_cleaned_clauses_drop_explanatory_notes(self) -> None:
        """条文说明不是规范正文，拿它生成规则会产出假规则。"""
        import json
        from pathlib import Path as _Path

        report = _Path(__file__).parent / "resources" / "standards" / "clean_report.json"
        if not report.exists():
            self.skipTest("还没有清洗结果")
        data = json.loads(report.read_text(encoding="utf-8"))
        self.assertLess(data["dropped_explanatory"], data["total"] * 0.1)
        for item in data["dropped_detail"]:
            self.assertTrue(item["reason"])

    def test_scene_subject_index_covers_every_scene(self) -> None:
        from catalogs.scene_checklists import SCENE_SUBJECTS

        for scene, subjects in SCENE_SUBJECTS.items():
            self.assertTrue(subjects, f"{scene} 没有绑定主体")
        self.assertGreaterEqual(len(SCENE_SUBJECTS["SCAFFOLD_CANTILEVER"]), 30)

    def test_main_system_rules_extend_the_catalogue(self) -> None:
        """主系统规则包带进来的主体与条款必须真正落到目录里。"""
        import json
        from pathlib import Path as _Path

        from verified_regulations import VERIFIED_REGULATIONS

        source = _Path(__file__).parent / "resources" / "main_system_rules.json"
        if not source.exists():
            self.skipTest("还没有导入主系统规则")
        payload = json.loads(source.read_text(encoding="utf-8"))
        self.assertGreater(payload["subject_count"], 300)
        from catalogs.scene_checklists import SUBJECT_CHECKLIST

        names = {group["subject"] for group in SUBJECT_CHECKLIST}
        for subject in payload["subjects"][:40]:
            self.assertIn(subject["subject"], names)
        # 主系统规则里的条款应当能在已核验目录里找到
        missing = 0
        for rule in payload["rules"][:120]:
            if rule["regulation_id"] not in VERIFIED_REGULATIONS:
                missing += 1
        self.assertEqual(missing, 0)

    def test_generated_rules_are_opt_in(self) -> None:
        """生成规则默认不进提示词，生产路径不能被动。"""
        import os

        from safety_check import build_scene_checklist_section

        previous = os.environ.pop("USE_GENERATED_RULES", None)
        try:
            section = build_scene_checklist_section("SCAFFOLD_CANTILEVER")
        finally:
            if previous is not None:
                os.environ["USE_GENERATED_RULES"] = previous
        self.assertNotIn("补充检查项", section)

    def test_risk_item_cap_is_configurable_and_bounded(self) -> None:
        import os

        from safety_check import build_single_vision_system_prompt, max_risk_items

        self.assertEqual(max_risk_items(), 3)
        previous = os.environ.get("MAX_RISK_ITEMS")
        os.environ["MAX_RISK_ITEMS"] = "5"
        try:
            self.assertEqual(max_risk_items(), 5)
            prompt = build_single_vision_system_prompt("高处作业", compact=True)
        finally:
            if previous is None:
                os.environ.pop("MAX_RISK_ITEMS", None)
            else:
                os.environ["MAX_RISK_ITEMS"] = previous
        self.assertIn("最多输出5条", prompt)
        self.assertNotIn("{MAX_RISK_ITEMS}", prompt)

    def test_scene_token_auto_is_not_the_general_site(self) -> None:
        from scene_rules import normalize_scene

        self.assertEqual(normalize_scene("auto"), "AUTO")
        self.assertEqual(normalize_scene("AUTO"), "AUTO")
        self.assertEqual(normalize_scene(None), "AUTO")

    def test_generic_scaffold_request_uses_the_family_checklist(self) -> None:
        from scene_rules import normalize_scene
        from safety_check import build_scene_checklist_section

        self.assertEqual(normalize_scene("脚手架"), "SCAFFOLD_FAMILY")
        section = build_scene_checklist_section("SCAFFOLD_FAMILY")
        self.assertIn("悬挑式", section)
        self.assertIn("盘扣式", section)
        # 具体型式仍然走各自的精确清单
        self.assertEqual(normalize_scene("扣件式钢管脚手架"), "SCAFFOLD_COUPLER_TYPE")
        self.assertEqual(normalize_scene("悬挑式脚手架"), "SCAFFOLD_CANTILEVER")


if __name__ == "__main__":
    unittest.main()


class CompactOutputTests(unittest.TestCase):
    """Compact mode trades long key names for short ones; expansion must be lossless."""

    def test_compact_prompt_keeps_semantic_rules(self) -> None:
        compact = build_single_vision_system_prompt("悬挑式脚手架", compact=True)
        full = build_single_vision_system_prompt("悬挑式脚手架", compact=False)
        self.assertIn("箱门关闭但未上锁", compact)
        self.assertIn("至少2条相互独立的直接视觉线索", compact)
        self.assertIn("法规匹配范围不得局限于临时用电", compact)
        self.assertIn('"iss"', compact)
        self.assertNotIn('"claim_validation"', compact)
        head = compact[: compact.index("只输出一个合法JSON对象")]
        self.assertEqual(head, full[: full.index("只输出一个合法JSON对象")])

    def test_compact_expansion_restores_canonical_keys(self) -> None:
        raw = {
            "st": "SCAFFOLD_CANTILEVER", "sc": "悬挑式脚手架", "sconf": 0.8, "sev": ["钢梁"],
            "objs": [{"id": "O1", "t": "架体", "b": [1, 2, 3, 4], "vis": "partial", "s": "底部有空隙"}],
            "ppl": [{"id": "P1", "b": [5, 6, 7, 8],
                     "bv": {"head": "complete", "lh": "partial", "rh": "not_visible", "tr": "complete",
                            "wa": "partial", "lf": "complete", "rf": "complete"},
                     "hp": {"lg": "not_worn", "rg": "not_applicable"},
                     "pp": {"helmet": "worn", "gloves": "not_applicable", "vest": "not_applicable",
                            "harness": "not_applicable"},
                     "ac": "走动", "sup": "楼板", "el": "GROUND_LEVEL"}],
            "iss": [{"s": "CLEAR", "i": "未戴安全帽", "cat": "SCAFFOLD_CANTILEVER", "key": "HELMET_NOT_WORN",
                     "at": "PPE", "ct": "PPE", "oid": ["O1"], "pid": ["P1"], "ppe": "helmet", "bp": ["head"],
                     "sp": {"sa": "头部", "ro": "架体", "bv": True, "el": "GROUND_LEVEL", "rs": "STATIC",
                            "pz": "OUTSIDE", "cu": ["头顶裸露", "无帽壳"]},
                     "cv": {"id": True, "bd": True, "rg": True, "cu": ["头顶裸露", "无帽壳"],
                            "alt": "远处人员", "ax": True, "sa": "头部", "ra": "架体", "rel": "ADJACENT"},
                     "t": "人员头部", "pos": "左侧", "e": "人员头部裸露", "vc": ["头顶裸露", "无帽壳"],
                     "reg": {"c": "JGJ184-2009", "a": "2.0.4"}, "cf": 0.7, "tv": "complete",
                     "ev": "direct", "cc": "medium", "b": [5, 6, 7, 8]}],
        }
        out = expand_compact_result(raw)
        self.assertEqual(out["scene_type"], "SCAFFOLD_CANTILEVER")
        self.assertEqual(out["objects"][0]["object_id"], "O1")
        self.assertEqual(out["objects"][0]["bbox_2d_1000"], [1, 2, 3, 4])
        self.assertEqual(out["persons"][0]["person_id"], "P1")
        self.assertEqual(out["persons"][0]["body_visibility"]["left_hand"], "partial")
        self.assertEqual(out["persons"][0]["hand_ppe_states"]["left_glove"], "not_worn")
        self.assertEqual(out["persons"][0]["ppe_states"]["reflective_vest"], "not_applicable")
        issue = out["issues"][0]
        self.assertEqual(issue["status"], "CLEAR")
        self.assertEqual(issue["item"], "未戴安全帽")
        self.assertEqual(issue["object_ids"], ["O1"])
        self.assertEqual(issue["spatial_relation"]["subject_anchor"], "头部")
        self.assertEqual(issue["spatial_relation"]["cues"], ["头顶裸露", "无帽壳"])
        self.assertEqual(issue["claim_validation"]["target_identity_confirmed"], True)
        self.assertEqual(issue["claim_validation"]["positive_cues"], ["头顶裸露", "无帽壳"])
        self.assertEqual(issue["regulation"]["standard_code"], "JGJ184-2009")
        self.assertEqual(issue["bbox_2d_1000"], [5, 6, 7, 8])

    def test_compact_expansion_accepts_long_keys(self) -> None:
        raw = {"issues": [{"status": "SUSPECTED", "item": "电缆拖地", "bbox_2d_1000": [1, 2, 3, 4],
                           "regulation": {"standard_code": "JGJ/T 46-2024", "article": "6.2.3"}}]}
        out = expand_compact_result(raw)
        self.assertEqual(out["issues"][0]["status"], "SUSPECTED")
        self.assertEqual(out["issues"][0]["regulation"]["standard_code"], "JGJ/T 46-2024")

    def test_catalog_backfills_standard_name_and_clause_text(self) -> None:
        candidate = {"standard_code": "JGJ184-2009", "article": "2.0.4", "standard_name": "",
                     "clause_summary": "", "match_status": "NEEDS_VERIFICATION"}
        _backfill_regulation_from_catalog(candidate)
        self.assertEqual(candidate["standard_name"], "建筑施工作业劳动防护用品配备及使用标准")
        self.assertIn("安全帽", candidate["clause_summary"])
        self.assertEqual(candidate["match_status"], "MATCHED")


class JsonRepairTests(unittest.TestCase):
    """The model occasionally emits malformed JSON; the pipeline must recover."""

    def test_plain_json_is_untouched(self) -> None:
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_trailing_comma_is_repaired(self) -> None:
        self.assertEqual(extract_json_object('{"a": 1,}'), {"a": 1})

    def test_inner_unescaped_quote_is_escaped(self) -> None:
        broken = '{"iss":[{"s":"CLEAR","i":"架体与"楼体"间空隙","b":[1,2,3,4]}]}'
        out = extract_json_object(broken)
        self.assertEqual(len(out["iss"]), 1)
        self.assertEqual(out["iss"][0]["s"], "CLEAR")
        self.assertIn("楼体", out["iss"][0]["i"])

    def test_truncated_document_is_salvaged(self) -> None:
        truncated = '{"st":"X","iss":[{"s":"CLEAR","i":"被截断'
        out = extract_json_object(truncated)
        self.assertEqual(out["st"], "X")
        self.assertEqual(len(out["iss"]), 1)

    def test_salvaged_issues_keep_sibling_objects(self) -> None:
        damaged = (
            '{"iss":[{"s":"CLEAR","i":"第一条","b":[1,2,3,4]},'
            '{"s":"SUSPECTED","i":"第二条带\"引号\"","b":[5,6,7,8]}]}'
        )
        out = extract_json_object(damaged)
        self.assertEqual(len(out["iss"]), 2)
        self.assertEqual(out["iss"][0]["i"], "第一条")


class ModelRetryTests(unittest.TestCase):
    """Transient failures may be retried, but never past the time budget."""

    def _client(self, side_effect):
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.side_effect = side_effect
        return client

    def _ok_response(self):
        response = SimpleNamespace()
        response.raise_for_status = Mock()
        response.json = Mock(return_value={
            "choices": [{"message": {"content": '{"status":"CLEAR"}'}, "finish_reason": "stop"}]
        })
        return response

    def test_dns_failure_is_retried_within_budget(self) -> None:
        calls = {"n": 0}

        async def post(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("[Errno 8] nodename nor servname provided")
            return self._ok_response()

        client = self._client(post)
        with patch("safety_check.httpx.AsyncClient", return_value=client):
            content, attempts = call_deepseek({}, "key", "https://example.invalid", timeout_seconds=5.0)
        self.assertEqual(content, '{"status":"CLEAR"}')
        self.assertEqual(attempts, 2)
        self.assertEqual(client.post.await_count, 2)

    def test_budget_exhaustion_is_not_retried(self) -> None:
        async def slow(*args, **kwargs):
            await asyncio.sleep(1)
            return self._ok_response()

        client = self._client(slow)
        with patch("safety_check.httpx.AsyncClient", return_value=client), \
                self.assertRaises(httpx.TimeoutException):
            call_deepseek({}, "key", "https://example.invalid", timeout_seconds=0.05)
        self.assertEqual(client.post.await_count, 1)

    def test_permanent_client_error_is_not_retried(self) -> None:
        response = SimpleNamespace()
        response.status_code = 401
        response.raise_for_status = Mock(side_effect=httpx.HTTPStatusError(
            "unauthorized", request=Mock(), response=response))
        response.json = Mock(return_value={})

        async def post(*args, **kwargs):
            return response

        client = self._client(post)
        with patch("safety_check.httpx.AsyncClient", return_value=client), \
                self.assertRaises(httpx.HTTPStatusError):
            call_deepseek({}, "key", "https://example.invalid", timeout_seconds=5.0)
        self.assertEqual(client.post.await_count, 1)

    def test_server_error_is_retried(self) -> None:
        calls = {"n": 0}

        async def post(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                response = SimpleNamespace()
                response.status_code = 503
                response.raise_for_status = Mock(side_effect=httpx.HTTPStatusError(
                    "unavailable", request=Mock(), response=response))
                response.json = Mock(return_value={})
                return response
            return self._ok_response()

        client = self._client(post)
        with patch("safety_check.httpx.AsyncClient", return_value=client):
            _, attempts = call_deepseek({}, "key", "https://example.invalid", timeout_seconds=5.0)
        self.assertEqual(attempts, 2)

    def test_transient_classification(self) -> None:
        self.assertFalse(is_transient_model_error(ModelBudgetExhausted("x")))
        self.assertTrue(is_transient_model_error(httpx.ConnectError("dns")))
        self.assertTrue(is_transient_model_error(httpx.ReadTimeout("read")))
        self.assertFalse(is_transient_model_error(ValueError("bad body")))
        response = SimpleNamespace(status_code=400)
        error = httpx.HTTPStatusError("bad", request=Mock(), response=response)
        self.assertFalse(is_transient_model_error(error))


class ModelQuotaTests(unittest.TestCase):
    """A billing rejection must be reported as such, never as a detection failure."""

    def _client(self, status: int, body: str):
        client = AsyncMock()
        client.__aenter__.return_value = client
        response = SimpleNamespace(status_code=status, text=body)

        def raise_for_status():
            if status >= 400:
                raise httpx.HTTPStatusError(
                    f"HTTP {status}", request=Mock(), response=response)

        response.raise_for_status = raise_for_status
        response.json = Mock(return_value={})

        async def post(*args, **kwargs):
            return response

        client.post.side_effect = post
        return client

    def test_payment_required_is_reported_as_quota_error(self) -> None:
        client = self._client(402, '{"error":{"message":"Insufficient Balance"}}')
        with patch("safety_check.httpx.AsyncClient", return_value=client), \
                self.assertRaises(ModelQuotaError):
            call_deepseek({}, "key", "https://example.invalid", timeout_seconds=5.0)
        self.assertEqual(client.post.await_count, 1)

    def test_rate_limit_is_reported_as_quota_error(self) -> None:
        client = self._client(429, '{"error":{"message":"rate limited"}}')
        with patch("safety_check.httpx.AsyncClient", return_value=client), \
                self.assertRaises(ModelQuotaError):
            call_deepseek({}, "key", "https://example.invalid", timeout_seconds=5.0)
        self.assertEqual(client.post.await_count, 1)

    def test_other_client_errors_keep_their_type(self) -> None:
        client = self._client(401, '{"error":{"message":"unauthorized"}}')
        with patch("safety_check.httpx.AsyncClient", return_value=client), \
                self.assertRaises(httpx.HTTPStatusError):
            call_deepseek({}, "key", "https://example.invalid", timeout_seconds=5.0)
        self.assertEqual(client.post.await_count, 1)
