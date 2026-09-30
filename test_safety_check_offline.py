import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from safety_check import (
    apply_static_image_scope_gate,
    bbox_1000_to_original,
    build_single_vision_system_prompt,
    build_vision_payload,
    emit_cli_error,
    model_judged_report,
    normalize_fact_result,
    normalize_bbox_1000,
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

    def test_single_model_prompt_requires_full_image_normalized_boxes(self) -> None:
        prompt = build_single_vision_system_prompt(None)
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

    def test_model_detected_scene_is_not_overwritten_by_scene_hint(self) -> None:
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
        self.assertEqual(result["scene_type"], "CONSTRUCTION_HOIST")
        self.assertEqual(result["scene"], "施工升降机")
        self.assertEqual(result["scene_hint"], "施工现场临时用电")

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


if __name__ == "__main__":
    unittest.main()
