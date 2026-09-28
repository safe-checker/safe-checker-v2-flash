"""Scene catalog and prompt metadata for construction safety.

The catalog is not the final safety judge. ``fields`` and the scene
description give the vision models a compact checklist and vocabulary.
``rules`` is retained as compatibility/documentation metadata for existing
reports; it is not executed to create, rewrite, or suppress issues. Design
calculations, functional tests, permits, qualifications, measurements without
a scale, and hidden/internal conditions remain uncertain.
"""

from __future__ import annotations

from typing import Any


YES_NO_STATES = "yes/no/uncertain/not_applicable"


def _rule(
    field: str,
    value: str,
    item: str,
    evidence: str,
    standard: str,
    status: str = "疑似违规",
) -> dict[str, str]:
    return {
        "field": field,
        "value": value,
        "item": item,
        "evidence": evidence,
        "standard": standard,
        "status": status,
    }


COMMON_PREAMBLE = (
    "只依据图片中清晰可见的静态证据；看不见、被遮挡、过小、反光或需要测量/试验的内容写 "
    '"uncertain"，不把“未入镜”写成“缺失”。字段值只能为 yes、no、uncertain、not_applicable。'
)


SCENE_PROFILES: dict[str, dict[str, Any]] = {
    "GENERAL_SITE": {
        "label": "施工现场综合安全",
        "aliases": (
            "综合安全",
            "施工现场",
            "工地安全",
            "复杂环境",
            "general",
            "general_site",
            "auto",
            "自动",
        ),
        "standards": "《建筑施工安全检查标准》（JGJ 59-2011）及《建筑施工安全技术统一规范》（GB 50870）",
        "description": "施工现场通用的人员防护、通道、材料堆放、消防和警示隔离。",
        "object_fields": (
            "work_area_visible",
            "person_visible",
            "passage_visible",
            "fire_facility_visible",
        ),
        "fields": {
            "work_area_visible": "施工区域或作业面是否清晰可见",
            "person_visible": "人员是否清晰可见",
            "helmet_worn": "可见人员是否佩戴安全帽",
            "passage_visible": "人员通道或作业通道是否可见",
            "passage_obstructed": "通道是否被材料、杂物或设备堵塞",
            "housekeeping_disorder": "材料、工具或杂物是否明显杂乱散落",
            "warning_isolation_visible": "危险区域是否设置明显警示或隔离",
            "fire_facility_visible": "灭火器、消防栓或消防设施是否可见",
            "fire_facility_accessible": "消防设施前是否保持可操作空间",
            "flammable_material_near_heat": "可燃物是否明显靠近火源或高温作业点",
        },
        "rules": (
            _rule(
                "helmet_worn",
                "no",
                "人员未佩戴安全帽",
                "人员头部清晰可见但未见安全帽；远景或头部被遮挡不能判定。",
                "可能涉及 JGJ 59-2011 关于个人防护用品的要求。",
            ),
            _rule(
                "passage_obstructed",
                "yes",
                "施工通道受阻",
                "可见通道被材料、设备或杂物明显占用，影响通行。",
                "可能涉及 JGJ 59-2011 关于施工现场通道和文明施工的要求。",
            ),
            _rule(
                "housekeeping_disorder",
                "yes",
                "材料或杂物堆放杂乱",
                "作业面存在明显散落材料、工具或杂物，存在绊倒或管理风险。",
                "可能涉及 JGJ 59-2011 关于现场材料堆放和文明施工的要求。",
            ),
            _rule(
                "warning_isolation_visible",
                "no",
                "危险区域未见明显警示或隔离",
                "危险作业区域边界清晰可见，但未见相应警示、围挡或隔离措施。",
                "可能涉及 JGJ 59-2011 关于警示、隔离和现场防护的要求。",
            ),
            _rule(
                "fire_facility_accessible",
                "no",
                "消防设施前操作空间受阻",
                "消防设施可见，但前方被材料、车辆或杂物明显遮挡。",
                "可能涉及 JGJ 59-2011 关于消防设施和疏散通道的要求。",
            ),
            _rule(
                "flammable_material_near_heat",
                "yes",
                "可燃物靠近火源或高温作业点",
                "可燃材料与明火、焊接或高温设备的近距离关系在图中清晰可见。",
                "可能涉及施工现场消防安全及 JGJ 59-2011 相关要求。",
            ),
        ),
        "review_pairs": (),
    },
    "TEMPORARY_ELECTRICITY": {
        "label": "临时用电/配电箱",
        "aliases": (
            "临时用电",
            "配电箱",
            "开关箱",
            "施工现场临时用电",
            "temporary_electricity",
            "electricity",
        ),
        "standards": "《建筑与市政工程施工现场临时用电安全技术标准》（JGJ/T 46-2024）",
        "description": "配电箱、开关箱、电缆敷设、插座接口、接地和潮湿环境。",
        "object_fields": (
            "distribution_box_visible",
            "electrical_cable_visible",
            "ground_visible",
        ),
        "fields": {
            "distribution_box_visible": "配电箱或开关箱是否可见",
            "electrical_cable_visible": "电气电缆或电源线是否可见",
            "ground_visible": "地面或设备周边地面是否可见",
            "box_door_visible": "配电箱主箱门或门板是否清晰可见",
            "socket_cover_visible": "插座盖、插头或接口区域是否清晰可见",
            "box_door_open": "配电箱箱门是否明显打开",
            "socket_or_interface_exposed": "插座口、插头或接线接口是否外露",
            "cable_ground_contact": "电缆是否与地面连续接触或沿地面敷设",
            "cable_tangled_or_disordered": "电缆是否明显缠绕或布设杂乱",
            "water_visible": "地面是否存在明确水面、水坑或液体聚集",
            "workspace_blocked": "配电箱周边操作空间是否受阻",
            "box_fixed_stably": "配电箱是否能确认稳固固定",
        },
        "rules": (
            _rule(
                "cable_ground_contact",
                "yes",
                "电缆沿地面敷设或拖地",
                "图片事实显示电气电缆与地面连续接触或沿地面敷设。",
                "对应条款由已核验法规目录根据风险类型匹配。",
                "明确违规",
            ),
            _rule(
                "cable_tangled_or_disordered",
                "yes",
                "电缆缠绕或布设杂乱",
                "图片可见电气电缆盘绕、缠绕或布设杂乱。",
                "对应条款由已核验法规目录根据风险类型匹配。",
            ),
            _rule(
                "workspace_blocked",
                "yes",
                "配电箱周边操作空间或通道受阻",
                "配电箱周边空间被电缆、构件或杂物占用。",
                "对应条款由已核验法规目录根据风险类型匹配。",
            ),
            _rule(
                "socket_or_interface_exposed",
                "yes",
                "配电箱插座口防护盖开启或接口外露",
                "视觉模型判断插座盖、插头、接口或接线区域处于开启/外露状态。",
                "对应条款由已核验法规目录根据风险类型匹配。",
            ),
            _rule(
                "box_door_open",
                "yes",
                "配电箱主箱门打开",
                "视觉模型判断主箱门门板、门锁、铰链、开口或内部元件显示箱门处于打开状态。",
                "对应条款由已核验法规目录根据风险类型匹配。",
            ),
            _rule(
                "water_visible",
                "yes",
                "地面疑似积水或电气设备处于液体环境",
                "可见明确水面、水坑边界或液体聚集，不把普通光影和反光作为积水。",
                "对应条款由已核验法规目录根据风险类型匹配。",
            ),
            _rule(
                "box_fixed_stably",
                "no",
                "配电箱安装固定不稳",
                "图片可见箱体倾倒、明显临时搁置或固定构件失效。",
                "对应条款由已核验法规目录根据风险类型匹配。",
            ),
        ),
        "review_pairs": (
            ("box_door_open", "socket_or_interface_exposed"),
        ),
    },
    "WORK_AT_HEIGHT": {
        "label": "高处作业",
        "aliases": ("高处作业", "临边防护", "洞口防护", "登高作业", "work_at_height", "highwork"),
        "standards": "《建筑施工高处作业安全技术规范》（JGJ 80-2016）、《安全带》（GB 6095）",
        "description": "临边、洞口、平台、梯子、安全带、防坠和坠物防护。",
        "object_fields": ("high_place_visible", "person_visible", "edge_or_opening_visible"),
        "fields": {
            "high_place_visible": "高处作业面、临边、洞口、平台或梯子是否可见",
            "person_visible": "高处作业人员是否可见",
            "edge_or_opening_visible": "临边或洞口及其边界是否可见",
            "edge_protection_complete": "临边栏杆、围挡或防护网是否完整",
            "opening_protection_complete": "洞口盖板、围挡或防护网是否完整",
            "safety_belt_worn": "人员是否佩戴安全带",
            "safety_belt_anchored": "安全带挂设路径和挂点是否清晰有效",
            "ladder_safe": "梯子支撑、角度和使用状态是否安全",
            "falling_object_control": "工具、材料和坠物风险是否得到可见控制",
            "access_route_safe": "上下通道、爬梯或作业平台是否连续安全",
        },
        "rules": (
            _rule(
                "edge_protection_complete",
                "no",
                "高处临边防护不完整",
                "临边边界清晰可见，但栏杆、挡脚板、围挡或安全网存在明显缺失或破损。",
                "可能涉及 JGJ 80-2016 关于临边防护的要求。",
            ),
            _rule(
                "opening_protection_complete",
                "no",
                "洞口防护不完整",
                "洞口边界清晰可见，但盖板、围挡或防护网明显缺失、移位或破损。",
                "可能涉及 JGJ 80-2016 关于洞口防护的要求。",
            ),
            _rule(
                "safety_belt_worn",
                "no",
                "高处作业人员未见佩戴安全带",
                "人员躯干和腰部清晰可见，但未见安全带；遮挡或远景只能判为不可判断。",
                "可能涉及 JGJ 80-2016、GB 6095 关于高处作业个体防护的要求。",
            ),
            _rule(
                "safety_belt_anchored",
                "no",
                "安全带挂设明显不可靠",
                "安全带和挂点路径清晰可见，但挂点脱开、挂设错误或明显低挂高用。",
                "可能涉及 JGJ 80-2016、GB 6095 关于安全带挂设的要求。",
            ),
            _rule(
                "ladder_safe",
                "no",
                "梯子使用状态存在明显风险",
                "梯子支撑不稳、踏步损坏、明显倾斜或人员存在跨越/站在不安全位置的直接证据。",
                "可能涉及 JGJ 80-2016 关于登高设施和梯子使用的要求。",
            ),
            _rule(
                "falling_object_control",
                "no",
                "高处坠物防护不足",
                "可见工具、材料或构件处于可能坠落位置，且未见系挂、挡护或隔离措施。",
                "可能涉及 JGJ 80-2016 关于防护棚、工具系挂和坠物防护的要求。",
            ),
            _rule(
                "access_route_safe",
                "no",
                "高处作业上下通道或作业平台不安全",
                "通道、爬梯或平台清晰可见，但存在明显断开、缺板、无防护或严重堵塞。",
                "可能涉及 JGJ 80-2016 关于通道和作业平台的要求。",
            ),
        ),
        "review_pairs": (
            ("safety_belt_worn", "safety_belt_anchored"),
        ),
    },
    "FOUNDATION_PIT": {
        "label": "基坑工程",
        "aliases": ("基坑", "基坑工程", "土方开挖", "foundation_pit", "excavation"),
        "standards": "《建筑基坑支护技术规程》（JGJ 120）、《建筑基坑工程监测技术标准》（GB 50497）",
        "description": "基坑临边、支护、上下通道、坑边堆载、排水和警戒隔离。",
        "object_fields": ("foundation_pit_visible", "pit_edge_visible", "support_structure_visible"),
        "fields": {
            "foundation_pit_visible": "基坑、沟槽或深开挖区域是否可见",
            "pit_edge_visible": "坑边、坡顶或临边范围是否可见",
            "support_structure_visible": "支护、边坡或支撑结构是否可见",
            "edge_protection_complete": "坑边临边防护是否完整",
            "access_route_safe": "上下基坑通道、梯道或坡道是否安全",
            "slope_or_support_distress": "边坡、支护或支撑是否有明显裂缝、变形、脱落或破坏",
            "edge_load_or_material_piled": "坑边是否明显堆载或停放机械",
            "drainage_or_water_problem": "坑内是否存在明确积水、渗漏或排水失效线索",
            "warning_isolation_visible": "坑边是否设置警示、围栏或隔离",
        },
        "rules": (
            _rule(
                "edge_protection_complete",
                "no",
                "基坑临边防护不完整",
                "坑边范围清晰可见，但栏杆、围挡、挡脚板或警戒隔离明显缺失。",
                "可能涉及 JGJ 120、JGJ 59 关于基坑临边防护的要求。",
            ),
            _rule(
                "access_route_safe",
                "no",
                "上下基坑通道存在明显风险",
                "可见梯道、坡道或通道断开、无防护、严重湿滑或被堵塞。",
                "可能涉及 JGJ 120 关于基坑上下通道和作业安全的要求。",
            ),
            _rule(
                "slope_or_support_distress",
                "yes",
                "基坑边坡或支护出现明显异常",
                "可见明确裂缝、明显变形、支撑脱落、支护破坏或持续渗漏等表观线索。",
                "可能涉及 JGJ 120、GB 50497 关于基坑支护和监测的要求。",
            ),
            _rule(
                "edge_load_or_material_piled",
                "yes",
                "坑边存在明显堆载或机械靠近",
                "材料、土方或机械与坑边的近距离关系清晰可见。",
                "可能涉及 JGJ 120 关于坑边荷载和安全距离的要求，距离需现场复核。",
            ),
            _rule(
                "drainage_or_water_problem",
                "yes",
                "基坑内存在积水、渗漏或排水异常线索",
                "可见明确水面、持续渗漏、排水沟堵塞或集水设施失效线索。",
                "可能涉及 JGJ 120、GB 50497 关于降排水和基坑环境的要求。",
            ),
            _rule(
                "warning_isolation_visible",
                "no",
                "基坑边缘未见明显警示或隔离",
                "坑边边界清晰可见，但未见围栏、警示牌或有效隔离。",
                "可能涉及 JGJ 59 关于基坑临边警示和隔离的要求。",
            ),
        ),
        "review_pairs": (),
    },
    "LIFTING_OPERATIONS": {
        "label": "起重吊装",
        "aliases": ("吊装", "吊装作业", "起重吊装", "起重作业", "lifting", "lifting_operations"),
        "standards": "《建筑施工起重吊装工程安全技术规范》（JGJ 276）、相关起重机械安全技术标准",
        "description": "吊物、吊钩、吊索具、吊点、支腿垫板、警戒区和人员站位。",
        "object_fields": ("lifting_operation_visible", "suspended_load_visible", "lifting_equipment_visible"),
        "fields": {
            "lifting_operation_visible": "起重吊装作业或吊装设备是否可见",
            "suspended_load_visible": "悬吊物或被吊构件是否可见",
            "lifting_equipment_visible": "起重机、吊车或起重设备是否可见",
            "person_under_load_or_in_swing_zone": "人员是否处于吊物下方或回转危险区",
            "sling_or_shackle_condition": "吊带、钢丝绳、卸扣或卡环是否存在明显缺陷",
            "hook_latch_visible": "吊钩防脱装置是否清晰可见且状态正常",
            "load_binding_or_lifting_point_clear": "吊物捆绑、吊点和受力路径是否清晰可靠",
            "outrigger_or_support_pads_visible": "支腿展开、垫板和支撑地面是否可见且稳定",
            "exclusion_zone_visible": "吊装危险区域是否设置警戒或隔离",
            "guide_rope_or_control_visible": "需要控制摆动的吊物是否可见溜绳或有效控制",
            "oblique_pull_or_unbalanced_load": "是否存在明显斜拉、偏吊或吊物严重失衡",
        },
        "rules": (
            _rule(
                "person_under_load_or_in_swing_zone",
                "yes",
                "人员进入吊物下方或回转危险区",
                "同一画面中人员与悬吊物下方、吊物运行路径或回转区域的空间关系清晰可见。",
                "可能涉及 JGJ 276 关于吊装作业区域人员隔离和禁止停留的要求。",
            ),
            _rule(
                "sling_or_shackle_condition",
                "no",
                "吊索具存在明显缺陷",
                "钢丝绳、吊带、卸扣或卡环出现清晰可见的断裂、严重磨损、打结、变形或脱落。",
                "可能涉及 JGJ 276 关于吊索具和起重吊具检查使用的要求。",
            ),
            _rule(
                "hook_latch_visible",
                "no",
                "吊钩防脱装置存在明显异常",
                "吊钩及防脱装置清晰可见，且可见防脱片缺失、失效或明显变形。",
                "可能涉及 JGJ 276 关于吊钩和防脱装置的要求。",
            ),
            _rule(
                "load_binding_or_lifting_point_clear",
                "no",
                "吊物捆绑或吊点存在明显风险",
                "吊点偏移、捆绑松脱、受力路径错误或构件明显滑移等直接证据清晰可见。",
                "可能涉及 JGJ 276 关于吊点、捆绑和吊装稳定性的要求。",
            ),
            _rule(
                "outrigger_or_support_pads_visible",
                "no",
                "起重设备支腿或垫板支撑存在明显问题",
                "支腿未展开、垫板明显缺失、支撑悬空或地面沉陷等状态清晰可见。",
                "可能涉及 JGJ 276 关于起重设备站位、支腿和地基支撑的要求。",
            ),
            _rule(
                "exclusion_zone_visible",
                "no",
                "吊装危险区域未见有效警戒或隔离",
                "吊装作业范围清晰可见，但人员通行区域与吊装危险区域未设置明显隔离。",
                "可能涉及 JGJ 276 关于吊装警戒区域和现场指挥的要求。",
            ),
            _rule(
                "guide_rope_or_control_visible",
                "no",
                "是否需要溜绳或摆动控制措施",
                "未见可见溜绳或其他控制措施，但单张静态图片通常不能确认吊物是否存在摆动控制需求。",
                "可能涉及 JGJ 276 关于吊物稳定和溜绳控制的要求，需结合吊物类型、作业过程或现场复核。",
                "不可判断",
            ),
            _rule(
                "oblique_pull_or_unbalanced_load",
                "yes",
                "存在斜拉、偏吊或吊物明显失衡线索",
                "吊索受力方向、吊物姿态或起重设备与吊物关系清晰显示明显斜拉、偏吊或失衡。",
                "可能涉及 JGJ 276 关于禁止斜拉斜吊和吊物稳定的要求。",
            ),
        ),
        "review_pairs": (
            ("hook_latch_visible", "sling_or_shackle_condition"),
            ("person_under_load_or_in_swing_zone", "exclusion_zone_visible"),
        ),
    },
    "TOWER_CRANE": {
        "label": "塔式起重机",
        "aliases": ("塔吊", "塔式起重机", "塔机", "tower_crane"),
        "standards": "《塔式起重机安全规程》（GB 5144）、《塔式起重机》（GB/T 5031）、JGJ 196",
        "description": "塔机吊钩钢丝绳、标准节、附着、通道、基础和吊装区域。",
        "object_fields": ("tower_crane_visible", "lifting_equipment_visible", "suspended_load_visible"),
        "fields": {
            "tower_crane_visible": "塔式起重机主体是否可见",
            "lifting_equipment_visible": "起重设备及其工作部件是否可见",
            "suspended_load_visible": "悬吊物或吊钩作业状态是否可见",
            "hook_latch_visible": "吊钩防脱装置是否清晰可见且状态正常",
            "wire_rope_condition": "钢丝绳是否存在明显断丝、扭结、严重磨损或脱槽",
            "tower_connection_condition": "标准节、螺栓、销轴或附着节点是否存在明显异常",
            "platform_guard_complete": "塔机平台、走道、爬梯和护栏是否完整",
            "exclusion_zone_visible": "吊物下方或回转区域是否设置警戒隔离",
            "power_line_clearance_visible": "塔机与外电线路的空间关系是否清晰且存在明显过近风险",
            "safety_device_function": "限位、力矩、制动和报警等功能是否有效",
        },
        "rules": (
            _rule(
                "wire_rope_condition",
                "no",
                "塔机钢丝绳存在明显缺陷",
                "钢丝绳清晰可见断丝、严重磨损、扭结、压扁、脱槽或异常缠绕。",
                "可能涉及 GB 5144、JGJ 196 关于钢丝绳和起升机构的要求。",
            ),
            _rule(
                "hook_latch_visible",
                "no",
                "塔机吊钩防脱装置存在明显异常",
                "吊钩特写清晰可见防脱装置缺失、变形或未闭合。",
                "可能涉及 GB 5144 关于吊钩防脱装置的要求。",
            ),
            _rule(
                "tower_connection_condition",
                "no",
                "塔机标准节或附着连接存在明显异常",
                "可见螺栓、销轴、附着杆或连接节点缺失、脱落、明显变形或开裂。",
                "可能涉及 GB 5144、JGJ 196 关于塔身和附着连接的要求。",
            ),
            _rule(
                "platform_guard_complete",
                "no",
                "塔机平台、爬梯或护栏防护不完整",
                "平台、通道或爬梯清晰可见，但存在明显缺口、护栏缺失或通行防护破损。",
                "可能涉及 GB 5144、JGJ 196 关于塔机通道和防护的要求。",
            ),
            _rule(
                "exclusion_zone_visible",
                "no",
                "塔机吊装危险区域未见有效隔离",
                "吊物下方或回转危险区域清晰可见，但未见有效警戒和隔离。",
                "可能涉及 GB 5144、JGJ 196 关于作业区域和人员安全的要求。",
            ),
            _rule(
                "power_line_clearance_visible",
                "yes",
                "塔机与外电线路存在明显近距离风险",
                "塔机、吊臂或吊物与外电线路的空间关系清晰可见且明显过近。",
                "可能涉及 GB 5144、JGJ 196 及施工现场外电线路防护要求，距离需现场复核。",
            ),
        ),
        "review_pairs": (
            ("hook_latch_visible", "wire_rope_condition"),
        ),
    },
    "CONSTRUCTION_HOIST": {
        "label": "施工升降机",
        "aliases": ("施工升降机", "施工电梯", "人货梯", "construction_hoist"),
        "standards": "《施工升降机安全规程》（GB 10055）、《施工升降机》（GB/T 34023）、JGJ 215",
        "description": "吊笼、层门、围栏、层站通道、导轨架、附墙和电缆导向。",
        "object_fields": ("construction_hoist_visible", "cage_or_landing_visible"),
        "fields": {
            "construction_hoist_visible": "施工升降机主体是否可见",
            "cage_or_landing_visible": "吊笼、层站或围栏是否可见",
            "landing_gate_guard_complete": "层门、底笼门和围栏防护是否完整",
            "cage_door_guard_complete": "吊笼门及其防护状态是否完整",
            "landing_access_safe": "层站通道、脚手板和护栏是否安全",
            "guide_rail_attachment_visible": "导轨架、标准节和附墙结构是否可见",
            "cable_protection_visible": "电缆导向和防护是否明显异常",
            "safety_device_function": "防坠器、门联锁、限位和制动功能是否有效",
            "load_or_person_risk": "是否存在明显超员、超载或人员处于危险位置",
        },
        "rules": (
            _rule(
                "landing_gate_guard_complete",
                "no",
                "施工升降机层门或围栏防护不完整",
                "层站边界清晰可见，但层门、围栏或防护栏存在明显缺失、破损或敞开无防护。",
                "可能涉及 GB 10055、JGJ 215 关于层门、围栏和层站防护的要求。",
            ),
            _rule(
                "cage_door_guard_complete",
                "no",
                "吊笼门防护存在明显问题",
                "吊笼门及开口清晰可见，但门体、门锁或防护存在明显缺失或异常。",
                "可能涉及 GB 10055、JGJ 215 关于吊笼门和防护的要求。",
            ),
            _rule(
                "landing_access_safe",
                "no",
                "施工升降机层站通道存在明显风险",
                "层站通道、脚手板或护栏清晰可见断开、缺失、严重堵塞或无防护。",
                "可能涉及 JGJ 215 关于层站通道和防护的要求。",
            ),
            _rule(
                "guide_rail_attachment_visible",
                "no",
                "导轨架或附墙结构存在明显异常",
                "标准节、导轨架或附墙节点清晰可见缺件、脱落、严重变形或连接异常。",
                "可能涉及 GB 10055、JGJ 215 关于导轨架和附墙的要求。",
            ),
            _rule(
                "cable_protection_visible",
                "no",
                "施工升降机电缆导向或防护异常",
                "电缆及导向装置清晰可见悬挂失控、拖地、破损或明显卡阻。",
                "可能涉及 JGJ 215 关于电缆导向和防护的要求。",
            ),
            _rule(
                "load_or_person_risk",
                "yes",
                "吊笼内人员或载荷存在明显风险",
                "画面直接显示人员处于明显危险位置或载荷明显超出吊笼边界；不凭外观估算额定载荷。",
                "可能涉及 GB 10055、JGJ 215 关于载荷、人员和运行区域的要求。",
            ),
        ),
        "review_pairs": (
            ("landing_gate_guard_complete", "cage_door_guard_complete"),
        ),
    },
}


_SCAFFOLD_COMMON_FIELDS = {
    "scaffold_visible": "脚手架架体或作业层是否可见",
    "base_stable": "架体基础、底座或支承是否明显稳定",
    "members_complete": "立杆、水平杆、节点或主要杆件是否完整",
    "ties_or_attachments_visible": "连墙件、附着或拉结是否可见且无明显异常",
    "bracing_complete": "剪刀撑、斜杆或稳定构造是否完整",
    "working_platform_complete": "脚手板、作业层和铺设状态是否完整",
    "guardrail_toeboard_complete": "外侧栏杆、挡脚板或安全网是否完整",
    "access_ladder_safe": "上下通道或爬梯是否安全",
    "material_disorder_or_overload": "作业层材料是否明显超载或杂乱堆放",
}


def _scaffold_profile(
    key: str,
    label: str,
    aliases: tuple[str, ...],
    standard: str,
    description: str,
    extra_fields: dict[str, str],
    extra_rules: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    rules = (
        _rule(
            "base_stable",
            "no",
            "脚手架基础或支承存在明显问题",
            "架体基础、底座或支承清晰可见沉陷、悬空、失稳或明显缺失。",
            f"可能涉及 {standard} 关于脚手架基础和支承的要求。",
        ),
        _rule(
            "members_complete",
            "no",
            "脚手架主要杆件或节点存在明显缺失",
            "立杆、水平杆、节点或主要构件清晰可见缺失、脱落或明显变形。",
            f"可能涉及 {standard} 关于架体构造和杆件连接的要求。",
        ),
        _rule(
            "ties_or_attachments_visible",
            "no",
            "脚手架连墙件或附着拉结存在明显问题",
            "连墙件、附着或拉结清晰可见缺失、脱落或明显失效。",
            f"可能涉及 {standard} 关于连墙件、附着和稳定的要求。",
        ),
        _rule(
            "bracing_complete",
            "no",
            "脚手架剪刀撑或稳定构造不完整",
            "剪刀撑、斜杆或稳定构造范围清晰可见缺失、断开或明显异常。",
            f"可能涉及 {standard} 关于架体整体稳定的要求。",
        ),
        _rule(
            "working_platform_complete",
            "no",
            "脚手板或作业层铺设不完整",
            "作业层清晰可见缺板、明显探头板、严重缝隙或板面破损。",
            f"可能涉及 {standard} 关于脚手板和作业层的要求。",
        ),
        _rule(
            "guardrail_toeboard_complete",
            "no",
            "脚手架外侧防护不完整",
            "外侧栏杆、挡脚板或安全网范围清晰可见缺失、断开或破损。",
            f"可能涉及 {standard} 关于架体防护的要求。",
        ),
        _rule(
            "access_ladder_safe",
            "no",
            "脚手架上下通道存在明显风险",
            "通道或爬梯清晰可见断开、无防护、损坏或严重堵塞。",
            f"可能涉及 {standard} 关于上下通道的要求。",
        ),
        _rule(
            "material_disorder_or_overload",
            "yes",
            "脚手架作业层材料堆放存在明显风险",
            "作业层可见材料明显集中堆放、外伸或影响人员通行。",
            f"可能涉及 {standard} 关于作业层荷载和材料堆放的要求，荷载数值需现场核验。",
        ),
        *extra_rules,
    )
    fields = dict(_SCAFFOLD_COMMON_FIELDS)
    fields.update(extra_fields)
    return {
        "label": label,
        "aliases": aliases,
        "standards": standard,
        "description": description,
        "object_fields": ("scaffold_visible",),
        "fields": fields,
        "rules": rules,
        "review_pairs": (),
    }


SCENE_PROFILES.update(
    {
        "SCAFFOLD_COUPLER_TYPE": _scaffold_profile(
            "SCAFFOLD_COUPLER_TYPE",
            "扣件式钢管脚手架",
            ("扣件式脚手架", "钢管脚手架", "脚手架", "scaffold", "scaffold_coupler_type"),
            "《建筑施工扣件式钢管脚手架安全技术规范》（JGJ 130）",
            "扣件式钢管脚手架的杆件、扣件、连墙件、剪刀撑、脚手板和防护。",
            {"coupler_condition": "扣件、连接节点是否明显松动、脱落或破损"},
            (
                _rule(
                    "coupler_condition",
                    "no",
                    "脚手架扣件或连接节点存在明显异常",
                    "扣件、连接节点清晰可见松动、脱落、缺失或明显破损。",
                    "可能涉及 JGJ 130 关于扣件连接和节点构造的要求。",
                ),
            ),
        ),
        "SCAFFOLD_DISC_BUCKLE": _scaffold_profile(
            "SCAFFOLD_DISC_BUCKLE",
            "承插型盘扣式钢管脚手架",
            ("盘扣脚手架", "承插型盘扣式脚手架", "scaffold_disc_buckle"),
            "《建筑施工承插型盘扣式钢管脚手架安全技术标准》（JGJ/T 231）",
            "盘扣节点、立杆套管、水平杆斜杆、底座顶托和架体防护。",
            {"disc_buckle_condition": "连接盘、插销、套管和节点是否明显缺失或异常"},
            (
                _rule(
                    "disc_buckle_condition",
                    "no",
                    "盘扣节点或插销存在明显异常",
                    "连接盘、插销、立杆套管或水平杆斜杆清晰可见缺失、脱落或明显变形。",
                    "可能涉及 JGJ/T 231 关于盘扣节点和杆件连接的要求。",
                ),
            ),
        ),
        "SCAFFOLD_CANTILEVER": _scaffold_profile(
            "SCAFFOLD_CANTILEVER",
            "悬挑式脚手架",
            ("悬挑脚手架", "悬挑式脚手架", "scaffold_cantilever"),
            "《建筑施工扣件式钢管脚手架安全技术规范》（JGJ 130）及悬挑脚手架地方标准",
            "悬挑承力架、型钢锚固、拉结支撑、架体和外侧防护。",
            {
                "cantilever_beam_anchor_visible": "悬挑梁、锚固和支承节点是否可见",
                "cantilever_support_condition": "悬挑承力架或支撑是否明显异常",
            },
            (
                _rule(
                    "cantilever_beam_anchor_visible",
                    "no",
                    "悬挑梁或锚固节点存在明显问题",
                    "悬挑梁、锚固件或支承节点清晰可见缺失、脱落、明显变形或悬空。",
                    "可能涉及 JGJ 130 及悬挑脚手架专项标准的要求，节点需现场复核。",
                ),
                _rule(
                    "cantilever_support_condition",
                    "no",
                    "悬挑承力架支撑存在明显异常",
                    "承力架、斜拉或下撑构件清晰可见缺失、脱落或明显失稳。",
                    "可能涉及悬挑脚手架专项标准关于承力架和稳定构造的要求。",
                ),
            ),
        ),
        "SCAFFOLD_ATTACHED_LIFTING": _scaffold_profile(
            "SCAFFOLD_ATTACHED_LIFTING",
            "附着式升降脚手架",
            ("附着式升降脚手架", "爬架", "提升架", "scaffold_attached_lifting"),
            "《建筑施工工具式脚手架安全技术规范》（JGJ 202）",
            "附着支承、防倾防坠、主框架、水平桁架和升降作业防护。",
            {
                "attachment_condition": "附着支承、导向和连接节点是否明显异常",
                "anti_fall_anti_tilt_visible": "防坠、防倾装置是否可见且无明显缺失",
                "lifting_zone_isolated": "升降作业区域是否设置隔离和警示",
            },
            (
                _rule(
                    "attachment_condition",
                    "no",
                    "附着支承或连接节点存在明显异常",
                    "附着支承、导向架或连接节点清晰可见缺失、脱落或明显变形。",
                    "可能涉及 JGJ 202 关于附着支承和连接的要求。",
                ),
                _rule(
                    "anti_fall_anti_tilt_visible",
                    "no",
                    "防坠或防倾装置存在明显缺失",
                    "防坠、防倾装置安装位置清晰可见，但装置明显缺失、脱落或损坏。",
                    "可能涉及 JGJ 202 关于防坠防倾装置的要求。",
                ),
                _rule(
                    "lifting_zone_isolated",
                    "no",
                    "附着式升降脚手架升降区域未见隔离",
                    "升降作业区域清晰可见，但未见警戒线、围挡或禁止人员进入措施。",
                    "可能涉及 JGJ 202 关于升降作业区域管理的要求。",
                ),
            ),
        ),
    }
)


SCENE_PROFILES["AUTO"] = {
    "label": "自动场景识别",
    "aliases": ("未指定", "智能识别", "auto", "smart"),
    "standards": "按识别出的专项场景匹配规范；未识别时按 JGJ 59-2011 做通用初筛",
    "description": "先识别图片主导作业场景，再选择一个专项规则包。",
    "object_fields": (),
    "fields": {
        "work_area_visible": "施工区域或作业面是否可见",
        "person_visible": "人员是否可见",
        "distribution_box_visible": "配电箱或开关箱是否可见",
        "high_place_visible": "高处、临边、洞口或平台是否可见",
        "foundation_pit_visible": "基坑或沟槽是否可见",
        "lifting_operation_visible": "吊装作业或悬吊物是否可见",
        "tower_crane_visible": "塔式起重机是否可见",
        "construction_hoist_visible": "施工升降机是否可见",
        "scaffold_visible": "脚手架是否可见",
    },
    "rules": (),
    "review_pairs": (),
}


def normalize_scene(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "AUTO"
    for key, profile in SCENE_PROFILES.items():
        aliases = {key.lower(), str(profile.get("label", "")).lower()}
        aliases.update(str(item).lower() for item in profile.get("aliases", ()))
        if raw in aliases or key.lower() in raw:
            return key
    if any(token in raw for token in ("吊装", "起重", "lifting")):
        return "LIFTING_OPERATIONS"
    if any(token in raw for token in ("高处", "临边", "洞口", "work_at_height")):
        return "WORK_AT_HEIGHT"
    if any(token in raw for token in ("基坑", "沟槽", "foundation_pit")):
        return "FOUNDATION_PIT"
    if any(token in raw for token in ("塔吊", "塔机", "tower_crane")):
        return "TOWER_CRANE"
    if any(token in raw for token in ("升降机", "施工电梯", "construction_hoist")):
        return "CONSTRUCTION_HOIST"
    if any(token in raw for token in ("脚手架", "scaffold")):
        return "SCAFFOLD_COUPLER_TYPE"
    if any(token in raw for token in ("用电", "配电箱", "开关箱", "电缆", "electricity")):
        return "TEMPORARY_ELECTRICITY"
    return "AUTO"


def get_scene_profile(scene: str | None) -> dict[str, Any]:
    return SCENE_PROFILES.get(normalize_scene(scene), SCENE_PROFILES["AUTO"])


def scene_catalog_text() -> str:
    rows = []
    for key, profile in SCENE_PROFILES.items():
        if key == "AUTO":
            continue
        rows.append(f"- {key}: {profile['label']}（{profile['description']}）")
    return "\n".join(rows)


def scene_fact_schema(scene: str | None) -> str:
    profile = get_scene_profile(scene)
    fields = profile.get("fields", {})
    return "\n".join(f'    "{name}": "{YES_NO_STATES}",  // {desc}' for name, desc in fields.items())


def scene_prompt_instructions(scene: str | None) -> str:
    profile = get_scene_profile(scene)
    if normalize_scene(scene) == "AUTO":
        return (
            f"候选场景：\n{scene_catalog_text()}\n"
            "请先输出 scene_type，选择与图片主导风险最匹配的场景；证据不足时输出 GENERAL_SITE。\n"
            "自动场景只做粗路由，最终事实字段以所选专项场景为准。"
        )
    return (
        f"当前检查场景：{profile['label']}（{normalize_scene(scene)}）\n"
        f"场景范围：{profile['description']}\n"
        f"主要规范依据：{profile['standards']}\n"
        "该场景是重点检查方向，但不要忽略图片中其他清晰可见的重大施工安全风险；"
        "不要把不可见的设计、功能、资质、台账或精确数值当成图片事实。"
    )


def scene_has_object(facts: dict[str, Any], scene: str | None) -> bool:
    fields = get_scene_profile(scene).get("object_fields", ())
    if not fields:
        return True
    return any(facts.get(field) == "yes" for field in fields)
