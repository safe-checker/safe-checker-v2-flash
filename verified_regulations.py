"""Verified construction-safety clauses and conservative local matching.

The visual model decides what is visible and whether it is risky. This module
only verifies its regulation candidate. An unmatched citation never removes a
risk; it is returned as requiring manual regulation verification.
"""

from __future__ import annotations

import re
from typing import Any


GB_TITLE = "建筑与市政施工现场安全卫生与职业健康通用规范"
GB_CODE = "GB 55034-2022"
GB_PDF = (
    "https://www.gov.cn/zhengce/zhengceku/2022-11/18/5727696/files/"
    "8a4cce0184d54010bea65c5296fc6ead.pdf"
)
JGJ_TITLE = "建筑与市政工程施工现场临时用电安全技术标准"
JGJ_CODE = "JGJ/T 46-2024"


def _record(
    regulation_id: str,
    scenes: str,
    risk_keys: str,
    aliases: str,
    standard_name: str,
    standard_code: str,
    article: str,
    clause_text: str,
    source_url: str,
    keywords: str,
    effective_date: str,
    official_source_url: str | None = None,
) -> dict[str, Any]:
    return {
        "regulation_id": regulation_id,
        "scene_types": scenes.split(),
        "risk_keys": risk_keys.split(),
        "risk_aliases": aliases.split("|"),
        "standard_name": standard_name,
        "standard_code": standard_code,
        "article": article,
        "clause_text": clause_text,
        "effective_date": effective_date,
        "version_status": "current",
        "source_url": source_url,
        "official_source_url": official_source_url,
        "keywords": keywords.split(),
    }


_GB_ROWS = (
    (
        "GENERAL_PPE_PROVISION",
        "GENERAL_SITE WORK_AT_HEIGHT LIFTING_OPERATIONS FOUNDATION_PIT TOWER_CRANE CONSTRUCTION_HOIST SCAFFOLD_COUPLER_TYPE SCAFFOLD_DISC_BUCKLE SCAFFOLD_CANTILEVER SCAFFOLD_ATTACHED_LIFTING",
        "PPE_MISSING PERSONAL_PROTECTIVE_EQUIPMENT_MISSING HELMET_NOT_WORN GLOVES_NOT_WORN REFLECTIVE_VEST_NOT_WORN",
        "未按作业条件配备劳动防护用品|劳动防护用品不足|未佩戴安全帽|未佩戴防护手套|未穿反光背心",
        "第2.0.5条",
        "应根据各工种的作业条件和劳动环境等为作业人员配备安全有效的劳动防护用品，并应及时开展劳动防护用品使用培训。",
        "https://gf.cabr-fire.com/article-62615.htm",
        "劳动防护用品 个人防护 防护用品",
    ),
    (
        "OVERHEAD_POWER_RESTRICTED_ZONE",
        "GENERAL_SITE TEMPORARY_ELECTRICITY LIFTING_OPERATIONS TOWER_CRANE",
        "WORK_UNDER_OVERHEAD_POWER_LINE MATERIAL_UNDER_OVERHEAD_POWER_LINE LIFTING_NEAR_OVERHEAD_POWER_LINE",
        "外电架空线路正下方施工|外电架空线路正下方吊装|外电架空线路正下方堆放材料",
        "第3.1.4条",
        "不得在外电架空线路正下方施工、吊装、搭设作业棚、建造生活设施或堆放构件、架具、材料及其他杂物等。",
        "https://gf.cabr-fire.com/article-62617.htm",
        "外电 架空线路 正下方 施工 吊装 堆放材料",
    ),
    (
        "GENERAL_WARNING_SIGN",
        "GENERAL_SITE WORK_AT_HEIGHT LIFTING_OPERATIONS FOUNDATION_PIT TOWER_CRANE CONSTRUCTION_HOIST",
        "WARNING_SIGN_MISSING DANGER_ZONE_WARNING_MISSING",
        "危险区域未设置安全警示标识|主要通道口缺少安全警示标识",
        "第3.1.2条",
        "施工现场应合理设置安全生产宣传标语和标牌，标牌设置应牢固可靠。应在主要施工部位、作业层面、危险区域以及主要通道口设置安全警示标识。",
        "https://gf.cabr-fire.com/article-62617.htm",
        "危险区域 警示标识 通道口 警示牌",
    ),
    (
        "HEIGHT_PPE_AND_PROTECTION",
        "WORK_AT_HEIGHT",
        "WORK_AT_HEIGHT_PPE_MISSING SAFETY_BELT_MISSING NO_SAFETY_HARNESS_AT_HEIGHT HIGH_PLACE_PROTECTION_MISSING",
        "高处作业未佩戴安全带|高处作业安全防护设施缺失|高处作业未采取防滑措施",
        "第3.2.1条",
        "在坠落高度基准面上方2m及以上进行高空或高处作业时，应设置安全防护设施并采取防滑措施，高处作业人员应正确佩戴安全帽、安全带等劳动防护用品。",
        "https://gf.cabr-fire.com/article-62618.htm",
        "高处作业 安全带 防滑 安全防护设施",
    ),
    (
        "HEIGHT_EDGE_OPENING_PROTECTION",
        "WORK_AT_HEIGHT FOUNDATION_PIT GENERAL_SITE",
        "EDGE_PROTECTION_MISSING OPENING_PROTECTION_MISSING UNPROTECTED_EDGE_OR_OPENING FOUNDATION_PIT_EDGE_PROTECTION_MISSING NO_EDGE_PROTECTION_FOUNDATION_PIT ELEVATOR_SHAFT_PROTECTION_MISSING",
        "临边防护缺失|洞口防护缺失|沟坑槽边沿未防护|基坑临边防护缺失|电梯井口防护缺失",
        "第3.2.3条",
        "在建工程的预留洞口、通道口、楼梯口、电梯井口等孔洞以及无围护设施或围护设施高度低于1.2m的楼层周边、楼梯侧边、平台或阳台边、屋面周边和沟、坑、槽等边沿应采取安全防护措施，并严禁随意拆除。",
        "https://gf.cabr-fire.com/article-62618.htm",
        "临边 洞口 防护 围护 坑槽边沿",
    ),
    (
        "HEIGHT_UNPROTECTED_COMPONENT",
        "WORK_AT_HEIGHT",
        "WORK_ON_UNFIXED_COMPONENT UNPROTECTED_COMPONENT_ACCESS",
        "在未固定构件上作业|在无防护管道上通行",
        "第3.2.4条",
        "严禁在未固定、无防护设施的构件及管道上进行作业或通行。",
        "https://gf.cabr-fire.com/article-62618.htm",
        "未固定构件 无防护设施 管道 作业通行",
    ),
    (
        "HEIGHT_PLATFORM_EDGE",
        "WORK_AT_HEIGHT GENERAL_SITE",
        "PLATFORM_EDGE_PROTECTION_MISSING WORK_PLATFORM_UNSAFE MOBILE_PLATFORM_GUARDRAIL_MISSING MOBILE_WORK_PLATFORM_EDGE_PROTECTION_MISSING",
        "操作平台周边未设置临边防护|作业平台临边防护缺失|移动式操作平台缺少临边防护",
        "第3.2.5条",
        "各类操作平台、载人装置应安全可靠，周边应设置临边防护，并应具有足够的强度、刚度和稳定性，施工作业荷载严禁超过其设计荷载。",
        "https://gf.cabr-fire.com/article-62618.htm",
        "操作平台 载人装置 临边防护 平台周边",
    ),
    (
        "LIFTING_SLING_CONDITION",
        "LIFTING_OPERATIONS TOWER_CRANE",
        "SLING_DAMAGED SLING_OR_SHACKLE_DEFECT LIFTING_ACCESSORY_DEFECT",
        "吊具索具明显破损|钢丝绳或吊带存在明显缺陷|卸扣卡环存在明显缺陷",
        "第3.4.2条",
        "吊具和索具的性能、规格应满足吊运要求并与环境条件相适应；作业前应检查并确认完好；承载时不得超过额定荷载。",
        "https://gf.cabr-fire.com/article-62620.htm",
        "吊具 索具 钢丝绳 吊带 卸扣 卡环 缺陷",
    ),
    (
        "LIFTING_OVERLOAD_OBLIQUE_PULL",
        "LIFTING_OPERATIONS TOWER_CRANE",
        "LIFTING_OVERLOAD OBLIQUE_PULL UNBALANCED_LIFT UNKNOWN_WEIGHT_LIFT",
        "吊装超载|斜拉斜吊|吊物明显偏吊失衡|起吊不明重量物体",
        "第3.4.3条",
        "吊装重量不应超过起重设备的额定起重量。吊装作业严禁超载、斜拉或起吊不明重量的物体。",
        "https://gf.cabr-fire.com/article-62620.htm",
        "吊装 超载 斜拉 斜吊 偏吊 失衡",
    ),
    (
        "FALLING_OBJECT_TEMP_FIX",
        "WORK_AT_HEIGHT LIFTING_OPERATIONS GENERAL_SITE",
        "HIGH_COMPONENT_NOT_FIXED FALLING_OBJECT_PROTECTION_MISSING",
        "高处构件未采取临时固定|高处部件缺少防坠措施",
        "第3.3.1条",
        "在高处安装构件、部件、设施时，应采取可靠的临时固定措施或防坠措施。",
        "https://gf.cabr-fire.com/article-62619.htm",
        "高处安装 临时固定 防坠措施 构件",
    ),
    (
        "FALLING_OBJECT_THROWING",
        "WORK_AT_HEIGHT GENERAL_SITE",
        "MATERIAL_THROWING_FROM_HEIGHT VERTICAL_CROSS_WORK_UNSAFE",
        "高处抛掷材料|上下同时拆除作业",
        "第3.3.2条",
        "在高处拆除或拆卸作业时，严禁上下同时进行。拆卸的施工材料、机具、构件、配件等，应运至地面，严禁抛掷。",
        "https://gf.cabr-fire.com/article-62619.htm",
        "高处拆除 上下同时 抛掷 拆卸材料",
    ),
    (
        "PASSAGE_OVERHEAD_PROTECTION",
        "WORK_AT_HEIGHT GENERAL_SITE",
        "PASSAGE_OVERHEAD_PROTECTION_MISSING",
        "安全通道上方未设置防护|通道缺少防坠物棚",
        "第3.3.4条",
        "安全通道上方应搭设防护设施，防护设施应具备抗高处坠物穿透的性能。",
        "https://gf.cabr-fire.com/article-62619.htm",
        "安全通道 上方防护 坠物 防护棚",
    ),
    (
        "LIFTING_EXCLUSION_ZONE",
        "LIFTING_OPERATIONS TOWER_CRANE",
        "PERSON_UNDER_SUSPENDED_LOAD LIFTING_EXCLUSION_MISSING PERSON_IN_LIFTING_DANGER_ZONE",
        "吊物下方人员停留|人员进入吊装危险区|吊装区域未隔离警示",
        "第3.4.1条",
        "吊装作业前应设置安全保护区域及警示标识，吊装作业时应安排专人监护，防止无关人员进入，严禁任何人在吊物或起重臂下停留或通过。",
        "https://gf.cabr-fire.com/article-62620.htm",
        "吊装 吊物下方 起重臂下 人员停留 安全保护区域 警示标识",
    ),
    (
        "LIFTING_TEMPORARY_FIX",
        "LIFTING_OPERATIONS TOWER_CRANE",
        "LIFTED_COMPONENT_UNSTABLE LIFTING_TEMP_FIX_MISSING",
        "吊装未形成稳定体系且无临时固定|构件未固定即解除吊具",
        "第3.4.6条",
        "吊装作业时，对未形成稳定体系的部分，应采取临时固定措施。对临时固定的构件，应在安装固定完成并经检查确认无误后，方可解除临时固定措施。",
        "https://gf.cabr-fire.com/article-62620.htm",
        "吊装构件 稳定体系 临时固定 解除固定",
    ),
    (
        "PIT_DRAINAGE",
        "FOUNDATION_PIT",
        "FOUNDATION_PIT_DRAINAGE_MISSING PIT_WATER_ACCUMULATION",
        "基坑未采取排水措施|基坑明显积水",
        "第3.5.2条",
        "边坡坡顶、基坑顶部及底部应采取截水或排水措施。",
        "https://gf.cabr-fire.com/article-62621.htm",
        "基坑 边坡 积水 截水 排水",
    ),
    (
        "SITE_MATERIAL_STACKING",
        "GENERAL_SITE FOUNDATION_PIT SCAFFOLD_COUPLER_TYPE SCAFFOLD_DISC_BUCKLE SCAFFOLD_CANTILEVER SCAFFOLD_ATTACHED_LIFTING",
        "MATERIAL_STACKING_DISORDER UNSTABLE_MATERIAL_STACK",
        "施工现场物料堆放杂乱|物料未固定存在倾倒风险",
        "第3.5.11条",
        "施工现场物料、物品等应整齐堆放，并应根据具体情况采取相应的固定措施。",
        "https://gf.cabr-fire.com/article-62621.htm",
        "物料 物品 堆放 杂乱 固定措施",
    ),
    (
        "MACHINE_GUARD",
        "GENERAL_SITE TOWER_CRANE CONSTRUCTION_HOIST",
        "MACHINE_GUARD_MISSING MACHINE_SAFETY_DEVICE_MISSING CONSTRUCTION_HOIST_GUARD_MISSING HOIST_GATE_GUARD_MISSING",
        "机械防护装置缺失|机械保险或报警装置被拆除|施工升降机围栏或层门防护缺失",
        "第3.6.3条",
        "机械上的各种安全防护装置、保险装置、报警装置应齐全有效，不得随意更换、调整或拆除。",
        "https://gf.cabr-fire.com/article-62622.htm",
        "机械 安全防护装置 保险装置 报警装置 拆除",
    ),
    (
        "MACHINE_EXCLUSION_ZONE",
        "GENERAL_SITE TOWER_CRANE CONSTRUCTION_HOIST",
        "PERSON_IN_MACHINE_DANGER_ZONE MACHINE_WORK_ZONE_UNCONTROLLED",
        "非作业人员进入机械作业区|机械作业区未设置安全区域",
        "第3.6.4条",
        "机械作业应设置安全区域，严禁非作业人员在作业区停留、通过、维修或保养机械。当进行清洁、保养、维修机械时，应设置警示标识，待切断电源、机械停稳后，方可进行操作。",
        "https://gf.cabr-fire.com/article-62622.htm",
        "机械作业区 安全区域 人员停留 机械维修 警示标识",
    ),
    (
        "VEHICLE_DANGEROUS_GOODS_SIGN",
        "GENERAL_SITE",
        "DANGEROUS_GOODS_VEHICLE_SIGN_MISSING VEHICLE_WARNING_SIGN_MISSING",
        "危险品运输车辆未悬挂警示牌|运输易燃易爆物品车辆缺少警示",
        "第3.8.1条",
        "施工车辆运输危险物品时应悬挂警示牌。",
        "https://gf.cabr-fire.com/article-62624.htm",
        "施工车辆 危险物品 警示牌 运输",
    ),
    (
        "VEHICLE_MOVING_BOARDING",
        "GENERAL_SITE",
        "BOARDING_MOVING_VEHICLE ALIGHTING_MOVING_VEHICLE PERSON_BOARDING_MOVING_VEHICLE",
        "车辆行驶中人员上下车|人员攀爬行驶中的施工车辆",
        "第3.8.3条",
        "车辆行驶过程中，严禁人员上下。",
        "https://gf.cabr-fire.com/article-62624.htm",
        "车辆 行驶 人员 上下车 攀爬",
    ),
    (
        "CONFINED_SPACE_VENTILATION",
        "GENERAL_SITE",
        "CONFINED_SPACE_VENTILATION_MISSING ENCLOSED_SPACE_VENTILATION_MISSING",
        "受限空间未设置通风换气|密闭空间未见通风设备",
        "第3.9.2条",
        "施工单位应根据施工环境设置通风、换气和照明等设备。",
        "https://gf.cabr-fire.com/article-62625.htm",
        "受限空间 密闭空间 通风 换气 照明",
    ),
    (
        "ELECTRICAL_LINE_PROTECTION",
        "GENERAL_SITE TEMPORARY_ELECTRICITY",
        "ELECTRICAL_LINE_UNPROTECTED CABLE_MECHANICAL_DAMAGE_RISK CABLE_CORROSION_RISK",
        "配电线路缺少机械损伤防护|电缆处于明显腐蚀环境",
        "第3.10.3条",
        "施工现场线缆敷设应采取有效保护措施，防止线路导体受到机械损伤和介质腐蚀。",
        "https://gf.cabr-fire.com/article-62626.htm",
        "配电线路 线缆 保护 机械损伤 介质腐蚀",
    ),
    (
        "ELECTRICAL_MAINTENANCE_LOCKOUT",
        "GENERAL_SITE TEMPORARY_ELECTRICITY",
        "LIVE_ELECTRICAL_MAINTENANCE LOCKOUT_TAGOUT_MISSING ELECTRICAL_MAINTENANCE_WARNING_MISSING",
        "带电检修电气设备|电气检修未断电上锁|检修未悬挂停电标识",
        "第3.10.5条",
        "电气设备检修、线路维修时严禁带电作业；应切断并隔离电源，确认断电，对配电间门、配电箱或开关上锁，并设置警示标识牌。",
        "https://gf.cabr-fire.com/article-62626.htm",
        "电气 检修 维修 带电作业 断电 上锁 警示标识",
    ),
    (
        "FLAMMABLE_CONTAINER_STORAGE",
        "GENERAL_SITE",
        "FLAMMABLE_CONTAINER_STORAGE_UNSAFE GAS_CYLINDER_STORAGE_UNSAFE FUEL_STORED_IN_OCCUPIED_ROOM",
        "易燃易爆容器未存放在专用场所|气瓶存放在住人用房|油料容器违规存放",
        "第3.11.1条",
        "柴油、汽油、氧气瓶、乙炔气瓶、煤气罐等易燃、易爆液体或气体容器应轻拿轻放，并设置专门的存储场所，严禁存放在住人用房。",
        "https://gf.cabr-fire.com/article-62628.htm",
        "柴油 汽油 氧气瓶 乙炔 气瓶 易燃易爆 存储场所",
    ),
    (
        "GAS_CYLINDER_ACCESSORIES",
        "GENERAL_SITE",
        "GAS_CYLINDER_ACCESSORY_MISSING OXYGEN_CYLINDER_REGULATOR_MISSING ACETYLENE_FLASHBACK_ARRESTOR_MISSING",
        "氧气瓶减压器缺损|乙炔瓶回火防止器缺失|气瓶附件明显缺损",
        "第3.11.7条",
        "压力容器及其附件应合格、完好和有效；严禁使用减压器等附件缺损的氧气瓶，严禁使用乙炔专用减压器、回火防止器等附件缺损的乙炔气瓶。",
        "https://gf.cabr-fire.com/article-62628.htm",
        "气瓶 氧气瓶 乙炔瓶 减压器 回火防止器 附件 缺损",
    ),
    (
        "WATER_WORK_LIFESAVING",
        "GENERAL_SITE",
        "WATER_WORK_LIFESAVING_MISSING LIFE_JACKET_MISSING",
        "水上作业未佩戴救生设施|水下作业人员缺少救生装备",
        "第3.14.2条",
        "水上或水下作业人员，应正确佩戴救生设施。",
        "https://gf.cabr-fire.com/article-62633.htm",
        "水上作业 水下作业 救生设施 救生衣",
    ),
    (
        "WATER_WORK_PLATFORM_PROTECTION",
        "GENERAL_SITE WORK_AT_HEIGHT",
        "WATER_PLATFORM_EDGE_PROTECTION_MISSING WATER_WORK_PROTECTION_MISSING",
        "水上操作平台周边未采取安全防护|水上作业面临边防护缺失",
        "第3.14.3条",
        "水上作业时，操作平台或操作面周边应采取安全防护措施。",
        "https://gf.cabr-fire.com/article-62633.htm",
        "水上作业 操作平台 作业面 周边 安全防护",
    ),
    (
        "CORROSIVE_MATERIAL_PROTECTION",
        "GENERAL_SITE",
        "CORROSIVE_STORAGE_UNSAFE CORROSIVE_WORK_PROTECTION_MISSING",
        "腐蚀性物质储存不当|酸碱作业缺少人员防护措施",
        "第3.15.3条",
        "具有腐蚀性的酸、碱、盐、有机物等应妥善储存、保管和使用，使用场所应有防止人员受到伤害的安全措施。",
        "https://gf.cabr-fire.com/article-62634.htm",
        "腐蚀性 酸 碱 盐 有机物 储存 防护措施",
    ),
)


_JGJ_ROWS = (
    ("TEMP_ELEC_CABLE_LAYOUT", "TEMP_ELEC_CABLE_LAYOUT CABLE_GROUND_LAYOUT CABLE_ON_GROUND", "电缆沿地面明设|电缆拖地|电缆未架空或埋地", "第6.2.3条", "电缆线路应采用埋地或架空敷设，严禁沿地面明设，并应避免机械损伤和介质腐蚀。", "https://gf.cabr-fire.com/m/article-68848.htm", "电缆 拖地 沿地面 明设 架空 埋地"),
    ("TEMP_ELEC_BOX_SPACE", "TEMP_ELEC_BOX_SPACE DISTRIBUTION_BOX_SPACE_BLOCKED", "配电箱周围堆放杂物|配电箱操作通道被占用", "第4.1.5条", "配电箱、开关箱周围应有足够2人同时工作的空间和通道，不得堆放任何妨碍操作和维修的物品，不得有灌木和杂草。", "https://gf.cabr-fire.com/m/article-68840.htm", "配电箱 开关箱 周围 操作空间 通道 堆放"),
    ("TEMP_ELEC_BOX_FIXED", "TEMP_ELEC_BOX_FIXED DISTRIBUTION_BOX_UNSTABLE", "配电箱安装不牢固|配电箱倾倒或悬空不稳", "第4.1.7条", "配电箱、开关箱应装设端正、牢固。固定式配电箱、开关箱的中心点与地面的垂直距离应为1.4m～1.6m。", "https://gf.cabr-fire.com/m/article-68840.htm", "配电箱 开关箱 端正 牢固 倾倒 不稳"),
    ("TEMP_ELEC_BOX_COMPONENT_FIX", "TEMP_ELEC_BOX_COMPONENT_FIX ELECTRICAL_COMPONENT_LOOSE", "箱内电器歪斜松动|插座未固定在安装板", "第4.1.9条", "配电箱、开关箱内的电器（含插座）应按其规定位置固定在电器安装板上，且不得歪斜和松动。", "https://gf.cabr-fire.com/m/article-68840.htm", "配电箱 插座 安装板 歪斜 松动"),
    ("TEMP_ELEC_N_PE_TERMINALS", "TEMP_ELEC_N_PE_TERMINALS N_PE_TERMINALS_NOT_SEPARATED", "N端子板和PE端子板未分设|保护零线端子设置不规范", "第4.1.10条", "配电箱、开关箱的电器安装板上必须分设N端子板和PE端子板；N端子板应与金属电器安装板绝缘，PE端子板应与金属电器安装板作电气连接。", "https://gf.cabr-fire.com/m/article-68840.htm", "N端子板 PE端子板 分设 保护零线"),
    ("TEMP_ELEC_WIRING_INSULATION", "TEMP_ELEC_WIRING_INSULATION EXPOSED_LIVE_PART OPEN_PANEL_LIVE_PARTS_EXPOSED DAMAGED_WIRE_INSULATION", "导线绝缘破损|带电部分外露|线束无绝缘保护", "第4.1.11条", "配电箱、开关箱内的连接线必须采用铜芯绝缘导线，线束应有外套绝缘管，导线应与电器端子连接牢固，不得有外露带电部分。", "https://gf.cabr-fire.com/m/article-68840.htm", "导线 绝缘 线束 外露带电 端子"),
    ("TEMP_ELEC_WIRING_ENTRY", "TEMP_ELEC_WIRING_ENTRY CABLE_ENTRY_UNPROTECTED", "进出线口无绝缘护套|电缆与箱体直接接触|进出线未成束卡固", "第4.1.15条", "配电箱、开关箱的进出线口应配置固定线卡，进出线应加绝缘护套并成束卡固在支架上，不得与箱体直接接触。", "https://gf.cabr-fire.com/m/article-68840.htm", "进出线口 固定线卡 绝缘护套 箱体接触 成束卡固"),
    ("TEMP_ELEC_BOX_WEATHER_PROTECTION", "TEMP_ELEC_BOX_WEATHER_PROTECTION DISTRIBUTION_BOX_WEATHER_PROTECTION_MISSING", "配电箱缺少防雨措施|开关箱缺少防尘措施", "第4.1.16条", "配电箱、开关箱应采取防雨、防尘措施。", "https://gf.cabr-fire.com/m/article-68840.htm", "配电箱 开关箱 防雨 防尘"),
    ("TEMP_ELEC_BOX_LABEL", "TEMP_ELEC_BOX_LABEL DISTRIBUTION_BOX_LABEL_MISSING", "配电箱缺少名称用途标记|配电箱缺少系统接线图", "第4.3.1条", "配电箱、开关箱应有名称、用途、分路标记及系统接线图。", "https://gf.cabr-fire.com/m/article-68842.htm", "配电箱 名称 用途 分路标记 系统接线图"),
    ("TEMP_ELEC_BOX_LOCK", "TEMP_ELEC_BOX_LOCK DISTRIBUTION_BOX_LOCK_MISSING", "配电箱箱门未配锁|开关箱无法上锁", "第4.3.2条", "配电箱、开关箱箱门应配锁，并应由专人负责。", "https://gf.cabr-fire.com/m/article-68842.htm", "配电箱 开关箱 箱门 配锁 锁具"),
    ("TEMP_ELEC_BOX_CLEAN", "TEMP_ELEC_BOX_CLEAN FOREIGN_OBJECT_IN_DISTRIBUTION_BOX", "配电箱内放置杂物|开关箱内不整洁", "第4.3.8条", "配电箱、开关箱内不得放置任何杂物，并应保持整洁。", "https://gf.cabr-fire.com/m/article-68842.htm", "配电箱内 开关箱内 杂物 整洁"),
    ("TEMP_ELEC_UNAUTHORIZED_CONNECTION", "TEMP_ELEC_UNAUTHORIZED_CONNECTION UNAUTHORIZED_ELECTRICAL_MODIFICATION", "配电箱内随意改动接线|违规跨接电气线路", "第4.3.10条", "配电箱、开关箱内的电器配置和接线不得随意改动；熔断器熔体更换时，不得采用不符合原规格的熔体代替。", "https://gf.cabr-fire.com/m/article-68842.htm", "配电箱 接线 随意改动 跨接 熔断器"),
    ("TEMP_ELEC_TERMINAL_EXTERNAL_FORCE", "TEMP_ELEC_TERMINAL_EXTERNAL_FORCE CABLE_SHARP_EDGE_CONTACT", "进出线受到外力|电缆接触金属尖锐断口", "第4.3.11条", "配电箱、开关箱的进出线应避免外力作用，严禁与金属尖锐断口和强腐蚀介质接触。", "https://gf.cabr-fire.com/m/article-68842.htm", "进出线 外力 尖锐断口 腐蚀介质"),
)


VERIFIED_REGULATIONS: dict[str, dict[str, Any]] = {}
for row in _GB_ROWS:
    entry = _record(
        row[0], row[1], row[2], row[3], GB_TITLE, GB_CODE,
        row[4], row[5], row[6], row[7], "2023-06-01", GB_PDF,
    )
    VERIFIED_REGULATIONS[entry["regulation_id"]] = entry
for row in _JGJ_ROWS:
    entry = _record(
        row[0], "TEMPORARY_ELECTRICITY", row[1], row[2], JGJ_TITLE, JGJ_CODE,
        row[3], row[4], row[5], row[6], "2025-01-01",
    )
    VERIFIED_REGULATIONS[entry["regulation_id"]] = entry


def _norm_code(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _norm_article(value: Any) -> str:
    return re.sub(r"[^0-9]", "", str(value or ""))


def _norm_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")


def _public(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "regulation_id", "standard_name", "standard_code", "article",
        "clause_text", "effective_date", "version_status", "source_url",
        "official_source_url",
    )
    return {key: record[key] for key in keys}


def _scene_ok(record: dict[str, Any], scene_type: str | None) -> bool:
    scene = str(scene_type or "AUTO").strip().upper()
    return scene in {"", "AUTO"} or scene in record["scene_types"]


def verify_regulation(
    *,
    candidate: dict[str, Any] | None,
    risk_key: Any,
    item_text: Any,
    target: Any,
    evidence: Any,
    scene_type: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Match exact citation first, then risk semantics with strict thresholds."""
    candidate = candidate if isinstance(candidate, dict) else {}
    combined = " ".join(str(value or "") for value in (risk_key, item_text, target, evidence))
    normalized_key = _norm_key(risk_key)
    code = _norm_code(candidate.get("standard_code"))
    article = _norm_article(candidate.get("article"))
    if code and article:
        for record in VERIFIED_REGULATIONS.values():
            record_keys = {_norm_key(value) for value in record["risk_keys"]}
            semantic_hits = sum(keyword in combined for keyword in record["keywords"] if keyword)
            semantically_compatible = (
                normalized_key in record_keys
                or any(alias in combined for alias in record["risk_aliases"] if alias)
                or semantic_hits >= 2
            )
            if (
                code == _norm_code(record["standard_code"])
                and article == _norm_article(record["article"])
                and _scene_ok(record, scene_type)
                and semantically_compatible
            ):
                return _public(record), _match("exact_code_article", 1.0, record, "标准编号和条款号与本地目录一致。")

    if normalized_key:
        for record in VERIFIED_REGULATIONS.values():
            if not _scene_ok(record, scene_type):
                continue
            if normalized_key in {_norm_key(value) for value in record["risk_keys"]}:
                return _public(record), _match("risk_key", 0.95, record, "风险键与本地现行条款映射一致。")

    for record in VERIFIED_REGULATIONS.values():
        if _scene_ok(record, scene_type) and any(alias in combined for alias in record["risk_aliases"] if alias):
            return _public(record), _match("risk_alias", 0.88, record, "风险描述与本地风险别名直接对应。")

    best: tuple[int, dict[str, Any]] | None = None
    for record in VERIFIED_REGULATIONS.values():
        if not _scene_ok(record, scene_type):
            continue
        hits = sum(keyword in combined for keyword in record["keywords"] if keyword)
        if hits >= 3 and (best is None or hits > best[0]):
            best = (hits, record)
    if best:
        hits, record = best
        confidence = min(0.86, 0.70 + hits * 0.04)
        return _public(record), _match("keyword", confidence, record, "场景兼容且多个风险关键词命中；建议复核适用性。")

    return None, {
        "method": "none",
        "confidence": 0.0,
        "verified": False,
        "catalog_regulation_id": None,
        "note": "本地目录未找到可靠对应条款；保留风险结论，法规需人工核验。",
    }


def _match(method: str, confidence: float, record: dict[str, Any], note: str) -> dict[str, Any]:
    return {
        "method": method,
        "confidence": round(confidence, 2),
        "verified": True,
        "catalog_regulation_id": record["regulation_id"],
        "note": note,
    }


def format_verified_regulation(regulation: dict[str, Any] | None) -> str:
    if not regulation:
        return "暂未匹配到本地已核验的现行条款；请人工查阅适用标准。"
    return (
        f"《{regulation['standard_name']}》{regulation['standard_code']} "
        f"{regulation['article']}：{regulation['clause_text']}"
        f"（来源：{regulation['source_url']}）"
    )
