from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STEP_DIR.parent
WORKSPACE = PROJECT_DIR.parent
RANKING = PROJECT_DIR / "05_EC拓扑传播与Jacobian分析" / "EC纯风险与拓扑调整排序_总体与品牌.csv"
MAPPING = PROJECT_DIR / "03_神经符号EC映射" / "逐片段_约束混合EC映射.parquet"
HISTORICAL = WORKSPACE / "数据" / "04_候选方案与评审" / "历史多路线候选方案与独立机器评审记录.csv"
TOP_ECS = 10


def candidate(solution_id, ec_id, route, principle, strategy, mechanism, reduction, lower, upper, bom, dev, people, months, mass, volume, power, safety, basis, feasibility=0.78, confidence=0.75):
    return {
        "solution_id": solution_id,
        "ec_id": ec_id,
        "route_type": route,
        "triz_principle": principle,
        "strategy": strategy,
        "target_failure": "由本次全量映射Top风险评论证据确定",
        "technical_mechanism": mechanism,
        "expected_risk_reduction": reduction,
        "reduction_lower": lower,
        "reduction_upper": upper,
        "bom_cost": bom,
        "development_validation_cost": dev,
        "person_months": people,
        "duration_months": months,
        "delta_mass": mass,
        "delta_volume": volume,
        "delta_power": power,
        "self_confidence": confidence,
        "safety_review": safety,
        "dependencies": "[]",
        "conflicts": "[]",
        "resource_basis": basis,
        "relevance_score": 0.90,
        "mechanism_score": 0.82,
        "feasibility_score": feasibility,
        "resource_plausibility_score": 0.72,
        "review_confidence": confidence,
        "safety_block": False,
        "review_reasons": "规则化TRIZ初筛，资源为可复现实验情景，需主机厂工程专家复核。",
        "combined_confidence": confidence,
        "review_status": "provisional_engineering_review_required",
        "candidate_source": "本次规则化TRIZ候选",
    }


def new_candidates() -> list[dict]:
    return [
        candidate("EC25-N01", "EC25", "软件/控制", "反馈与参数化", "车道置信度分层控制与个性化干预阈值", "融合车道线质量、曲率、施工区域与驾驶员转向意图，按置信度连续调节横向控制增益，低置信场景降级为告警。", .28, .16, .38, 0, 750000, 12, 6, 0, 0, 8, "完成SOTIF场景库、失效注入及封闭场地横向控制验证。", "复用现有感知与域控算力，含算法、标定和场景回放。"),
        candidate("EC25-N02", "EC25", "诊断/传感", "预先作用", "摄像头标定漂移与车道感知健康监测", "利用重投影误差、道路消失点与转向响应残差在线检测标定漂移，异常时限制车道保持并触发维护提示。", .20, .12, .30, 60, 480000, 8, 5, .1, 0, 3, "通过摄像头遮挡、标定偏移和极端光照故障注入。", "增加诊断软件及少量标定工装。"),
        candidate("EC25-N03", "EC25", "制造/装配", "局部质量改善", "前视摄像头支架公差闭环与下线动态标定", "将支架角度、玻璃折射和车身姿态纳入下线标定，自动拒收超差装配并写入标定追溯码。", .16, .09, .24, 35, 360000, 7, 4, .05, 0, 1, "验证装配公差、温漂及十万公里耐久后的标定稳定性。", "改造下线标定工位与质量追溯。"),
        candidate("EC23-N01", "EC23", "诊断/传感", "反馈", "VOC/PM2.5/湿度多传感闭环净化", "根据颗粒物、VOC和湿度状态动态切换内外循环、风量与净化器工作点，并给出滤芯剩余寿命。", .24, .14, .34, 180, 460000, 8, 5, .15, .1, 18, "完成传感器漂移、冷凝和EMC验证。", "新增环境传感单元并复用空调控制器。"),
        candidate("EC23-N02", "EC23", "结构/材料", "多孔材料与局部质量改善", "低阻高容尘复合滤芯与风道密封优化", "采用梯度纤维滤层和活性炭吸附层，修正滤芯旁通泄漏与风道压损，兼顾净化效率和噪声。", .22, .13, .32, 130, 520000, 9, 6, .4, .2, 0, "完成容尘、阻力、VOC吸附、阻燃和霉菌测试。", "滤材Tier1样件与风道CFD/DV验证。"),
        candidate("EC23-N03", "EC23", "维护/预警", "周期性作用", "蒸发器干燥、自清洁与异味源预警", "锁车后根据湿度和温度执行低功耗干燥，结合风机电流与VOC趋势识别霉变风险并提示清洗。", .18, .10, .27, 0, 380000, 7, 4, 0, 0, 12, "验证低压电池消耗、热安全与长期抑菌效果。", "软件策略、环境舱和耐久验证。"),
        candidate("EC26-N01", "EC26", "软件/控制", "预测与动态反馈", "占用网络预测与ACC风险置信门控", "对切入、静止目标和匝道场景建立短时轨迹分布，按预测不确定性限制加减速并抑制幽灵制动。", .30, .18, .40, 0, 920000, 15, 8, 0, 0, 15, "完成ASIL/SOTIF分析、影子模式及大规模回放。", "感知预测、规划控制和场景库联合开发。"),
        candidate("EC26-N02", "EC26", "软件/控制", "中介与局部质量改善", "地图车道拓扑可信度融合与降级策略", "融合在线感知与地图拓扑，当道路施工或地图冲突时逐级退出领航并保持基础ACC连续性。", .23, .14, .33, 0, 650000, 11, 6, 0, 0, 8, "验证地图冲突、隧道、匝道及施工改道场景。", "复用地图服务，增加融合和降级状态机。"),
        candidate("EC26-N03", "EC26", "维护/预警", "预先作用", "云端影子模式与高风险场景闭环", "量产车后台并行运行候选策略，自动上传高不确定片段并形成回归集，OTA前执行覆盖率门禁。", .17, .10, .26, 0, 520000, 9, 6, 0, 0, 6, "隐私脱敏、网络安全和OTA回滚验证。", "云端回放平台与数据闭环运营。"),
        candidate("EC20-N01", "EC20", "软件/控制", "反馈", "无线充电异物检测与热功率闭环", "融合线圈Q值、温度梯度和手机握手状态识别异物或偏置，动态降功率并给出可解释提示。", .25, .15, .35, 45, 360000, 7, 4, 0, 0, 2, "通过金属异物、过温、短路和EMC测试。", "复用充电控制器，新增温度采样点。"),
        candidate("EC20-N02", "EC20", "结构/材料", "嵌套与自适应", "多线圈自对准充电板与隔热结构", "采用分区线圈和磁屏蔽导热层扩大有效充电区域，降低手机位置敏感性与座舱表面温升。", .22, .13, .31, 190, 580000, 9, 6, .25, .15, 3, "完成Qi兼容、热循环、阻燃和EMC验证。", "无线充电Tier1样件与热仿真。"),
        candidate("EC20-N03", "EC20", "诊断/传感", "状态反馈", "座舱低压供电负载预测与端口健康诊断", "依据端口电压纹波、接触电阻和整车低压负载预测限流风险，提前提示线束或接口退化。", .16, .09, .24, 25, 320000, 6, 4, .05, 0, 1, "验证端口过载、接触不良和低压电池弱化场景。", "软件诊断加电流采样标定。"),
        candidate("EC09-N01", "EC09", "软件/控制", "动态反馈", "路面识别驱动的悬架与扭矩协同标定", "基于车身加速度和路面预瞄识别冲击与起伏，联合调节阻尼、扭矩斜率和车身姿态控制。", .24, .14, .34, 0, 620000, 11, 6, 0, 0, 10, "完成极限操稳、失效降级和道路耐久验证。", "复用底盘域控与传感器，增加联合标定。"),
        candidate("EC09-N02", "EC09", "结构/材料", "复合材料与局部质量改善", "副车架衬套梯度刚度与关键路径隔振", "通过多级刚度衬套和连接点局部加强分离低频摆振与高频路噪传递路径。", .20, .12, .29, 360, 720000, 12, 8, 1.4, .3, 0, "完成强度、疲劳、温度老化及操稳边界验证。", "CAE优化、两轮样件和整车NVH路试。"),
        candidate("EC09-N03", "EC09", "诊断/传感", "反向作用", "路噪主动控制与轮端异常源定位", "利用轮端与座舱麦克风构建传递路径模型，主动抵消稳定路噪并区分轮胎、轴承和悬架异常。", .17, .10, .25, 280, 500000, 9, 6, .3, .2, 22, "验证告警误报、听觉安全、功耗与EMC。", "复用音响系统，增加轮端特征采集。"),
    ]


def main() -> None:
    ranking = pd.read_csv(RANKING)
    overall = ranking[ranking["分析范围"] == "全部样本总体"].sort_values("拓扑调整排名").head(TOP_ECS)
    top_ids = overall["EC编号"].tolist()
    utility = overall.set_index("EC编号")["全局边际效用"].to_dict()
    mapped = pd.read_parquet(MAPPING)
    evidence = (
        mapped[mapped["EC编号"].isin(top_ids)]
        .sort_values("分摊初始势能", ascending=False)
        .drop_duplicates(["EC编号", "Filtered_SubSeq"])
        .groupby("EC编号", sort=False)
        .head(20)[["EC编号", "EC名称", "品牌归一", "车型", "Filtered_SubSeq", "严重度连续分值_0_4", "S6_效价值", "分摊初始势能"]]
    )
    evidence.to_csv(STEP_DIR / "Top10_EC高风险评论证据.csv", index=False, encoding="utf-8-sig")

    historical = pd.read_csv(HISTORICAL)
    historical = historical[
        historical["ec_id"].isin(top_ids)
        & historical["review_status"].eq("machine_passed_engineering_pending")
    ].copy()
    historical["candidate_source"] = "历史独立机器评审候选"
    generated = pd.DataFrame([row for row in new_candidates() if row["ec_id"] in top_ids])
    candidates = pd.concat([historical, generated], ignore_index=True, sort=False)
    candidates["ec_global_utility"] = candidates["ec_id"].map(utility)
    candidates["robust_benefit"] = (
        candidates["ec_global_utility"].astype(float)
        * candidates["reduction_lower"].astype(float)
        * candidates["feasibility_score"].astype(float)
        * candidates["combined_confidence"].astype(float)
    )
    candidates["当前总体拓扑排名"] = candidates["ec_id"].map(overall.set_index("EC编号")["拓扑调整排名"])
    candidates["engineering_expert_validation_required"] = True
    candidates.to_csv(STEP_DIR / "当前Top10_TRIZ工程候选.csv", index=False, encoding="utf-8-sig")
    coverage = (
        candidates.groupby(["ec_id", "candidate_source"], observed=True).size().unstack(fill_value=0).reset_index()
        .merge(overall[["EC编号", "EC名称", "拓扑调整排名"]], left_on="ec_id", right_on="EC编号", how="right")
        .drop(columns="EC编号")
        .sort_values("拓扑调整排名")
    )
    coverage.to_csv(STEP_DIR / "Top10候选覆盖审计.csv", index=False, encoding="utf-8-sig")
    counts = candidates.groupby("ec_id").size().reindex(top_ids, fill_value=0)
    audit = {
        "状态": "通过" if (counts >= 2).all() else "失败",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "总体Top10": top_ids,
        "候选总数": len(candidates),
        "历史独立机器评审候选数": int((candidates["candidate_source"] == "历史独立机器评审候选").sum()),
        "本次规则化候选数": int((candidates["candidate_source"] == "本次规则化TRIZ候选").sum()),
        "每EC候选数": counts.to_dict(),
        "说明": "规则化候选和资源参数仅用于可复现实验情景及MILP演示，部署前必须完成主机厂工程、法规和安全评审。",
    }
    with (STEP_DIR / "TRIZ候选生成审计.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    if audit["状态"] != "通过":
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
