from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PACKAGE_ROOT / "主代码"
STEPS = [
    ("02_QFD接口转换", "02_全量评分转QFD接口.py"),
    ("03_神经符号EC映射", "03_分片式神经符号EC映射.py"),
    ("04_风险势能与CVaR聚合", "04_总体与品牌CVaR风险.py"),
    ("05_EC拓扑传播与Jacobian分析", "05_拓扑传播与Jacobian.py"),
    ("06_总体双品牌与EC差异分析", "06_品牌与车型EC风险差异.py"),
    ("07_EC优先级与稳定性分析", "07_EC优先级Bootstrap稳定性.py"),
    ("08_TRIZ工程方案生成", "08_TRIZ候选与风险证据.py"),
    ("09_MILP资源约束组合优化", "09_稳健MILP组合优化.py"),
    ("10_消融敏感性与稳健性实验", "10_消融与五参数敏感性.py"),
    ("10_消融敏感性与稳健性实验", "10_五参数广范围敏感性.py"),
    ("11_外部验证", "11_蔚来外部排名一致性验证.py"),
]


def main() -> None:
    for folder, filename in STEPS:
        script = CODE_ROOT / folder / filename
        if not script.is_file():
            raise FileNotFoundError(script)
        print(f"\n===== 运行：{folder}/{filename} =====", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)


if __name__ == "__main__":
    main()
