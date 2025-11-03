import json
import os
from typing import Any, Dict, Optional



class Config:

    def __init__(self, config_path: str, add_args: Optional[Dict[str, Any]] = None):
        # 检查配置文件是否存在
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"❌ 未找到配置文件: {config_path}")

        add_args = add_args or {}

        # 读取 JSON 配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # 验证文件内容是否为字典格式
        if not isinstance(config_dict, dict):
            raise ValueError(f"配置文件 {config_path} 必须包含一个 JSON 对象。")

        # 合并配置（add_args 的键值优先于文件内容）
        merged_args: Dict[str, Any] = {**config_dict, **add_args}
        self._merged_args = merged_args

        # 将键值对动态设置为类属性，便于点操作访问
        for key, value in merged_args.items():
            setattr(self, key, value)

    # -----------------------------
    # 工具方法
    # -----------------------------

    def update_from_dict(self, new_args: Dict[str, Any]) -> None:
        """
        从一个字典中更新配置参数。

        参数：
            new_args (`Dict[str, Any]`): 需要更新的键值对字典。
        """
        for key, value in new_args.items():
            setattr(self, key, value)
            self._merged_args[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """返回整个配置内容的 Python 字典形式。"""
        return dict(self._merged_args)

    def to_json_string(self) -> str:
        """返回格式化后的 JSON 字符串，便于打印或保存。"""
        return json.dumps(self._merged_args, indent=4, ensure_ascii=False)

    def __getitem__(self, item: str) -> Any:
        """允许使用字典风格访问配置项，如 `config['learning_rate']`。"""
        return getattr(self, item)

    def __repr__(self) -> str:
        """返回类的字符串表示（打印时显示为 JSON 格式）。"""
        return self.to_json_string()
    
    
# -----------------------------
# 使用示例
# -----------------------------
if __name__ == "__main__":
    # 从文件加载配置，并添加一个额外参数覆盖原值
    test_config = Config("config/test_config.json", {"learning_rate": 0.00005})

    # 打印完整配置
    print(test_config)

    # 使用字典方式访问配置项
    print(test_config["model_path"])