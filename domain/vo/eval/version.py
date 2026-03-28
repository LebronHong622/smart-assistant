"""
版本值对象
遵循主版本.次版本格式：vX.Y 或 X.Y
"""
import re
from pydantic import BaseModel, field_validator


class Version(BaseModel):
    """版本值对象

    遵循主版本.次版本格式：
    - 主版本：重大变更，不兼容旧版本
    - 次版本：新增功能，向后兼容
    """
    major: int
    minor: int

    @field_validator('major', 'minor')
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """验证版本号不能为负数"""
        if v < 0:
            raise ValueError("版本号不能为负数")
        return v

    @classmethod
    def parse(cls, version_str: str) -> 'Version':
        """解析版本字符串

        支持格式：v1.0, 1.0, V1.0
        """
        match = re.match(r'^[vV]?(\d+)\.(\d+)$', version_str.strip())
        if not match:
            raise ValueError(f"无效版本格式: {version_str}, 应为 vX.Y 或 X.Y 格式")
        return cls(major=int(match.group(1)), minor=int(match.group(2)))

    def to_string(self) -> str:
        """转换为标准字符串格式 vX.Y"""
        return f"v{self.major}.{self.minor}"

    def next_minor(self) -> 'Version':
        """生成下一个次版本

        新增功能，主版本不变，次版本+1
        """
        return Version(major=self.major, minor=self.minor + 1)

    def next_major(self) -> 'Version':
        """生成下一个主版本

        重大变更，主版本+1，次版本归零
        """
        return Version(major=self.major + 1, minor=0)

    def __gt__(self, other: 'Version') -> bool:
        """比较版本大小"""
        if self.major > other.major:
            return True
        if self.major == other.major and self.minor > other.minor:
            return True
        return False

    def __ge__(self, other: 'Version') -> bool:
        return self > other or self == other

    def __lt__(self, other: 'Version') -> bool:
        return not self >= other

    def __le__(self, other: 'Version') -> bool:
        return not self > other

    def __str__(self) -> str:
        return self.to_string()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.major == other.major and self.minor == other.minor

    def __hash__(self) -> int:
        return hash((self.major, self.minor))
