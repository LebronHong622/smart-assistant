"""
文档验证领域服务
提供纯业务逻辑的文档验证规则
"""

from typing import List, Optional
from pydantic import BaseModel
from enum import Enum


class ValidationSeverity(str, Enum):
    """验证严重级别"""
    ERROR = "error"  # 错误，必须修复
    WARNING = "warning"  # 警告，建议修复
    INFO = "info"  # 信息，仅供参考


class ValidationError(BaseModel):
    """验证错误"""
    code: str
    message: str
    severity: ValidationSeverity
    field: Optional[str] = None

    model_config = {"frozen": True}


class ValidationResult(BaseModel):
    """验证结果值对象"""
    is_valid: bool
    errors: List[ValidationError]

    model_config = {"frozen": True}

    def has_errors(self) -> bool:
        """是否有错误级别的验证问题"""
        return any(e.severity == ValidationSeverity.ERROR for e in self.errors)

    def has_warnings(self) -> bool:
        """是否有警告级别的验证问题"""
        return any(e.severity == ValidationSeverity.WARNING for e in self.errors)

    def get_error_count(self) -> int:
        """获取错误数量"""
        return sum(1 for e in self.errors if e.severity == ValidationSeverity.ERROR)

    def get_warning_count(self) -> int:
        """获取警告数量"""
        return sum(1 for e in self.errors if e.severity == ValidationSeverity.WARNING)


class DocumentValidationService:
    """
    文档验证领域服务
    负责文档验证的纯业务逻辑
    """

    def __init__(self):
        # 默认的验证规则配置
        self.min_content_length = 10
        self.max_content_length = 100000
        self.required_metadata_fields = []

    def validate_content(self, content: str) -> ValidationResult:
        """
        验证文档内容

        Args:
            content: 文档内容

        Returns:
            验证结果
        """
        errors = []

        if not content or not content.strip():
            errors.append(ValidationError(
                code="EMPTY_CONTENT",
                message="文档内容不能为空",
                severity=ValidationSeverity.ERROR,
                field="content"
            ))
        else:
            # 检查长度
            content_length = len(content.strip())

            if content_length < self.min_content_length:
                errors.append(ValidationError(
                    code="CONTENT_TOO_SHORT",
                    message=f"文档内容过短，至少需要 {self.min_content_length} 个字符",
                    severity=ValidationSeverity.ERROR,
                    field="content"
                ))

            if content_length > self.max_content_length:
                errors.append(ValidationError(
                    code="CONTENT_TOO_LONG",
                    message=f"文档内容过长，最多支持 {self.max_content_length} 个字符",
                    severity=ValidationSeverity.WARNING,
                    field="content"
                ))

        return ValidationResult(
            is_valid=not any(e.severity == ValidationSeverity.ERROR for e in errors),
            errors=errors
        )

    def validate_metadata(self, metadata: dict, required_fields: List[str] = None) -> ValidationResult:
        """
        验证文档元数据

        Args:
            metadata: 文档元数据字典
            required_fields: 必需字段列表

        Returns:
            验证结果
        """
        errors = []
        required_fields = required_fields or self.required_metadata_fields

        # 检查必需字段
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                errors.append(ValidationError(
                    code="MISSING_REQUIRED_FIELD",
                    message=f"缺少必需的元数据字段: {field}",
                    severity=ValidationSeverity.ERROR,
                    field=field
                ))

        # 检查常见字段的合法性
        if "title" in metadata:
            title = metadata["title"]
            if not title or not str(title).strip():
                errors.append(ValidationError(
                    code="INVALID_TITLE",
                    message="文档标题不能为空",
                    severity=ValidationSeverity.ERROR,
                    field="title"
                ))

        if "language" in metadata:
            language = metadata["language"]
            if language not in ["zh", "en", "ja", "ko", "other"]:
                errors.append(ValidationError(
                    code="INVALID_LANGUAGE",
                    message=f"不支持的语言代码: {language}",
                    severity=ValidationSeverity.WARNING,
                    field="language"
                ))

        return ValidationResult(
            is_valid=not any(e.severity == ValidationSeverity.ERROR for e in errors),
            errors=errors
        )

    def validate_document(self, content: str, metadata: dict, required_metadata_fields: List[str] = None) -> ValidationResult:
        """
        综合验证文档内容和元数据

        Args:
            content: 文档内容
            metadata: 文档元数据
            required_metadata_fields: 必需的元数据字段

        Returns:
            验证结果
        """
        content_result = self.validate_content(content)
        metadata_result = self.validate_metadata(metadata, required_metadata_fields)

        all_errors = content_result.errors + metadata_result.errors

        return ValidationResult(
            is_valid=not any(e.severity == ValidationSeverity.ERROR for e in all_errors),
            errors=all_errors
        )

    def validate_chunk(self, chunk: str) -> ValidationResult:
        """
        验证文档分块

        Args:
            chunk: 文档分块

        Returns:
            验证结果
        """
        errors = []

        if not chunk or not chunk.strip():
            errors.append(ValidationError(
                code="EMPTY_CHUNK",
                message="文档分块不能为空",
                severity=ValidationSeverity.ERROR
            ))
        else:
            # 分块通常不需要太长
            if len(chunk) > 5000:
                errors.append(ValidationError(
                    code="CHUNK_TOO_LONG",
                    message=f"文档分块过长: {len(chunk)} 字符",
                    severity=ValidationSeverity.WARNING
                ))

        return ValidationResult(
            is_valid=not any(e.severity == ValidationSeverity.ERROR for e in errors),
            errors=errors
        )

    def set_validation_rules(self, min_length: int = None, max_length: int = None, required_fields: List[str] = None):
        """
        自定义验证规则

        Args:
            min_length: 最小内容长度
            max_length: 最大内容长度
            required_fields: 必需的元数据字段
        """
        if min_length is not None and min_length > 0:
            self.min_content_length = min_length

        if max_length is not None and max_length > 0:
            self.max_content_length = max_length

        if required_fields is not None:
            self.required_metadata_fields = required_fields
