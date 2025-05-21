class ComplianceResult:
    def __init__(self, rule_name: str, passed: bool, explanation: str):
        self.rule_name = rule_name
        self.passed = passed
        self.explanation = explanation 