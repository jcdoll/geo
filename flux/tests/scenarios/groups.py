"""
Scenario group management for test organization.
"""

from typing import Dict, Any, Type, List, Tuple


class ScenarioGroup:
    """Container for related test scenarios."""
    
    def __init__(self, name: str, description: str):
        """Initialize scenario group."""
        self.name = name
        self.description = description
        self.scenarios: Dict[str, Tuple[Type, Dict[str, Any]]] = {}
        
    def add_scenario(self, key: str, scenario_class: Type, **default_params):
        """Add a scenario to this group."""
        self.scenarios[key] = (scenario_class, default_params)
        
    def list_scenarios(self) -> List[str]:
        """List all scenario keys in this group."""
        return list(self.scenarios.keys())