"""
Lightweight test visualizer that extends the normal visualizer.

Simply prints test status to terminal and handles test-specific reset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from visualizer import GeologyVisualizer
from tests.test_framework import ScenarioRunner


class TestScenarioVisualizer(GeologyVisualizer):
    """Minimal extension of GeologyVisualizer for test scenarios."""
    
    def __init__(self, scenario_runner: ScenarioRunner, max_steps: int = 1000, **kwargs):
        """Initialize test visualizer.
        
        Args:
            scenario_runner: The scenario runner to visualize
            max_steps: Maximum steps before auto-stopping
            **kwargs: Additional arguments for GeologyVisualizer
        """
        self.scenario_runner = scenario_runner
        self.max_steps = max_steps
        self.test_completed = False
        
        # Set simulation dimensions from runner
        if 'sim_width' not in kwargs:
            kwargs['sim_width'] = scenario_runner.sim_width
        if 'sim_height' not in kwargs:
            kwargs['sim_height'] = scenario_runner.sim_height
            
        super().__init__(**kwargs)
        
        # Replace default simulation with test scenario
        self.simulation = self.scenario_runner.setup()
        
        # Get initial evaluation
        self._evaluate_and_print()
        
    def _evaluate_and_print(self):
        """Evaluate current state and print to terminal."""
        evaluation = self.scenario_runner.scenario.evaluate(self.simulation)
        
        # Print status
        step = self.scenario_runner.step_count
        success = evaluation.get('success', False)
        message = evaluation.get('message', '')
        
        status = "PASS" if success else "FAIL" if self.test_completed else "..."
        print(f"[Step {step:4d}] {status} - {message}")
        
        # Print key metrics if they exist
        if 'metrics' in evaluation and evaluation['metrics']:
            metrics_str = ", ".join(f"{k}={v}" for k, v in evaluation['metrics'].items() 
                                  if k != 'step')  # Skip step since we already show it
            if metrics_str:
                print(f"            Metrics: {metrics_str}")
        
    def step_simulation(self):
        """Step the simulation and print test status."""
        if not self.test_completed and self.scenario_runner.step_count < self.max_steps:
            # Run one step
            self.scenario_runner.step()
            
            # Evaluate and print
            self._evaluate_and_print()
            
            # Check if test completed
            evaluation = self.scenario_runner.scenario.evaluate(self.simulation)
            if evaluation.get('success', False):
                self.test_completed = True
                print(f"\n✓ Test passed at step {self.scenario_runner.step_count}!")
                
        elif self.scenario_runner.step_count >= self.max_steps:
            self.test_completed = True
            print(f"\n✗ Test reached maximum steps ({self.max_steps}) without passing")
                        
    def _handle_keyboard(self, event):
        """Handle keyboard input with test-specific additions."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            # Override reset to restore test scenario (not planet default)
            self._reset_to_scenario()
        else:
            # Delegate all other keys to parent
            super()._handle_keyboard(event)
        
    def _reset_to_scenario(self):
        """Reset simulation to the test scenario's initial state."""
        print("Resetting to test scenario initial state...")
        
        # Re-run setup
        self.scenario_runner.setup()
        self.simulation = self.scenario_runner.sim
        
        # Reset evaluation
        self.scenario_runner.step_count = 0
        self.scenario_runner.evaluation_history = []
        self.current_evaluation = self.scenario_runner.scenario.evaluate(self.simulation)
        self.test_completed = False
        
        # Pause after reset
        self.paused = True
        
 