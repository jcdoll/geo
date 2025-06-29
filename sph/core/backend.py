"""
Backend selection and dispatch system for SPH.

Supports three backends:
1. CPU (NumPy) - Always available, baseline implementation
2. Numba - JIT-compiled CPU, 10-50x faster
3. GPU (CuPy) - CUDA GPU, 50-200x faster for large N

The backend can be selected globally or per-function call.
"""

import enum
import warnings
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass


class Backend(enum.Enum):
    """Available computation backends."""
    CPU = "cpu"      # NumPy (always available)
    NUMBA = "numba"  # Numba JIT
    GPU = "gpu"      # CuPy/CUDA


@dataclass
class BackendInfo:
    """Information about a backend."""
    backend: Backend
    available: bool
    device_name: str = "CPU"
    memory_gb: float = 0.0
    compute_capability: tuple = (0, 0)


class BackendManager:
    """Manages backend selection and dispatching."""
    
    def __init__(self):
        self._current_backend = Backend.CPU
        self._available_backends = {}
        self._implementations = {}
        self._detect_backends()
        
    def _detect_backends(self):
        """Detect which backends are available."""
        # CPU is always available
        self._available_backends[Backend.CPU] = BackendInfo(
            backend=Backend.CPU,
            available=True,
            device_name="CPU (NumPy)"
        )
        
        # Check for Numba
        try:
            import numba
            self._available_backends[Backend.NUMBA] = BackendInfo(
                backend=Backend.NUMBA,
                available=True,
                device_name=f"CPU (Numba {numba.__version__})"
            )
        except ImportError:
            self._available_backends[Backend.NUMBA] = BackendInfo(
                backend=Backend.NUMBA,
                available=False
            )
        
        # Check for GPU (CuPy or PyTorch)
        gpu_available = False
        
        # Try PyTorch first (better RTX 5080 support)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                device_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                memory_gb = props.total_memory / 1e9
                compute_capability = (props.major, props.minor)
                
                self._available_backends[Backend.GPU] = BackendInfo(
                    backend=Backend.GPU,
                    available=True,
                    device_name=f"{device_name} (PyTorch)",
                    memory_gb=memory_gb,
                    compute_capability=compute_capability
                )
        except ImportError:
            pass
        
        # Fall back to CuPy if PyTorch not available
        if not gpu_available:
            try:
                import cupy as cp
                # Get GPU info
                device = cp.cuda.Device()
                props = device.attributes
                memory = device.mem_info[1] / 1e9  # Total memory in GB
                
                self._available_backends[Backend.GPU] = BackendInfo(
                    backend=Backend.GPU,
                    available=True,
                    device_name=f"{device.name.decode()} (CuPy)",
                    memory_gb=memory,
                    compute_capability=(props['ComputeCapabilityMajor'], 
                                      props['ComputeCapabilityMinor'])
                )
            except (ImportError, Exception):
                self._available_backends[Backend.GPU] = BackendInfo(
                    backend=Backend.GPU,
                    available=False
                )
    
    @property
    def current_backend(self) -> Backend:
        """Get current backend."""
        return self._current_backend
    
    @property
    def available_backends(self) -> list[Backend]:
        """Get list of available backends."""
        return [b for b, info in self._available_backends.items() if info.available]
    
    def set_backend(self, backend: Backend) -> bool:
        """Set the current backend.
        
        Args:
            backend: Backend to use
            
        Returns:
            True if backend was set successfully
        """
        if not self._available_backends[backend].available:
            warnings.warn(f"Backend {backend.value} not available, keeping {self._current_backend.value}")
            return False
        
        self._current_backend = backend
        print(f"Backend set to: {self._available_backends[backend].device_name}")
        return True
    
    def auto_select_backend(self, n_particles: int) -> Backend:
        """Automatically select best backend based on problem size.
        
        Args:
            n_particles: Number of particles
            
        Returns:
            Selected backend
        """
        # Heuristics for backend selection
        if self._available_backends[Backend.GPU].available and n_particles > 50000:
            # GPU is best for large problems
            return Backend.GPU
        elif self._available_backends[Backend.NUMBA].available and n_particles > 1000:
            # Numba is best for medium problems
            return Backend.NUMBA
        else:
            # CPU for small problems or if nothing else available
            return Backend.CPU
    
    def register_implementation(self, function_name: str, backend: Backend, 
                               implementation: Callable):
        """Register a backend-specific implementation.
        
        Args:
            function_name: Name of the function
            backend: Backend for this implementation
            implementation: The implementation function
        """
        if function_name not in self._implementations:
            self._implementations[function_name] = {}
        self._implementations[function_name][backend] = implementation
    
    def get_implementation(self, function_name: str, 
                          backend: Optional[Backend] = None) -> Callable:
        """Get implementation for a function.
        
        Args:
            function_name: Name of the function
            backend: Backend to use (None for current)
            
        Returns:
            Implementation function
            
        Raises:
            ValueError: If no implementation found
        """
        if backend is None:
            backend = self._current_backend
        
        if function_name not in self._implementations:
            raise ValueError(f"No implementations registered for {function_name}")
        
        # Try requested backend
        if backend in self._implementations[function_name]:
            return self._implementations[function_name][backend]
        
        # Fall back to CPU
        if Backend.CPU in self._implementations[function_name]:
            if backend != Backend.CPU:
                warnings.warn(f"No {backend.value} implementation for {function_name}, using CPU")
            return self._implementations[function_name][Backend.CPU]
        
        raise ValueError(f"No implementation found for {function_name}")
    
    def dispatch(self, function_name: str, *args, backend: Optional[Backend] = None, **kwargs):
        """Dispatch a function call to appropriate backend.
        
        Args:
            function_name: Name of the function
            *args: Positional arguments
            backend: Backend to use (None for current)
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        impl = self.get_implementation(function_name, backend)
        return impl(*args, **kwargs)
    
    def print_info(self):
        """Print information about available backends."""
        print("\nSPH Backend Information")
        print("=" * 60)
        
        for backend, info in self._available_backends.items():
            status = "✓" if info.available else "✗"
            print(f"{status} {backend.value:6s}: {info.device_name}")
            
            if backend == Backend.GPU and info.available:
                print(f"           Memory: {info.memory_gb:.1f} GB")
                print(f"           Compute: {info.compute_capability[0]}.{info.compute_capability[1]}")
        
        print(f"\nCurrent backend: {self._current_backend.value}")
        print("=" * 60)


# Global backend manager instance
_backend_manager = BackendManager()


# Public API
def set_backend(backend: str) -> bool:
    """Set the global backend.
    
    Args:
        backend: 'cpu', 'numba', or 'gpu'
        
    Returns:
        True if successful
    """
    try:
        backend_enum = Backend(backend.lower())
        return _backend_manager.set_backend(backend_enum)
    except ValueError:
        warnings.warn(f"Invalid backend: {backend}. Choose from: cpu, numba, gpu")
        return False


def get_backend() -> str:
    """Get current backend name."""
    return _backend_manager.current_backend.value


def list_backends() -> Dict[str, bool]:
    """Get dictionary of backend availability."""
    return {
        b.value: info.available 
        for b, info in _backend_manager._available_backends.items()
    }


def auto_select_backend(n_particles: int) -> str:
    """Auto-select best backend for particle count."""
    backend = _backend_manager.auto_select_backend(n_particles)
    _backend_manager.set_backend(backend)
    return backend.value


def print_backend_info():
    """Print backend information."""
    _backend_manager.print_info()


# Decorator for backend-specific implementations
def backend_function(function_name: str):
    """Decorator to register backend-specific implementations.
    
    Usage:
        @backend_function("compute_density")
        @for_backend(Backend.NUMBA)
        def compute_density_numba(...):
            ...
    """
    def decorator(func):
        # Check if function has _backend attribute set by @for_backend
        if hasattr(func, '_backend'):
            backend = func._backend
            _backend_manager.register_implementation(function_name, backend, func)
        return func
    return decorator


def for_backend(backend: Backend):
    """Helper decorator to specify backend."""
    def decorator(func):
        func._backend = backend
        return func
    return decorator


# Dispatch functions
def dispatch(function_name: str, *args, backend: Optional[str] = None, **kwargs):
    """Dispatch function to appropriate backend.
    
    Args:
        function_name: Name of the function
        *args: Positional arguments
        backend: Override backend (None for current)
        **kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    backend_enum = Backend(backend) if backend else None
    return _backend_manager.dispatch(function_name, *args, backend=backend_enum, **kwargs)