import gc
import psutil
import numpy as np
import torch
import logging

class MemoryManager:
    def __init__(self):
        self.allocated_memory = 0
        self.max_memory_limit = psutil.virtual_memory().total * 0.8  # 80% of total RAM
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def optimize_allocation(self):
        """Free up unused memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Memory optimization completed.")

    def allocate_memory(self, size, dtype=np.float32):
        """Allocate memory for a given size and data type."""
        if self.allocated_memory + size > self.max_memory_limit:
            logging.warning("Memory limit exceeded. Optimizing memory...")
            self.optimize_allocation()
        self.allocated_memory += size
        logging.info(f"Allocated {size} bytes of memory. Total allocated: {self.allocated_memory} bytes.")
        return np.zeros(size, dtype=dtype)

    def deallocate_memory(self, obj):
        """Deallocate memory for a given object."""
        del obj
        self.optimize_allocation()
        logging.info("Object deallocated and memory optimized.")

    def monitor_usage(self):
        """Monitor system memory usage and optimize if necessary."""
        usage = psutil.virtual_memory().percent
        if usage > 85:
            logging.warning(f"High memory usage detected: {usage}%. Optimizing memory...")
            self.optimize_allocation()

    def adaptive_caching(self, model):
        """Disable gradients for large models to reduce memory usage."""
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if model_size * 4 > self.max_memory_limit:  # 4 bytes per float32
            for param in model.parameters():
                param.requires_grad = False
            logging.info("Adaptive caching applied: Gradients disabled for large model.")
        return model

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.optimize_allocation()
        logging.info("MemoryManager context exited. Memory optimized.")

# Example usage
if __name__ == "__main__":
    memory_manager = MemoryManager()

    # Allocate memory
    array = memory_manager.allocate_memory(1000000)  # 1 million elements
    print(array.shape)

    # Deallocate memory
    memory_manager.deallocate_memory(array)

    # Monitor usage
    memory_manager.monitor_usage()

    # Adaptive caching for a model
    model = torch.nn.Linear(1000, 1000)
    model = memory_manager.adaptive_caching(model)

    # Using context manager
    with MemoryManager() as mm:
        array = mm.allocate_memory(1000000)
        print(array.shape)