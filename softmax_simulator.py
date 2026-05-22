#!/usr/bin/env python3
"""
RISC-V Vector Processor Softmax Simulator

This simulator models a RISC-V vector processor executing softmax operations
with configurable architecture parameters and instruction scheduling.
"""

from enum import Enum
import concurrent.futures
import csv
import os
import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import math
import copy
import argparse


class InstructionType(Enum):
    REDUCE = "reduce"
    FMA = "fma" 
    LOAD = "load"
    STORE = "store"
    EXP2 = "exp2"


class ExecutionMode(Enum):
    IN_ORDER = "in_order"
    OUT_OF_ORDER = "out_of_order"


def _dependency_set(dependencies) -> Set[int]:
    """Normalize optional dependency inputs to a set of instruction IDs."""
    if dependencies is None:
        return set()
    return set(dependencies)


@dataclass
class ProcessorConfig:
    """Configuration parameters for the RISC-V vector processor"""
    register_width: int  # rl: 512, 1024, 2048 bits
    reduce_compute_unit_width: int  # vl: 128, 256, 512, 1024 bits for reduce operations
    simple_elementwise_compute_unit_width: int  # vl: 128, 256, 512, 1024 bits for simple elementwise (FMA)
    complex_elementwise_compute_unit_width: int  # vl: 128, 256, 512, 1024 bits for complex elementwise (EXP2)
    cache_bandwidth: int  # 32, 64, 128 bytes per cycle
    execution_mode: ExecutionMode
    chaining_enabled: bool = False
    chaining_granularity: int = 32  # 32, 64, 128, 256 bytes
    
    # Instruction latencies
    reduce_latency: int = 7
    fma_latency: int = 4
    load_latency: int = 10
    store_latency: int = 10
    exp2_latency: int = 20
    
    # Oldest not-yet-completed uops visible to the issue queue.
    issue_queue_window: int = 10

    # Number of not-yet-issued uops the out-of-order scheduler may choose from
    # inside the issue queue window.
    ooo_scheduler_window_size: int = 16

    # When disabled, repeated logical registers within a context create
    # RAW/WAR/WAW dependencies across independent heads or rows.
    register_renaming: bool = True

    # Issue width (max μops per cycle)
    issue_width: int = 2

    # Number of hardware contexts (register groups)
    num_contexts: int = 1

    # RVV multi-lane stride CSRs: vlane0-vlane3, in bytes.
    lane_strides: Tuple[int, int, int, int] = (0, 0, 0, 0)
    
    def __post_init__(self):
        # Check that all compute unit widths don't exceed register width
        if self.reduce_compute_unit_width > self.register_width:
            raise ValueError("Reduce compute unit width cannot exceed register width")
        if self.simple_elementwise_compute_unit_width > self.register_width:
            raise ValueError("Simple elementwise compute unit width cannot exceed register width")
        if self.complex_elementwise_compute_unit_width > self.register_width:
            raise ValueError("Complex elementwise compute unit width cannot exceed register width")
        if self.issue_queue_window < 1:
            raise ValueError("Issue queue window must be at least 1")
        if self.ooo_scheduler_window_size < 1:
            raise ValueError("Out-of-order scheduler window size must be at least 1")
        
        valid_reg_widths = [512, 1024, 2048, 4096]
        valid_compute_widths = [128, 256, 512, 1024]
        valid_cache_bw = [32, 64, 128]
        valid_chain_gran = [32, 64, 128, 256]
        
        if self.register_width not in valid_reg_widths:
            raise ValueError(f"Invalid register width: {self.register_width}")
        if self.reduce_compute_unit_width not in valid_compute_widths:
            raise ValueError(f"Invalid reduce compute unit width: {self.reduce_compute_unit_width}")
        if self.simple_elementwise_compute_unit_width not in valid_compute_widths:
            raise ValueError(f"Invalid simple elementwise compute unit width: {self.simple_elementwise_compute_unit_width}")
        if self.complex_elementwise_compute_unit_width not in valid_compute_widths:
            raise ValueError(f"Invalid complex elementwise compute unit width: {self.complex_elementwise_compute_unit_width}")
        if self.cache_bandwidth not in valid_cache_bw:
            raise ValueError(f"Invalid cache bandwidth: {self.cache_bandwidth}")
        if self.chaining_granularity not in valid_chain_gran:
            raise ValueError(f"Invalid chaining granularity: {self.chaining_granularity}")
        if self.num_contexts < 1 or self.num_contexts > 8:
            raise ValueError("num_contexts must be between 1 and 8")
        if len(self.lane_strides) != 4:
            raise ValueError("lane_strides must contain exactly 4 entries")
        if any(stride < 0 for stride in self.lane_strides):
            raise ValueError("lane_strides entries must be non-negative byte counts")


@dataclass
class Instruction:
    """Represents a single instruction in the instruction stream"""
    id: int
    type: InstructionType
    dependencies: Set[int]  # IDs of instructions this depends on
    data_size: int  # Size of data to process (in bytes)
    target_register: Optional[int] = None
    source_registers: Tuple[int, ...] = ()
    logical_target_register: Optional[int] = None
    logical_source_registers: Tuple[int, ...] = ()

    # Chaining support
    element_wise_src: bool = False
    element_wise_dest: bool = False

    # Context (register group) this instruction belongs to
    context_id: int = 0

    # RVV multi-lane load/store stride context (vlane0-vlane3)
    vlane_ctx: int = 0

    # Execution state
    issued: bool = False
    started: bool = False
    completed: bool = False
    issue_cycle: int = -1  # When instruction was issued
    start_cycle: int = -1  # When execution started
    complete_cycle: int = -1


# Instruction wrapper classes for simplified instruction creation
class LoadInstruction:
    """Simplified wrapper for creating LOAD instructions"""
    
    def __init__(self, id: int, target_register: int, dependencies: Set[int] = None,
                 data_size: int = 256, vlane_ctx: int = 0,
                 logical_target_register: int = None):
        if logical_target_register is None:
            logical_target_register = target_register
        self.instruction = Instruction(
            id=id,
            type=InstructionType.LOAD,
            dependencies=_dependency_set(dependencies),
            data_size=data_size,  # Default 2048 bits = 256 bytes
            target_register=target_register,
            source_registers=(),
            logical_target_register=logical_target_register,
            logical_source_registers=(),
            element_wise_dest=True,
            vlane_ctx=vlane_ctx,
        )
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


class ReduceInstruction:
    """Simplified wrapper for creating REDUCE instructions"""
    
    def __init__(self, id: int, target_register: int, source_registers: List[int], 
                 dependencies: Set[int] = None, data_size: int = 256,
                 logical_target_register: int = None, logical_source_registers: List[int] = None):
        # If dependencies not explicitly provided, derive from source_registers
        if dependencies is None:
            dependencies = set(source_registers) if source_registers else set()
        else:
            dependencies = _dependency_set(dependencies)
        if logical_target_register is None:
            logical_target_register = target_register
        if logical_source_registers is None:
            logical_source_registers = source_registers
        
        self.instruction = Instruction(
            id=id,
            type=InstructionType.REDUCE,
            dependencies=dependencies,
            data_size=data_size,  # Default 2048 bits = 256 bytes
            target_register=target_register,
            source_registers=tuple(source_registers or ()),
            logical_target_register=logical_target_register,
            logical_source_registers=tuple(logical_source_registers or ()),
            element_wise_src=True,
        )
        # Reduce instruction specific
        self.first_level_uop_count: int = 0  # Number of first-level uops for reduce instructions
        
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


class FMAInstruction:
    """Simplified wrapper for creating FMA instructions"""
    
    def __init__(self, id: int, target_register: int, source_registers: List[int],
                 dependencies: Set[int] = None, data_size: int = 256,
                 logical_target_register: int = None, logical_source_registers: List[int] = None):
        # If dependencies not explicitly provided, derive from source_registers
        if dependencies is None:
            dependencies = set(source_registers) if source_registers else set()
        else:
            dependencies = _dependency_set(dependencies)
        if logical_target_register is None:
            logical_target_register = target_register
        if logical_source_registers is None:
            logical_source_registers = source_registers
        
        self.instruction = Instruction(
            id=id,
            type=InstructionType.FMA,
            dependencies=dependencies,
            data_size=data_size,  # Default 2048 bits = 256 bytes
            target_register=target_register,
            source_registers=tuple(source_registers or ()),
            logical_target_register=logical_target_register,
            logical_source_registers=tuple(logical_source_registers or ()),
            element_wise_src=True,
            element_wise_dest=True,
        )
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


class EXP2Instruction:
    """Simplified wrapper for creating EXP2 instructions"""
    
    def __init__(self, id: int, target_register: int, source_registers: List[int],
                 dependencies: Set[int] = None, data_size: int = 256,
                 logical_target_register: int = None, logical_source_registers: List[int] = None):
        # If dependencies not explicitly provided, derive from source_registers
        if dependencies is None:
            dependencies = set(source_registers) if source_registers else set()
        else:
            dependencies = _dependency_set(dependencies)
        if logical_target_register is None:
            logical_target_register = target_register
        if logical_source_registers is None:
            logical_source_registers = source_registers
        
        self.instruction = Instruction(
            id=id,
            type=InstructionType.EXP2,
            dependencies=dependencies,
            data_size=data_size,  # Default 2048 bits = 256 bytes
            target_register=target_register,
            source_registers=tuple(source_registers or ()),
            logical_target_register=logical_target_register,
            logical_source_registers=tuple(logical_source_registers or ()),
            element_wise_src=True,
            element_wise_dest=True,
        )
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


class StoreInstruction:
    """Simplified wrapper for creating STORE instructions"""
    
    def __init__(self, id: int, source_registers: List[int], 
                 dependencies: Set[int] = None, target_mem: int = None, data_size: int = 256,
                 vlane_ctx: int = 0, logical_source_registers: List[int] = None):
        # If dependencies not explicitly provided, derive from source_registers
        if dependencies is None:
            dependencies = set(source_registers) if source_registers else set()
        else:
            dependencies = _dependency_set(dependencies)
        if logical_source_registers is None:
            logical_source_registers = source_registers
        
        self.instruction = Instruction(
            id=id,
            type=InstructionType.STORE,
            dependencies=dependencies,
            data_size=data_size,  # Default 2048 bits = 256 bytes
            target_register=target_mem,
            source_registers=tuple(source_registers or ()),
            logical_target_register=None,
            logical_source_registers=tuple(logical_source_registers or ()),
            element_wise_src=True,
            vlane_ctx=vlane_ctx,
        )
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


@dataclass
class MicroOp:
    """Represents a micro-operation (uop) - a unit of work that can be executed"""
    instruction_id: int
    uop_id: int
    type: InstructionType
    data_size: int  # Size of data this uop processes
    dependencies: Set[int]  # Other uops this depends on
    latency: int

    # Context this uop belongs to
    context_id: int = 0

    # RVV multi-lane load/store stride context and modeled logical address
    vlane_ctx: int = 0
    address: Optional[int] = None

    # Execution state
    issued: bool = False
    started: bool = False
    completed: bool = False
    start_cycle: int = -1
    complete_cycle: int = -1
    
    # Chaining support - tracks how many elements are ready
    ready_elements: int = 0


class InstructionExecutor:
    """Handles the execution logic for different instruction types"""
    
    def __init__(self, config: ProcessorConfig, quiet: bool = False):
        self.config = config
        self.quiet = quiet
    
    def split_instruction_to_uops(self, instruction: Instruction) -> List[MicroOp]:
        """Split an instruction into micro-operations based on processor configuration"""
        uops = []

        if instruction.type == InstructionType.REDUCE:
            uops = self._split_reduce_instruction(instruction)
        elif instruction.type in [InstructionType.FMA, InstructionType.EXP2]:
            uops = self._split_arithmetic_instruction(instruction)
        elif instruction.type in [InstructionType.LOAD, InstructionType.STORE]:
            uops = self._split_memory_instruction(instruction)

        for uop in uops:
            uop.context_id = instruction.context_id
            uop.vlane_ctx = instruction.vlane_ctx

        return uops
    
    def _split_reduce_instruction(self, instruction: Instruction) -> List[MicroOp]:
        """Split reduce instruction into uops with tree reduction logic"""
        # Max elements per instruction: M = rl/16 (bf16 = 2 bytes, so 16 bits)
        max_elements = self.config.register_width // 16
        # Elements per cycle: N = vl/16  
        elements_per_cycle = self.config.reduce_compute_unit_width // 16
        
        # Actual elements to process
        actual_elements = min(max_elements, instruction.data_size // 2)  # bf16 = 2 bytes
        
        uops = []
        uop_id = 0
        
        # Phase 1: Parallel reduction of groups
        if actual_elements <= elements_per_cycle:
            # Single uop can handle all elements
            uop = MicroOp(
                instruction_id=instruction.id,
                uop_id=uop_id,
                type=InstructionType.REDUCE,
                data_size=actual_elements * 2,  # bf16 elements
                dependencies=set(),
                latency=self.config.reduce_latency
            )
            uops.append(uop)
            instruction.first_level_uop_count = 1
        else:
            # Multiple uops needed for first phase
            first_phase_uops = math.ceil(actual_elements / elements_per_cycle)
            
            for i in range(first_phase_uops):
                elements_in_uop = min(elements_per_cycle, 
                                    actual_elements - i * elements_per_cycle)
                uop = MicroOp(
                    instruction_id=instruction.id,
                    uop_id=uop_id,
                    type=InstructionType.REDUCE,
                    data_size=elements_in_uop * 2,
                    dependencies=set(),
                    latency=self.config.reduce_latency
                )
                uops.append(uop)
                uop_id += 1
            instruction.first_level_uop_count = first_phase_uops
            
            # Phase 2: Reduce the results from phase 1
            if first_phase_uops > 1:
                # Create dependency on all first phase uops
                first_phase_deps = set(range(len(uops)))
                
                # Calculate how many reduction levels are needed
                remaining_elements = first_phase_uops
                while remaining_elements > 1:
                    next_level_uops = math.ceil(remaining_elements / elements_per_cycle)
                    level_deps = first_phase_deps.copy()
                    
                    for i in range(next_level_uops):
                        uop = MicroOp(
                            instruction_id=instruction.id,
                            uop_id=uop_id,
                            type=InstructionType.REDUCE,
                            data_size=min(elements_per_cycle, remaining_elements) * 2,
                            dependencies=level_deps,
                            latency=self.config.reduce_latency
                        )
                        uops.append(uop)
                        uop_id += 1
                    
                    remaining_elements = next_level_uops
                    first_phase_deps = set(range(len(uops) - next_level_uops, len(uops)))
        
        return uops
    
    def _split_arithmetic_instruction(self, instruction: Instruction) -> List[MicroOp]:
        """Split FMA or EXP2 instruction into uops"""
        # Max elements per instruction: rl/16 (bf16)
        max_elements = self.config.register_width // 16
        # Elements per cycle: vl/16 - use appropriate compute unit width based on instruction type
        if instruction.type == InstructionType.FMA:
            elements_per_cycle = self.config.simple_elementwise_compute_unit_width // 16
        elif instruction.type == InstructionType.EXP2:
            elements_per_cycle = self.config.complex_elementwise_compute_unit_width // 16
        else:
            raise ValueError(f"Invalid arithmetic instruction type: {instruction.type}")
        
        actual_elements = min(max_elements, instruction.data_size // 2)  # bf16 = 2 bytes
        
        uops = []
        remaining_elements = actual_elements
        uop_id = 0
        
        while remaining_elements > 0:
            elements_in_uop = min(elements_per_cycle, remaining_elements)
            
            latency = (self.config.fma_latency if instruction.type == InstructionType.FMA 
                      else self.config.exp2_latency)
            
            uop = MicroOp(
                instruction_id=instruction.id,
                uop_id=uop_id,
                type=instruction.type,
                data_size=elements_in_uop * 2,
                dependencies=set(),
                latency=latency
            )
            uops.append(uop)
            
            remaining_elements -= elements_in_uop
            uop_id += 1
        
        return uops
    
    def _split_memory_instruction(self, instruction: Instruction) -> List[MicroOp]:
        """Split load or store instruction into uops"""
        # Max bytes per instruction: vl/8
        max_bytes_per_instruction = self.config.register_width // 8
        # Bytes per cycle limited by cache bandwidth
        bytes_per_cycle = self.config.cache_bandwidth
        actual_bytes = instruction.data_size // 8

        if not self.quiet:
            print(f"Max bytes per instruction: {max_bytes_per_instruction}")
            print(f"Bytes per cycle: {bytes_per_cycle}")
            print(f"Instruction data bytes: {actual_bytes}")

        assert actual_bytes <= max_bytes_per_instruction
        if instruction.vlane_ctx < 0 or instruction.vlane_ctx >= len(self.config.lane_strides):
            raise ValueError(f"Invalid vlane_ctx: {instruction.vlane_ctx}")
        
        uops = []
        remaining_bytes = actual_bytes
        uop_id = 0
        chunk_offset = 0
        lane_stride = self.config.lane_strides[instruction.vlane_ctx]
        
        while remaining_bytes > 0:
            bytes_in_uop = min(bytes_per_cycle, remaining_bytes)
            
            latency = (self.config.load_latency if instruction.type == InstructionType.LOAD
                      else self.config.store_latency)
            
            # Create dependencies: each uop depends on the previous one (except the first)
            dependencies = set()
            
            uop = MicroOp(
                instruction_id=instruction.id,
                uop_id=uop_id,
                type=instruction.type,
                data_size=bytes_in_uop,
                dependencies=dependencies,
                latency=latency,
                vlane_ctx=instruction.vlane_ctx,
                address=chunk_offset + instruction.context_id * lane_stride,
            )
            # print(f"uop {uop_id} dependencies: {dependencies}")
            uops.append(uop)
            
            remaining_bytes -= bytes_in_uop  
            chunk_offset += bytes_in_uop
            uop_id += 1
        
        return uops


class VectorProcessor:
    """Main vector processor simulator"""
    
    def __init__(self, config: ProcessorConfig, quiet: bool = False):
        self.config = config
        self.quiet = quiet
        self.executor = InstructionExecutor(config, quiet=quiet)
        
        # Simulation state
        self.current_cycle = 0
        self.instructions: Dict[int, Instruction] = {}
        self.uops: List[MicroOp] = []
        self.instruction_uop_map: Dict[int, List[int]] = {}  # instruction_id -> uop_ids
        self.context_uop_indices: Dict[int, List[int]] = {}
        self.context_issue_cursors: Dict[int, int] = {}
        self.instruction_remaining_uops: Dict[int, int] = {}
        self.remaining_instructions = 0
        
        # Execution units (simplified model)
        self.execution_units = {
            InstructionType.REDUCE: [],
            InstructionType.FMA: [],
            InstructionType.EXP2: [],
            InstructionType.LOAD: [],
            InstructionType.STORE: []
        }
        
        # Cache bandwidth tracking
        self.cache_bandwidth_used = 0
        
        # Separate arithmetic bandwidth tracking for each instruction type (in bytes)
        self.reduce_bandwidth_used = 0
        self.simple_elementwise_bandwidth_used = 0
        self.complex_elementwise_bandwidth_used = 0
        
        # Outstanding (issued but not completed) counts tracking
        self.outstanding_instruction_count = 0
        self.outstanding_uop_count = 0

        # Per-cycle utilization tracking
        self.per_cycle_stats = []
        self._cycle_issued_count = 0
    
    def load_instructions(self, instructions: List[Instruction]):
        """Load instruction stream into the processor"""
        self.instructions = {inst.id: inst for inst in instructions}
        self.uops = []
        self.instruction_uop_map = {}
        self.context_uop_indices = {ctx: [] for ctx in range(self.config.num_contexts)}
        self.remaining_instructions = len(self.instructions)
        
        # Split all instructions into uops
        uop_offset = 0
        for instruction in self.instructions.values():
            instruction_uops = self.executor.split_instruction_to_uops(instruction)
            self.uops.extend(instruction_uops)
            
            uop_ids = list(range(uop_offset, uop_offset + len(instruction_uops)))
            self.instruction_uop_map[instruction.id] = uop_ids
            uop_offset += len(instruction_uops)
        
        # Fix reduce instruction dependencies
        self._fix_reduce_dependencies()
        
        # Fix memory instruction dependencies
        self._fix_memory_dependencies()

        # Model false register dependencies when physical register renaming is absent.
        if not self.config.register_renaming:
            self._add_register_hazard_dependencies()
        
        # Establish chaining dependencies if enabled
        if self.config.chaining_enabled:
            self._establish_chaining_dependencies()

        for idx, uop in enumerate(self.uops):
            self.context_uop_indices.setdefault(uop.context_id, []).append(idx)

    def _add_register_hazard_dependencies(self):
        """Add WAR/WAW/RAW ordering constraints without register renaming."""
        last_writer = {}
        last_readers = {}

        for instruction in self.instructions.values():
            ctx = instruction.context_id
            source_regs = tuple(instruction.logical_source_registers)
            target_reg = instruction.logical_target_register

            for reg in source_regs:
                writer = last_writer.get((ctx, reg))
                if writer is not None:
                    instruction.dependencies.add(writer)
                last_readers.setdefault((ctx, reg), set()).add(instruction.id)

            if target_reg is not None:
                reg_key = (ctx, target_reg)
                writer = last_writer.get(reg_key)
                if writer is not None:
                    instruction.dependencies.add(writer)
                instruction.dependencies.update(last_readers.get(reg_key, set()))
                last_writer[reg_key] = instruction.id
                last_readers[reg_key] = set()
    
    def _fix_reduce_dependencies(self):
        """Fix dependencies for reduce instruction uops after all uops are loaded"""
        for instruction in self.instructions.values():
            if instruction.type == InstructionType.REDUCE:
                uop_ids = self.instruction_uop_map[instruction.id]
                if len(uop_ids) > 1:  # Multiple uops for this reduce instruction
                    # Find the boundary between first phase and subsequent phases
                    # First, calculate expected first phase uops
                    max_elements = self.config.register_width // 16
                    elements_per_cycle = self.config.reduce_compute_unit_width // 16
                    actual_elements = min(max_elements, instruction.data_size // 2)
                    
                    if actual_elements > elements_per_cycle:
                        first_phase_uops = math.ceil(actual_elements / elements_per_cycle)
                        
                        # Clear existing dependencies for all uops of this instruction
                        for uop_id in uop_ids:
                            self.uops[uop_id].dependencies.clear()
                        
                        # Set correct dependencies: second phase uops depend on all first phase uops
                        if len(uop_ids) > first_phase_uops:
                            first_phase_global_ids = set(uop_ids[:first_phase_uops])
                            for i in range(first_phase_uops, len(uop_ids)):
                                uop_global_id = uop_ids[i]
                                self.uops[uop_global_id].dependencies = first_phase_global_ids.copy()
    
    def _fix_memory_dependencies(self):
        """Fix dependencies for memory instruction uops after all uops are loaded"""
        for instruction in self.instructions.values():
            if instruction.type in [InstructionType.LOAD, InstructionType.STORE]:
                uop_ids = self.instruction_uop_map[instruction.id]
                
                # Fix dependencies: convert local IDs to global IDs
                for i, global_uop_id in enumerate(uop_ids):
                    uop = self.uops[global_uop_id]
                    new_dependencies = set()
                    
                    for local_dep_id in uop.dependencies:
                        # Convert local dependency ID to global ID
                        global_dep_id = uop_ids[local_dep_id]
                        new_dependencies.add(global_dep_id)
                    
                    uop.dependencies = new_dependencies
    
    def _establish_chaining_dependencies(self):
        """Establish chaining dependencies between producer and consumer instructions"""
        for producer_inst in self.instructions.values():
            # Skip if producer doesn't have element-wise destination
            if not producer_inst.element_wise_dest:
                continue
            
            # Find consumer instructions that depend on this producer
            for consumer_inst in self.instructions.values():
                # Skip if consumer doesn't have element-wise source
                if not consumer_inst.element_wise_src:
                    continue
                
                # Skip if consumer doesn't depend on producer
                if producer_inst.id not in consumer_inst.dependencies:
                    continue
                
                # Get uops for both instructions
                producer_uop_ids = self.instruction_uop_map[producer_inst.id]
                consumer_uop_ids = self.instruction_uop_map[consumer_inst.id]
                
                # Assert that data sizes match
                assert producer_inst.data_size == consumer_inst.data_size, \
                    f"Chaining requires matching data sizes: producer {producer_inst.id} " \
                    f"({producer_inst.data_size}) vs consumer {consumer_inst.id} " \
                    f"({consumer_inst.data_size})"
                
                # Assert that number of uops match
                if consumer_inst.type == InstructionType.REDUCE:
                    assert len(producer_uop_ids) == consumer_inst.first_level_uop_count, \
                        f"Chaining requires matching uop counts: producer {producer_inst.id} " \
                        f"({len(producer_uop_ids)} uops) vs consumer {consumer_inst.id} " \
                        f"({consumer_inst.first_level_uop_count} uops)"
                else:
                    assert len(producer_uop_ids) == len(consumer_uop_ids), \
                        f"Chaining requires matching uop counts: producer {producer_inst.id} " \
                        f"({len(producer_uop_ids)} uops) vs consumer {consumer_inst.id} " \
                        f"({len(consumer_uop_ids)} uops)"
                
                # Assert for load instructions - they should match compute instruction uop count
                if producer_inst.type == InstructionType.LOAD:
                    # Find the next compute instruction that depends on this load
                    for next_inst in self.instructions.values():
                        if (producer_inst.id in next_inst.dependencies and 
                            next_inst.type in [InstructionType.FMA, InstructionType.EXP2]):
                            next_uop_count = len(self.instruction_uop_map[next_inst.id])
                            assert len(producer_uop_ids) == next_uop_count, \
                                f"Load instruction {producer_inst.id} uop count ({len(producer_uop_ids)}) " \
                                f"must match compute instruction {next_inst.id} uop count ({next_uop_count})"
                            break
                
                # Establish one-to-one chaining dependencies
                if not self.quiet:
                    print(f"Establishing chaining between instruction {producer_inst.id} -> {consumer_inst.id}")
                for i, (prod_uop_id, cons_uop_id) in enumerate(zip(producer_uop_ids, consumer_uop_ids)):
                    # Consumer uop depends on corresponding producer uop completion
                    self.uops[cons_uop_id].dependencies.add(prod_uop_id)
                    if not self.quiet:
                        print(f"  uop {consumer_inst.id}.{i} now depends on uop {producer_inst.id}.{i}")

                # Remove instruction-level dependency since we now have uop-level dependencies
                consumer_inst.dependencies.discard(producer_inst.id)
                if not self.quiet:
                    print(f"  Removed instruction-level dependency {producer_inst.id} -> {consumer_inst.id}")
    
    def simulate(self, max_cycles: int = 10000) -> Dict:
        """Run the simulation and return results"""
        self.current_cycle = 0
        
        # Reset all state
        for instruction in self.instructions.values():
            instruction.issued = instruction.started = instruction.completed = False
            instruction.issue_cycle = instruction.start_cycle = instruction.complete_cycle = -1
        
        for uop in self.uops:
            uop.issued = uop.started = uop.completed = False
            uop.start_cycle = uop.complete_cycle = -1
            uop.ready_elements = 0

        self.context_issue_cursors = {ctx: 0 for ctx in range(self.config.num_contexts)}
        self.instruction_remaining_uops = {
            inst_id: len(uop_ids)
            for inst_id, uop_ids in self.instruction_uop_map.items()
        }
        self.remaining_instructions = len(self.instructions)
        
        # Reset outstanding counts
        self.outstanding_instruction_count = 0
        self.outstanding_uop_count = 0

        # Reset utilization stats
        self.per_cycle_stats = []
        
        while not self._all_instructions_completed() and self.current_cycle < max_cycles:
            self._simulate_cycle()
            self.current_cycle += 1
        
        if self.current_cycle >= max_cycles and not self.quiet:
            print(f"Warning: Simulation reached maximum cycles ({max_cycles})")
            print(f"Completed instructions: {sum(1 for inst in self.instructions.values() if inst.completed)}/{len(self.instructions)}")
            print(f"Completed uops: {sum(1 for uop in self.uops if uop.completed)}/{len(self.uops)}")
        
        return self._generate_results()
    
    def _simulate_cycle(self):
        """Simulate a single cycle"""
        # Reset per-cycle state
        self.cache_bandwidth_used = 0
        self.reduce_bandwidth_used = 0
        self.simple_elementwise_bandwidth_used = 0
        self.complex_elementwise_bandwidth_used = 0
        self._cycle_issued_count = 0
        
        # Update execution units and complete uops
        self._update_execution_units()
        
        # Issue new uops based on execution mode
        if self.config.execution_mode == ExecutionMode.IN_ORDER:
            self._issue_in_order()
        else:
            self._issue_out_of_order()
        
        # Handle chaining if enabled
        if self.config.chaining_enabled:
            self._handle_chaining()

        # Record per-cycle utilization stats
        active_by_type = {}
        for inst_type in self.execution_units:
            active_by_type[inst_type] = len(self.execution_units[inst_type])
        self.per_cycle_stats.append({
            'issued_count': self._cycle_issued_count,
            'active_by_type': active_by_type,
            'cache_bw_used': self.cache_bandwidth_used,
            'reduce_bw_used': self.reduce_bandwidth_used,
            'simple_ew_bw_used': self.simple_elementwise_bandwidth_used,
            'complex_ew_bw_used': self.complex_elementwise_bandwidth_used,
        })
    
    def _update_execution_units(self):
        """Update execution units and complete finished uops"""
        for inst_type in self.execution_units:
            completed_uops = []
            
            for uop_info in self.execution_units[inst_type]:
                uop_id, complete_cycle = uop_info
                if self.current_cycle >= complete_cycle:
                    uop = self.uops[uop_id]
                    uop.completed = True
                    uop.complete_cycle = self.current_cycle
                    completed_uops.append(uop_info)
                    
                    # Update outstanding uop count
                    self.outstanding_uop_count -= 1

                    self._complete_uop(uop)
            
            # Remove completed uops
            for uop_info in completed_uops:
                self.execution_units[inst_type].remove(uop_info)

    def _complete_uop(self, uop: MicroOp):
        remaining = self.instruction_remaining_uops.get(uop.instruction_id, 0) - 1
        self.instruction_remaining_uops[uop.instruction_id] = remaining
        if remaining != 0:
            return

        instruction = self.instructions[uop.instruction_id]
        if instruction.completed:
            return

        uop_ids = self.instruction_uop_map[instruction.id]
        instruction.completed = True
        instruction.complete_cycle = max(self.uops[uop_id].complete_cycle for uop_id in uop_ids)
        if instruction.start_cycle == -1:
            instruction.start_cycle = min(self.uops[uop_id].start_cycle for uop_id in uop_ids)
        self.outstanding_instruction_count -= 1
        self.remaining_instructions -= 1
    
    def _issue_queue_window(self, ctx: int) -> List[int]:
        """Return the oldest not-yet-completed uops visible to issue."""
        window = []
        ctx_indices = self.context_uop_indices.get(ctx, [])
        cursor = self.context_issue_cursors.get(ctx, 0)

        while cursor < len(ctx_indices) and self.uops[ctx_indices[cursor]].completed:
            cursor += 1
        self.context_issue_cursors[ctx] = cursor

        for idx in ctx_indices[cursor:]:
            uop = self.uops[idx]
            if uop.completed:
                continue
            window.append(idx)
            if len(window) >= self.config.issue_queue_window:
                break
        return window

    def _issue_from_candidates(self, candidate_indices: List[int], max_to_issue: int) -> int:
        issued = 0
        for idx in candidate_indices:
            if issued >= max_to_issue:
                break
            uop = self.uops[idx]
            if not uop.issued and self._can_issue_uop(uop):
                if self._issue_uop(idx):
                    issued += 1
        return issued

    def _scheduler_candidates(self, ctx: int, scheduler_window_size: int) -> List[int]:
        """Return the oldest not-yet-issued uops inside the visible issue queue."""
        candidates = []
        for idx in self._issue_queue_window(ctx):
            if self.uops[idx].issued:
                continue
            candidates.append(idx)
            if len(candidates) >= scheduler_window_size:
                break
        return candidates

    def _issue_with_scheduler_window(self, scheduler_window_size: int):
        """Issue ready uops from scheduler candidates, sharing issue width globally."""
        total_issued = 0
        for ctx in range(self.config.num_contexts):
            candidate_indices = self._scheduler_candidates(ctx, scheduler_window_size)
            remaining_issue_slots = self.config.issue_width - total_issued
            total_issued += self._issue_from_candidates(candidate_indices, remaining_issue_slots)
            if total_issued >= self.config.issue_width:
                return

    def _issue_in_order(self):
        """Issue only the oldest not-yet-issued uop in the visible issue queue."""
        self._issue_with_scheduler_window(1)

    def _issue_out_of_order(self):
        """Issue ready uops from the oldest scheduler candidates in the issue queue."""
        self._issue_with_scheduler_window(self.config.ooo_scheduler_window_size)
    
    def _can_issue_uop(self, uop: MicroOp) -> bool:
        """Check if a uop can be issued"""
        # Check dependencies
        for dep_uop_id in uop.dependencies:
            if not self.uops[dep_uop_id].completed:
                return False
        
        # Check instruction dependencies
        instruction = self.instructions[uop.instruction_id]
        for dep_inst_id in instruction.dependencies:
            if not self.instructions[dep_inst_id].completed:
                return False
        
        return True
    
    def _issue_uop(self, uop_index: int) -> bool:
        """Try to issue a uop to an execution unit"""
        uop = self.uops[uop_index]
        # Check resource availability for memory operations
        if uop.type in [InstructionType.LOAD, InstructionType.STORE]:
            # Each memory uop size is typically equal to cache_bandwidth (except possibly the last one)
            # This ensures at most one memory uop can be issued per cycle due to bandwidth constraints
            if self.cache_bandwidth_used + uop.data_size > self.config.cache_bandwidth:
                return False
            self.cache_bandwidth_used += uop.data_size
        
        # Check resource availability for arithmetic operations with separate bandwidth tracking
        if uop.type == InstructionType.REDUCE:
            # Check if issuing this uop would exceed the reduce compute unit width
            reduce_compute_unit_bytes = self.config.reduce_compute_unit_width // 8
            if self.reduce_bandwidth_used + uop.data_size > reduce_compute_unit_bytes:
                return False
            self.reduce_bandwidth_used += uop.data_size
        elif uop.type == InstructionType.FMA:
            # Check if issuing this uop would exceed the simple elementwise compute unit width
            simple_compute_unit_bytes = self.config.simple_elementwise_compute_unit_width // 8
            if self.simple_elementwise_bandwidth_used + uop.data_size > simple_compute_unit_bytes:
                return False
            self.simple_elementwise_bandwidth_used += uop.data_size
        elif uop.type == InstructionType.EXP2:
            # Check if issuing this uop would exceed the complex elementwise compute unit width
            complex_compute_unit_bytes = self.config.complex_elementwise_compute_unit_width // 8
            if self.complex_elementwise_bandwidth_used + uop.data_size > complex_compute_unit_bytes:
                return False
            self.complex_elementwise_bandwidth_used += uop.data_size
        
        # Issue the uop
        uop.issued = True
        uop.started = True  
        uop.start_cycle = self.current_cycle
        complete_cycle = self.current_cycle + uop.latency
        
        self.execution_units[uop.type].append((uop_index, complete_cycle))

        instruction = self.instructions[uop.instruction_id]
        if instruction.issue_cycle == -1:
            instruction.issue_cycle = self.current_cycle
            instruction.issued = True
            self.outstanding_instruction_count += 1
        
        # Update outstanding uop count
        self.outstanding_uop_count += 1

        # Track issued count for utilization
        self._cycle_issued_count += 1

        return True
    
    def _handle_chaining(self):
        """Handle chaining between instructions"""
        if not self.config.chaining_enabled:
            return
            
        # Calculate how many elements are ready for each running uop
        for uop in self.uops:
            if uop.started and not uop.completed:
                # Calculate how many elements are ready based on progress
                cycles_elapsed = self.current_cycle - uop.start_cycle
                progress = min(1.0, cycles_elapsed / uop.latency)
                elements_total = uop.data_size // 2  # bf16 elements (2 bytes each)
                uop.ready_elements = int(progress * elements_total)
                
                # For demonstration, print chaining progress
                if cycles_elapsed == 1 and not self.quiet:  # Only print once per uop
                    producer_inst = self.instructions[uop.instruction_id]
                    if producer_inst.element_wise_dest:
                        print(f"Chaining: uop {uop.instruction_id}.{uop.uop_id} has "
                              f"{uop.ready_elements}/{elements_total} elements ready")
        
        # Note: The actual dependency checking for chaining is handled in _can_issue_uop
        # based on the uop-level dependencies we established in _establish_chaining_dependencies
    
    def _all_instructions_completed(self) -> bool:
        """Check if all instructions have completed"""
        return self.remaining_instructions == 0
    
    def _generate_results(self) -> Dict:
        """Generate simulation results"""
        # Instruction completion is now handled in _all_instructions_completed()
        
        return {
            'total_cycles': self.current_cycle,
            'instructions': [
                {
                    'id': inst.id,
                    'type': inst.type.value,
                    'issue_cycle': inst.issue_cycle,
                    'start_cycle': inst.start_cycle,
                    'complete_cycle': inst.complete_cycle,
                    'execution_time': inst.complete_cycle - inst.start_cycle if inst.start_cycle >= 0 else -1
                }
                for inst in self.instructions.values()
            ],
            'uops': [
                {
                    'instruction_id': uop.instruction_id,
                    'uop_id': uop.uop_id,
                    'type': uop.type.value,
                    'context_id': uop.context_id,
                    'vlane_ctx': uop.vlane_ctx,
                    'address': uop.address,
                    'start_cycle': uop.start_cycle,
                    'complete_cycle': uop.complete_cycle,
                    'execution_time': uop.complete_cycle - uop.start_cycle if uop.start_cycle >= 0 else -1
                }
                for uop in self.uops
            ]
        }

    def get_utilization_metrics(self) -> Dict:
        """Return compact utilization statistics for reports and benchmarks."""
        total_cycles = len(self.per_cycle_stats)
        if total_cycles == 0:
            return {
                'total_cycles': 0,
                'issue_slot_utilization': 0.0,
                'average_issued': 0.0,
                'unit_active_pct': {inst_type.value: 0.0 for inst_type in self.execution_units}
            }

        total_issued = sum(s['issued_count'] for s in self.per_cycle_stats)
        max_possible = total_cycles * self.config.issue_width
        unit_active_pct = {}
        for inst_type in self.execution_units:
            active_cycles = sum(
                1 for s in self.per_cycle_stats
                if s['active_by_type'].get(inst_type, 0) > 0
            )
            unit_active_pct[inst_type.value] = active_cycles / total_cycles * 100

        return {
            'total_cycles': total_cycles,
            'issue_slot_utilization': total_issued / max_possible * 100 if max_possible else 0.0,
            'average_issued': total_issued / total_cycles,
            'unit_active_pct': unit_active_pct,
        }
    
    def get_outstanding_instruction_count(self) -> int:
        """Get the current number of outstanding instructions (issued but not completed)"""
        return self.outstanding_instruction_count
    
    def get_outstanding_uop_count(self) -> int:
        """Get the current number of outstanding uops (issued but not completed)"""
        return self.outstanding_uop_count
    
    def visualize_execution(self):
        """Generate ASCII visualization of instruction execution timeline"""
        if not self.instructions:
            print("No instructions to visualize")
            return
        
        total_cycles = self.current_cycle
        if total_cycles == 0:
            print("No execution to visualize (0 cycles)")
            return
            
        print("ASCII Execution Timeline:")
        print("@ = Issue, - = Execution, ! = Complete")
        print()
        
        # Print cycle numbers header
        cycle_header = "Instruction".ljust(20) + " " + "".join(f"{i % 10}" for i in range(total_cycles))
        print(cycle_header)
        print("-" * len(cycle_header))
        
        # Generate timeline for each instruction
        for inst in self.instructions.values():
            timeline = [' '] * total_cycles
            
            # Mark issue cycle with @
            if inst.issue_cycle >= 0 and inst.issue_cycle < total_cycles:
                timeline[inst.issue_cycle] = '@'
            
            # Mark execution cycles with -
            if inst.start_cycle >= 0 and inst.complete_cycle >= 0:
                for cycle in range(max(inst.start_cycle, 0), 
                                 min(inst.complete_cycle, total_cycles)):
                    if timeline[cycle] == ' ':  # Don't overwrite issue marker
                        timeline[cycle] = '-'
            
            # Mark complete cycle with !
            if inst.complete_cycle >= 0 and inst.complete_cycle < total_cycles:
                timeline[inst.complete_cycle] = '!'
            
            # Create instruction label
            inst_label = f"Inst{inst.id} ({inst.type.value})".ljust(20)
            timeline_str = "".join(timeline)
            
            print(f"{inst_label} {timeline_str}")
        
        print()
        print(f"Total execution time: {total_cycles} cycles")
    
    def visualize_uop_execution(self):
        """Generate ASCII visualization of uop execution timeline"""
        if not self.uops:
            print("No uops to visualize")
            return
        
        total_cycles = self.current_cycle
        if total_cycles == 0:
            print("No uop execution to visualize (0 cycles)")
            return
            
        print("ASCII uop Execution Timeline:")
        print("@ = Start, - = Execution, ! = Complete")
        print()
        
        # Print cycle numbers header
        cycle_header = "uop".ljust(25) + " " + "".join(f"{i % 10}" for i in range(total_cycles))
        print(cycle_header)
        print("-" * len(cycle_header))
        
        # Generate timeline for each uop
        for uop in self.uops:
            timeline = [' '] * total_cycles
            
            # Mark start cycle with @
            if uop.start_cycle >= 0 and uop.start_cycle < total_cycles:
                timeline[uop.start_cycle] = '@'
            
            # Mark execution cycles with -
            if uop.start_cycle >= 0 and uop.complete_cycle >= 0:
                for cycle in range(max(uop.start_cycle, 0), 
                                 min(uop.complete_cycle, total_cycles)):
                    if timeline[cycle] == ' ':  # Don't overwrite start marker
                        timeline[cycle] = '-'
            
            # Mark complete cycle with !
            if uop.complete_cycle >= 0 and uop.complete_cycle < total_cycles:
                timeline[uop.complete_cycle] = '!'
            
            # Create uop label with instruction_id.uop_id format
            uop_label = f"uop {uop.instruction_id}.{uop.uop_id} ({uop.type.value})".ljust(25)
            timeline_str = "".join(timeline)
            
            print(f"{uop_label} {timeline_str}")
        
        print()
        print(f"Total uop execution time: {total_cycles} cycles")

    def print_utilization(self):
        """Print utilization statistics across execution"""
        total_cycles = len(self.per_cycle_stats)
        if total_cycles == 0:
            return

        max_issue_width = self.config.issue_width
        unit_types = [InstructionType.REDUCE, InstructionType.FMA, InstructionType.EXP2,
                      InstructionType.LOAD, InstructionType.STORE]

        print("=== Utilization Statistics ===")
        print()

        # 1. Per execution unit: fraction of cycles with at least one active μop
        print("[Per Execution Unit]")
        for inst_type in unit_types:
            active_cycles = sum(1 for s in self.per_cycle_stats
                                if s['active_by_type'].get(inst_type, 0) > 0)
            pct = active_cycles / total_cycles * 100
            print(f"  {inst_type.value:8s}: {pct:5.1f}% ({active_cycles}/{total_cycles} cycles active)")
        print()

        # 2. Issue slot utilization
        print("[Issue Slot Utilization]")
        total_issued = sum(s['issued_count'] for s in self.per_cycle_stats)
        max_possible = total_cycles * max_issue_width
        cycles_with_issue = sum(1 for s in self.per_cycle_stats if s['issued_count'] > 0)
        cycles_full = sum(1 for s in self.per_cycle_stats if s['issued_count'] >= max_issue_width)
        avg_issued = total_issued / total_cycles
        print(f"  Max issue width: {max_issue_width} μop/cycle")
        print(f"  Cycles with issue: {cycles_with_issue}/{total_cycles} ({cycles_with_issue/total_cycles*100:.1f}%)")
        print(f"  Cycles at full issue: {cycles_full}/{total_cycles} ({cycles_full/total_cycles*100:.1f}%)")
        print(f"  Average issued: {avg_issued:.2f} μops/cycle")
        print(f"  Overall issue slot utilization: {total_issued/max_possible*100:.1f}%")
        print()

        # 3. Per time window utilization
        window_size = max(1, total_cycles // 10)
        print(f"[Per Time Window (window size: {window_size} cycles)]")
        print(f"  {'Window':>12s}  "
              f"{'REDUCE':>7s}  {'FMA':>7s}  {'EXP2':>7s}  {'LOAD':>7s}  {'STORE':>7s}  {'Issue':>7s}")
        for w_start in range(0, total_cycles, window_size):
            w_end = min(w_start + window_size, total_cycles)
            window = self.per_cycle_stats[w_start:w_end]
            w_len = len(window)

            active_pcts = {}
            for inst_type in unit_types:
                cycles = sum(1 for s in window if s['active_by_type'].get(inst_type, 0) > 0)
                active_pcts[inst_type] = cycles / w_len * 100

            issued_in_window = sum(s['issued_count'] for s in window)
            issue_pct = issued_in_window / (w_len * max_issue_width) * 100

            print(f"  {w_start:4d}-{w_end-1:<4d}   "
                  f"{active_pcts[InstructionType.REDUCE]:6.1f}%  "
                  f"{active_pcts[InstructionType.FMA]:6.1f}%  "
                  f"{active_pcts[InstructionType.EXP2]:6.1f}%  "
                  f"{active_pcts[InstructionType.LOAD]:6.1f}%  "
                  f"{active_pcts[InstructionType.STORE]:6.1f}%  "
                  f"{issue_pct:6.1f}%")


def create_softmax_instruction_stream(reg_width, has_exp2_unit, num_heads, seq_chunk_bit,
                                      num_contexts=1, lane_strides=None) -> List[Instruction]:
    """Create a sample instruction stream for softmax computation"""
    # Use custom data size (1024 bytes) for this example to maintain compatibility
    # with existing simulation, override the default 256 bytes

    explicit_split_count = seq_chunk_bit // reg_width
    assert explicit_split_count * reg_width == seq_chunk_bit
    data_size = reg_width
    
    # Softmax typically involves:
    # 1. Load input data
    # 2. Find maximum (reduce)
    # 3. Subtract max from all elements (FMA)
    # 4. Compute exp2 of all elements
    # 5. Sum all exp values (reduce) 
    # 6. Divide by sum (FMA)
    # 7. Store result

    all_insts = []

    for h in range(num_heads):
        head_id = h*1000
        per_head_insts = []

        def lreg(chunk: int, slot: int) -> int:
            return chunk * 100 + slot

        max_reduce_fake_dest = []
        for i in range(explicit_split_count):
            dep_group_id = head_id + i*100
            per_head_insts += [
                # Load input vector
                LoadInstruction(id=dep_group_id + 0, target_register=dep_group_id + 0,
                                data_size=reg_width, vlane_ctx=0,
                                logical_target_register=lreg(i, 0)),
                
                # Find maximum value
                ReduceInstruction(id=dep_group_id + 1, target_register=dep_group_id + 1, source_registers=[dep_group_id + 0], 
                                data_size=reg_width,
                                logical_target_register=lreg(i, 1),
                                logical_source_registers=[lreg(i, 0)]),
            ]
            max_reduce_fake_dest.append(dep_group_id + 1)


        
        if not has_exp2_unit:
            for i in range(explicit_split_count):
                dep_group_id = head_id + i*100
                per_head_insts += [
                    # Subtract max from all elements (x - max)
                    LoadInstruction(id=dep_group_id + 2, target_register=dep_group_id + 2,
                                dependencies=max_reduce_fake_dest, data_size=reg_width, vlane_ctx=0,
                                logical_target_register=lreg(i, 2)),
                    FMAInstruction(id=dep_group_id + 3, target_register=dep_group_id + 3, source_registers=[dep_group_id + 2],
                                data_size=reg_width,
                                logical_target_register=lreg(i, 3),
                                logical_source_registers=[lreg(i, 2)]),
                    FMAInstruction(id=dep_group_id + 4, target_register=dep_group_id + 4, source_registers=[dep_group_id + 3],
                                data_size=reg_width,
                                logical_target_register=lreg(i, 4),
                                logical_source_registers=[lreg(i, 3)]),
                    FMAInstruction(id=dep_group_id + 5, target_register=dep_group_id + 5, source_registers=[dep_group_id + 4],
                                data_size=reg_width,
                                logical_target_register=lreg(i, 5),
                                logical_source_registers=[lreg(i, 4)]),
                    FMAInstruction(id=dep_group_id + 6, target_register=dep_group_id + 6, source_registers=[dep_group_id + 5],
                                data_size=reg_width,
                                logical_target_register=lreg(i, 6),
                                logical_source_registers=[lreg(i, 5)]),
                    FMAInstruction(id=dep_group_id + 7, target_register=dep_group_id + 7, source_registers=[dep_group_id + 6],
                                data_size=reg_width,
                                logical_target_register=lreg(i, 7),
                                logical_source_registers=[lreg(i, 6)]),
                    FMAInstruction(id=dep_group_id + 8, target_register=dep_group_id + 8, source_registers=[dep_group_id + 7],
                                data_size=reg_width,
                                logical_target_register=lreg(i, 8),
                                logical_source_registers=[lreg(i, 7)]),
                    StoreInstruction(id=dep_group_id + 9, target_mem=dep_group_id + 9,
                                     source_registers=[dep_group_id + 8], data_size=reg_width, vlane_ctx=3,
                                     logical_source_registers=[lreg(i, 8)])
                ]
        else:
            for i in range(explicit_split_count):
                dep_group_id = head_id + i*100
                per_head_insts += [
                    # Compute exp2(x - max)
                    LoadInstruction(id=dep_group_id + 2, target_register=dep_group_id + 2, dependencies=max_reduce_fake_dest,
                                    data_size=reg_width, vlane_ctx=0,
                                    logical_target_register=lreg(i, 2)),
                    EXP2Instruction(id=dep_group_id + 8, target_register=dep_group_id + 8, source_registers=[dep_group_id + 2],
                                    data_size=reg_width,
                                    logical_target_register=lreg(i, 8),
                                    logical_source_registers=[lreg(i, 2)]),
                    StoreInstruction(id=dep_group_id + 9, target_mem=dep_group_id + 9,
                                     source_registers=[dep_group_id + 8], data_size=reg_width, vlane_ctx=3,
                                     logical_source_registers=[lreg(i, 8)])
                ]

        sum_reduce_fake_dest = []
        for i in range(explicit_split_count):
            dep_group_id = head_id + i*100
            per_head_insts += [
                # Sum all exp values
                ReduceInstruction(id=dep_group_id + 10, target_register=dep_group_id + 10, source_registers=[dep_group_id + 8],
                                  data_size=reg_width,
                                  logical_target_register=lreg(i, 10),
                                  logical_source_registers=[lreg(i, 8)]),
            ]
            sum_reduce_fake_dest.append(dep_group_id + 10)
        
        for i in range(explicit_split_count):
            dep_group_id = head_id + i*100
            per_head_insts += [
                LoadInstruction(id=dep_group_id + 11, target_register=dep_group_id + 11,
                                dependencies=[dep_group_id + 9], data_size=reg_width, vlane_ctx=3,
                                logical_target_register=lreg(i, 11)),
                # Divide by sum (exp / sum)
                FMAInstruction(id=dep_group_id + 12, target_register=dep_group_id + 12,
                               source_registers=[dep_group_id + 11] + sum_reduce_fake_dest,
                               data_size=reg_width,
                               logical_target_register=lreg(i, 12),
                               logical_source_registers=[lreg(i, 11)] + [lreg(j, 10) for j in range(explicit_split_count)]),
                # Store result
                StoreInstruction(id=dep_group_id + 13, source_registers=[dep_group_id + 12],
                                 data_size=reg_width, vlane_ctx=1,
                                 logical_source_registers=[lreg(i, 12)])
            ]

        # Assign context (register group) for this head
        ctx = h % num_contexts
        for wrapper in per_head_insts:
            wrapper.instruction.context_id = ctx

        all_insts.extend(per_head_insts)
    
    # Extract the underlying Instruction objects for compatibility
    return [wrapper.instruction for wrapper in all_insts]


def create_rmsnorm_instruction_stream(reg_width, num_rows, num_contexts=1, lane_strides=None) -> List[Instruction]:
    """Create an RMSNorm-like instruction stream.

    vlane0 models per-row input, vlane1 per-row output, and vlane2 shared
    weights. The dependencies intentionally form a long chain to expose TLP
    benefits from multiple contexts.
    """
    data_size = reg_width
    all_insts = []

    for row in range(num_rows):
        base = row * 100
        row_insts = [
            LoadInstruction(id=base + 0, target_register=base + 0,
                            data_size=data_size, vlane_ctx=0,
                            logical_target_register=0),
            FMAInstruction(id=base + 1, target_register=base + 1,
                           source_registers=[base + 0], data_size=data_size,
                           logical_target_register=1,
                           logical_source_registers=[0]),
            ReduceInstruction(id=base + 2, target_register=base + 2,
                              source_registers=[base + 1], data_size=data_size,
                              logical_target_register=2,
                              logical_source_registers=[1]),
            FMAInstruction(id=base + 3, target_register=base + 3,
                           source_registers=[base + 0, base + 2],
                           dependencies=[base + 0, base + 2], data_size=data_size,
                           logical_target_register=3,
                           logical_source_registers=[0, 2]),
            LoadInstruction(id=base + 4, target_register=base + 4,
                            dependencies=[base + 3], data_size=data_size, vlane_ctx=2,
                            logical_target_register=4),
            FMAInstruction(id=base + 5, target_register=base + 5,
                           source_registers=[base + 3, base + 4], data_size=data_size,
                           logical_target_register=5,
                           logical_source_registers=[3, 4]),
            StoreInstruction(id=base + 6, source_registers=[base + 5],
                             data_size=data_size, vlane_ctx=1,
                             logical_source_registers=[5]),
        ]

        ctx = row % num_contexts
        for wrapper in row_insts:
            wrapper.instruction.context_id = ctx
        all_insts.extend(row_insts)

    return [wrapper.instruction for wrapper in all_insts]


def create_silu_instruction_stream(reg_width, has_exp2_unit, num_rows,
                                   num_contexts=1, lane_strides=None) -> List[Instruction]:
    """Create a SiLU activation instruction stream."""
    data_size = reg_width
    all_insts = []

    for row in range(num_rows):
        base = row * 100
        row_insts = [
            LoadInstruction(id=base + 0, target_register=base + 0,
                            data_size=data_size, vlane_ctx=0,
                            logical_target_register=0),
            FMAInstruction(id=base + 1, target_register=base + 1,
                           source_registers=[base + 0], data_size=data_size,
                           logical_target_register=1,
                           logical_source_registers=[0]),
        ]
        if has_exp2_unit:
            row_insts += [
                EXP2Instruction(id=base + 2, target_register=base + 2,
                                source_registers=[base + 1], data_size=data_size,
                                logical_target_register=2,
                                logical_source_registers=[1]),
                FMAInstruction(id=base + 3, target_register=base + 3,
                               source_registers=[base + 2], data_size=data_size,
                               logical_target_register=3,
                               logical_source_registers=[2]),
                FMAInstruction(id=base + 4, target_register=base + 4,
                               source_registers=[base + 3], data_size=data_size,
                               logical_target_register=4,
                               logical_source_registers=[3]),
                StoreInstruction(id=base + 5, source_registers=[base + 4],
                                 data_size=data_size, vlane_ctx=1,
                                 logical_source_registers=[4]),
            ]
        else:
            row_insts += [
                FMAInstruction(id=base + 2, target_register=base + 2,
                               source_registers=[base + 1], data_size=data_size,
                               logical_target_register=2,
                               logical_source_registers=[1]),
                FMAInstruction(id=base + 3, target_register=base + 3,
                               source_registers=[base + 2], data_size=data_size,
                               logical_target_register=3,
                               logical_source_registers=[2]),
                FMAInstruction(id=base + 4, target_register=base + 4,
                               source_registers=[base + 3], data_size=data_size,
                               logical_target_register=4,
                               logical_source_registers=[3]),
                FMAInstruction(id=base + 5, target_register=base + 5,
                               source_registers=[base + 4], data_size=data_size,
                               logical_target_register=5,
                               logical_source_registers=[4]),
                FMAInstruction(id=base + 6, target_register=base + 6,
                               source_registers=[base + 5], data_size=data_size,
                               logical_target_register=6,
                               logical_source_registers=[5]),
                FMAInstruction(id=base + 7, target_register=base + 7,
                               source_registers=[base + 6], data_size=data_size,
                               logical_target_register=7,
                               logical_source_registers=[6]),
                FMAInstruction(id=base + 8, target_register=base + 8,
                               source_registers=[base + 7], data_size=data_size,
                               logical_target_register=8,
                               logical_source_registers=[7]),
                FMAInstruction(id=base + 9, target_register=base + 9,
                               source_registers=[base + 8], data_size=data_size,
                               logical_target_register=9,
                               logical_source_registers=[8]),
                StoreInstruction(id=base + 10, source_registers=[base + 9],
                                 data_size=data_size, vlane_ctx=1,
                                 logical_source_registers=[9]),
            ]

        ctx = row % num_contexts
        for wrapper in row_insts:
            wrapper.instruction.context_id = ctx
        all_insts.extend(row_insts)

    return [wrapper.instruction for wrapper in all_insts]


def create_rope_instruction_stream(reg_width, num_rows, num_contexts=1, lane_strides=None) -> List[Instruction]:
    """Create a simplified RoPE instruction stream.

    vlane2 is used for shared theta, vlane0 for input, and vlane1 for output.
    The sin/cos polynomial expansion is abstracted as FMA work.
    """
    data_size = reg_width
    all_insts = []

    for row in range(num_rows):
        base = row * 100
        row_insts = [
            LoadInstruction(id=base + 0, target_register=base + 0,
                            data_size=data_size, vlane_ctx=2,
                            logical_target_register=0),
            FMAInstruction(id=base + 1, target_register=base + 1,
                           source_registers=[base + 0], data_size=data_size,
                           logical_target_register=1,
                           logical_source_registers=[0]),
            LoadInstruction(id=base + 2, target_register=base + 2,
                            dependencies=[base + 1], data_size=data_size, vlane_ctx=0,
                            logical_target_register=2),
            FMAInstruction(id=base + 3, target_register=base + 3,
                           source_registers=[base + 1, base + 2], data_size=data_size,
                           logical_target_register=3,
                           logical_source_registers=[1, 2]),
            FMAInstruction(id=base + 4, target_register=base + 4,
                           source_registers=[base + 1, base + 2, base + 3],
                           dependencies=[base + 1, base + 2, base + 3], data_size=data_size,
                           logical_target_register=4,
                           logical_source_registers=[1, 2, 3]),
            StoreInstruction(id=base + 5, source_registers=[base + 3, base + 4],
                             data_size=data_size, vlane_ctx=1,
                             logical_source_registers=[3, 4]),
        ]

        ctx = row % num_contexts
        for wrapper in row_insts:
            wrapper.instruction.context_id = ctx
        all_insts.extend(row_insts)

    return [wrapper.instruction for wrapper in all_insts]


def create_instruction_stream(kernel, reg_width, has_exp2_unit, num_heads, num_rows,
                              seq_chunk_bits, num_contexts=1, lane_strides=None) -> List[Instruction]:
    """Create a kernel-specific instruction stream with a common interface."""
    if kernel == "softmax":
        return create_softmax_instruction_stream(
            reg_width, has_exp2_unit, num_heads, seq_chunk_bits,
            num_contexts=num_contexts, lane_strides=lane_strides,
        )
    if kernel == "rmsnorm":
        return create_rmsnorm_instruction_stream(
            reg_width, num_rows, num_contexts=num_contexts, lane_strides=lane_strides,
        )
    if kernel == "silu":
        return create_silu_instruction_stream(
            reg_width, has_exp2_unit, num_rows,
            num_contexts=num_contexts, lane_strides=lane_strides,
        )
    if kernel == "rope":
        return create_rope_instruction_stream(
            reg_width, num_rows, num_contexts=num_contexts, lane_strides=lane_strides,
        )
    raise ValueError(f"Unknown kernel: {kernel}")


def run_kernel_simulation(args, kernel=None, num_contexts=None, issue_width=None, quiet=True) -> Tuple[Dict, VectorProcessor, List[Instruction]]:
    """Build a config, run one simulation, and return results plus processor state."""
    selected_kernel = kernel or args.kernel
    selected_contexts = num_contexts if num_contexts is not None else args.num_contexts
    selected_issue_width = issue_width if issue_width is not None else args.issue_width

    execution_mode = ExecutionMode.IN_ORDER if args.execution_mode == "in-order" else ExecutionMode.OUT_OF_ORDER

    if args.all_compute_widths is not None:
        reduce_width = args.all_compute_widths
        simple_width = args.all_compute_widths
        complex_width = args.all_compute_widths
    else:
        reduce_width = args.reduce_compute_width
        simple_width = args.simple_elementwise_width
        complex_width = args.complex_elementwise_width

    lane_strides = (
        args.lane_stride_0,
        args.lane_stride_1,
        args.lane_stride_2,
        args.lane_stride_3,
    )

    config = ProcessorConfig(
        register_width=args.register_width,
        reduce_compute_unit_width=reduce_width,
        simple_elementwise_compute_unit_width=simple_width,
        complex_elementwise_compute_unit_width=complex_width,
        cache_bandwidth=args.cache_bandwidth,
        chaining_enabled=args.chaining,
        chaining_granularity=64,
        execution_mode=execution_mode,
        issue_queue_window=args.issue_queue_window,
        ooo_scheduler_window_size=args.ooo_scheduler_window_size,
        register_renaming=args.register_renaming,
        issue_width=selected_issue_width,
        num_contexts=selected_contexts,
        lane_strides=lane_strides,
    )

    instructions = create_instruction_stream(
        selected_kernel,
        args.register_width,
        args.exp2_unit,
        args.num_heads,
        args.num_rows,
        args.seq_chunk_bits,
        num_contexts=selected_contexts,
        lane_strides=lane_strides,
    )
    for inst in instructions:
        if inst.type in [InstructionType.LOAD, InstructionType.STORE] and inst.vlane_ctx >= len(lane_strides):
            raise ValueError(f"Instruction {inst.id} has invalid vlane_ctx {inst.vlane_ctx}")

    processor = VectorProcessor(config, quiet=quiet)
    processor.load_instructions(instructions)
    results = processor.simulate(max_cycles=args.max_cycles)

    assert processor.get_outstanding_instruction_count() == 0
    assert processor.get_outstanding_uop_count() == 0

    return results, processor, instructions


BENCHMARK_ALIASES = {
    'lanes': 'num_contexts',
    'lane_stride0': 'lane_stride_0',
    'lane_stride1': 'lane_stride_1',
    'lane_stride2': 'lane_stride_2',
    'lane_stride3': 'lane_stride_3',
    'lane-stride-0': 'lane_stride_0',
    'lane-stride-1': 'lane_stride_1',
    'lane-stride-2': 'lane_stride_2',
    'lane-stride-3': 'lane_stride_3',
    'issue-width': 'issue_width',
    'num-contexts': 'num_contexts',
    'num-heads': 'num_heads',
    'num-rows': 'num_rows',
    'seq-chunk-bits': 'seq_chunk_bits',
    'register-width': 'register_width',
    'cache-bandwidth': 'cache_bandwidth',
    'ooo_window_size': 'ooo_scheduler_window_size',
    'ooo-window-size': 'ooo_scheduler_window_size',
    'scheduler-window-size': 'ooo_scheduler_window_size',
    'scheduler_window_size': 'ooo_scheduler_window_size',
    'issue-queue-window': 'issue_queue_window',
    'register-renaming': 'register_renaming',
}

BENCHMARK_INT_FIELDS = {
    'num_heads',
    'num_rows',
    'seq_chunk_bits',
    'register_width',
    'all_compute_widths',
    'reduce_compute_width',
    'simple_elementwise_width',
    'complex_elementwise_width',
    'cache_bandwidth',
    'issue_queue_window',
    'ooo_scheduler_window_size',
    'issue_width',
    'num_contexts',
    'lane_stride_0',
    'lane_stride_1',
    'lane_stride_2',
    'lane_stride_3',
    'max_cycles',
}

BENCHMARK_BOOL_FIELDS = {'exp2_unit', 'chaining', 'quiet', 'register_renaming'}
BENCHMARK_METADATA_KEYS = {
    'metadata',
    'description',
    'dependency_chain',
    'latency_diagram',
    'parameter_notes',
    'lane_mapping',
    'microarchitecture',
    'workload_model',
    'notes',
}


def _normalize_benchmark_key(key: str) -> str:
    normalized = key.strip().replace('-', '_')
    return BENCHMARK_ALIASES.get(normalized, normalized)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _md_cell(value) -> str:
    return str(value).replace("|", "\\|")


def _format_pct(value) -> str:
    return f"{value:.1f}%"


def _ordered_unit_metrics(metrics: Dict) -> List[Tuple[str, float]]:
    unit_active_pct = metrics.get('unit_active_pct', {})
    return [
        ('load', unit_active_pct.get('load', 0.0)),
        ('store', unit_active_pct.get('store', 0.0)),
        ('fma', unit_active_pct.get('fma', 0.0)),
        ('reduce', unit_active_pct.get('reduce', 0.0)),
        ('exp2', unit_active_pct.get('exp2', 0.0)),
    ]


def format_compact_utilization(results: Dict, processor: VectorProcessor, instruction_count: int = 0) -> str:
    """Format a short simulation summary suitable for --quiet output."""
    metrics = processor.get_utilization_metrics()
    lines = [
        f"Total execution time: {results['total_cycles']} cycles",
        f"Issue slot utilization: {_format_pct(metrics['issue_slot_utilization'])} "
        f"(avg {metrics['average_issued']:.2f} uops/cycle)",
        "Execution unit utilization: " +
        ", ".join(f"{name} {_format_pct(value)}" for name, value in _ordered_unit_metrics(metrics)),
    ]
    if instruction_count and results['total_cycles']:
        throughput = instruction_count / results['total_cycles']
        lines.append(f"Instruction throughput: {throughput:.3f} instructions/cycle")
    return "\n".join(lines)


def _coerce_benchmark_value(key: str, value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    if key in BENCHMARK_BOOL_FIELDS:
        return _parse_bool(value)
    if key in BENCHMARK_INT_FIELDS:
        return int(value)
    return value


def _split_csv_sweep_value(value):
    if not isinstance(value, str):
        return value
    if '|' in value:
        return [part.strip() for part in value.split('|') if part.strip()]
    return value


def _expand_benchmark_case(raw_case: Dict) -> List[Dict]:
    """Expand list-valued fields into concrete benchmark cases."""
    normalized = {}
    passthrough = {}
    for raw_key, raw_value in raw_case.items():
        key = _normalize_benchmark_key(str(raw_key))
        if key in BENCHMARK_METADATA_KEYS:
            passthrough[key] = raw_value
            continue
        value = _split_csv_sweep_value(raw_value)
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple)):
            values = [_coerce_benchmark_value(key, item) for item in value]
            normalized[key] = [item for item in values if item is not None]
        else:
            coerced = _coerce_benchmark_value(key, value)
            if coerced is not None:
                normalized[key] = coerced

    expanded = [{}]
    for key, value in normalized.items():
        values = value if isinstance(value, list) else [value]
        next_expanded = []
        for partial in expanded:
            for item in values:
                new_case = partial.copy()
                new_case[key] = item
                next_expanded.append(new_case)
        expanded = next_expanded

    if passthrough:
        for case in expanded:
            case.update(passthrough)

    return expanded


def _default_benchmark_cases() -> List[Dict]:
    raw_cases = []
    for kernel in ["softmax", "rmsnorm", "silu", "rope"]:
        raw_cases.append({
            'kernel': kernel,
            'execution_mode': ['in-order', 'out-of-order'],
            'num_contexts': [1, 2, 4],
            'issue_width': [1, 2],
        })

    cases = []
    for raw_case in raw_cases:
        cases.extend(_expand_benchmark_case(raw_case))
    return cases


def _load_yaml_benchmark_cases(path: str) -> Tuple[List[Dict], Dict]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML benchmark configs require PyYAML. Install it or use CSV."
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return [], {}
    if isinstance(data, list):
        return data, {}
    if not isinstance(data, dict):
        raise ValueError("YAML benchmark config must be a list or mapping")

    defaults = data.get('defaults', {})
    hardware_configs = data.get('hardware_configs') or data.get('hardware') or [{}]
    if isinstance(hardware_configs, dict):
        hardware_configs = [hardware_configs]
    if not isinstance(hardware_configs, list):
        raise ValueError("YAML hardware_configs must be a mapping or list of mappings")

    raw_cases = (
        data.get('cases') or
        data.get('benchmarks') or
        data.get('sweeps') or
        []
    )
    if not raw_cases:
        raw_cases = [
            {key: value for key, value in data.items()
             if key not in {'defaults', 'hardware_configs', 'hardware'}}
        ]

    combined_cases = []
    for hardware in hardware_configs:
        if hardware is None:
            hardware = {}
        if not isinstance(hardware, dict):
            raise ValueError(f"Hardware config must be a mapping, got {type(hardware).__name__}")
        hardware_name = hardware.get('name')
        hardware_fields = {key: value for key, value in hardware.items() if key != 'name'}
        for raw_case in raw_cases:
            if raw_case is None:
                continue
            if not isinstance(raw_case, dict):
                raise ValueError(f"Benchmark case must be a mapping, got {type(raw_case).__name__}")
            merged = hardware_fields.copy()
            merged.update(raw_case)
            if hardware_name is not None:
                metadata = merged.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata = metadata.copy()
                metadata.setdefault('hardware_config', hardware_name)
                merged['metadata'] = metadata
                if merged.get('name'):
                    merged['name'] = f"{hardware_name}/{merged['name']}"
            combined_cases.append(merged)

    return combined_cases, defaults


def _load_csv_benchmark_cases(path: str) -> Tuple[List[Dict], Dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            line for line in f
            if line.strip() and not line.lstrip().startswith("#")
        )
        if reader.fieldnames is None:
            return [], {}
        return [dict(row) for row in reader], {}


def load_benchmark_cases(path: str) -> List[Dict]:
    """Load benchmark cases from CSV or YAML and expand sweep fields."""
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if suffix in {"yaml", "yml"}:
        raw_cases, defaults = _load_yaml_benchmark_cases(path)
    elif suffix == "csv":
        raw_cases, defaults = _load_csv_benchmark_cases(path)
    else:
        raise ValueError("Benchmark config must use .csv, .yaml, or .yml")

    cases = []
    normalized_defaults = {}
    for key, value in defaults.items():
        normalized_key = _normalize_benchmark_key(str(key))
        normalized_defaults[normalized_key] = value

    for raw_case in raw_cases:
        if raw_case is None:
            continue
        if not isinstance(raw_case, dict):
            raise ValueError(f"Benchmark case must be a mapping, got {type(raw_case).__name__}")
        merged = normalized_defaults.copy()
        merged.update(raw_case)
        cases.extend(_expand_benchmark_case(merged))

    return cases


def _apply_benchmark_case(args, case: Dict):
    case_args = copy.copy(args)
    allowed_keys = set(vars(args).keys()) | {'name'}
    for raw_key, raw_value in case.items():
        key = _normalize_benchmark_key(str(raw_key))
        if key == 'name' or key in BENCHMARK_METADATA_KEYS:
            continue
        if key not in allowed_keys:
            raise ValueError(f"Unknown benchmark config field: {raw_key}")
        setattr(case_args, key, raw_value)
    return case_args


def _run_benchmark_case(args, case: Dict) -> Dict:
    case_args = _apply_benchmark_case(args, case)
    results, processor, instructions = run_kernel_simulation(case_args, quiet=True)
    metrics = processor.get_utilization_metrics()
    metadata = case.get('metadata', {})
    if not isinstance(metadata, dict):
        metadata = {}
    description = case.get('description') or metadata.get('description', '')
    dependency_chain = case.get('dependency_chain') or metadata.get('dependency_chain', '')
    latency_diagram = case.get('latency_diagram') or metadata.get('latency_diagram', '')
    parameter_notes = case.get('parameter_notes') or metadata.get('parameter_notes', {})
    return {
        'name': str(case.get('name', '')),
        'description': str(description),
        'dependency_chain': str(dependency_chain),
        'latency_diagram': str(latency_diagram),
        'parameter_notes': parameter_notes,
        'execution_mode': case_args.execution_mode,
        'kernel': case_args.kernel,
        'exp2_unit': case_args.exp2_unit,
        'lanes': case_args.num_contexts,
        'issue_width': case_args.issue_width,
        'cycles': results['total_cycles'],
        'speedup': 1.0,
        'ooo_speedup': None,
        'issue_util': metrics['issue_slot_utilization'],
        'load_util': metrics['unit_active_pct']['load'],
        'store_util': metrics['unit_active_pct']['store'],
        'fma_util': metrics['unit_active_pct']['fma'],
        'reduce_util': metrics['unit_active_pct']['reduce'],
        'exp2_util': metrics['unit_active_pct']['exp2'],
        'instruction_count': len(instructions),
        'uop_count': len(processor.uops),
        'num_heads': case_args.num_heads,
        'num_rows': case_args.num_rows,
        'seq_chunk_bits': case_args.seq_chunk_bits,
        'register_width': case_args.register_width,
        'cache_bandwidth': case_args.cache_bandwidth,
        'reduce_compute_width': case_args.reduce_compute_width,
        'simple_elementwise_width': case_args.simple_elementwise_width,
        'complex_elementwise_width': case_args.complex_elementwise_width,
        'all_compute_widths': case_args.all_compute_widths,
        'chaining': case_args.chaining,
        'register_renaming': case_args.register_renaming,
        'issue_queue_window': case_args.issue_queue_window,
        'ooo_scheduler_window_size': case_args.ooo_scheduler_window_size,
        'lane_strides': (
            case_args.lane_stride_0,
            case_args.lane_stride_1,
            case_args.lane_stride_2,
            case_args.lane_stride_3,
        ),
    }


def _benchmark_mode_compare_key(row: Dict) -> Tuple:
    return (
        row['name'],
        row['kernel'],
        row['exp2_unit'],
        row['lanes'],
        row['issue_width'],
        row['num_heads'],
        row['num_rows'],
        row['seq_chunk_bits'],
        row['register_width'],
        row['cache_bandwidth'],
        row['reduce_compute_width'],
        row['simple_elementwise_width'],
        row['complex_elementwise_width'],
        row['all_compute_widths'],
        row['chaining'],
        row['register_renaming'],
        row['issue_queue_window'],
        row['ooo_scheduler_window_size'],
        row['lane_strides'],
    )


def _benchmark_baseline_key(row: Dict) -> Tuple:
    return (
        row['name'],
        row['kernel'],
        row['exp2_unit'],
        row['num_heads'],
        row['num_rows'],
        row['seq_chunk_bits'],
        row['register_width'],
        row['cache_bandwidth'],
        row['reduce_compute_width'],
        row['simple_elementwise_width'],
        row['complex_elementwise_width'],
        row['all_compute_widths'],
        row['chaining'],
        row['register_renaming'],
        row['issue_queue_window'],
        row['ooo_scheduler_window_size'],
        row['lane_strides'],
    )


def _annotate_speedups(rows: List[Dict]):
    baseline_by_config = {}
    in_order_cycles_by_config = {}

    for row in rows:
        if (row['execution_mode'] == 'in-order' and
                row['lanes'] == 1 and row['issue_width'] == 1):
            baseline_by_config.setdefault(_benchmark_baseline_key(row), row['cycles'])
        if row['execution_mode'] == 'in-order':
            in_order_cycles_by_config.setdefault(_benchmark_mode_compare_key(row), row['cycles'])

    for row in rows:
        baseline_by_config.setdefault(_benchmark_baseline_key(row), row['cycles'])

    for row in rows:
        baseline_cycles = baseline_by_config[_benchmark_baseline_key(row)]
        row['speedup'] = baseline_cycles / row['cycles'] if row['cycles'] else 0.0
        in_order_cycles = in_order_cycles_by_config.get(_benchmark_mode_compare_key(row))
        row['ooo_speedup'] = in_order_cycles / row['cycles'] if in_order_cycles and row['cycles'] else None


def _format_benchmark_values(rows: List[Dict], key: str) -> str:
    """Format the concrete values used by expanded benchmark cases."""
    values = []
    for row in rows:
        value = row[key]
        if value not in values:
            values.append(value)
    return ", ".join(str(value) for value in values)


def _resolve_benchmark_workers(args, case_count: int) -> int:
    """Resolve benchmark worker count; 0 or less means auto parallelism."""
    if case_count < 1:
        return 1
    requested_workers = args.benchmark_workers
    if requested_workers is None or requested_workers <= 0:
        return max(1, min(case_count, os.cpu_count() or 1))
    return max(1, min(requested_workers, case_count))


def run_benchmark_suite(args) -> str:
    """Run a compact multi-kernel, multi-lane benchmark and return Markdown."""
    if args.benchmark_config:
        cases = load_benchmark_cases(args.benchmark_config)
        source = args.benchmark_config
    else:
        cases = _default_benchmark_cases()
        source = "built-in default sweep"

    if not cases:
        raise ValueError("No benchmark cases to run")

    benchmark_workers = _resolve_benchmark_workers(args, len(cases))
    if benchmark_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=benchmark_workers) as executor:
            futures = [executor.submit(_run_benchmark_case, args, case) for case in cases]
            rows = [future.result() for future in futures]
    else:
        rows = [_run_benchmark_case(args, case) for case in cases]
    _annotate_speedups(rows)
    has_case_names = any(row['name'] for row in rows)
    has_case_descriptions = any(row['description'] for row in rows)
    has_case_notes = any(
        row['description'] or row['dependency_chain'] or row['latency_diagram'] or row['parameter_notes']
        for row in rows
    )

    lines = [
        "# Multi-Kernel Benchmark Results",
        "",
        "Generated by `python3 softmax_simulator.py --benchmark`.",
        "",
        "## Configuration",
        "",
        f"- benchmark source: `{source}`",
        f"- benchmark workers: `{benchmark_workers}`",
        f"- execution modes: `{_format_benchmark_values(rows, 'execution_mode')}`",
        f"- register renaming: `{_format_benchmark_values(rows, 'register_renaming')}`",
        f"- issue queue windows: `{_format_benchmark_values(rows, 'issue_queue_window')}` not-yet-completed uops",
        f"- out-of-order scheduler window sizes: `{_format_benchmark_values(rows, 'ooo_scheduler_window_size')}` not-yet-issued uops",
        f"- register widths: `{_format_benchmark_values(rows, 'register_width')}` bits",
        f"- sequence chunks: `{_format_benchmark_values(rows, 'seq_chunk_bits')}` bits",
        f"- rows per non-softmax kernel: `{_format_benchmark_values(rows, 'num_rows')}`",
        f"- softmax heads: `{_format_benchmark_values(rows, 'num_heads')}`",
        f"- cache bandwidths: `{_format_benchmark_values(rows, 'cache_bandwidth')}` bytes/cycle",
        f"- all compute widths: `{_format_benchmark_values(rows, 'all_compute_widths')}` bits",
        f"- exp2 unit values: `{_format_benchmark_values(rows, 'exp2_unit')}`",
        "",
        "`baseline speedup` is normalized per kernel to the matching in-order `lanes=1, issue_width=1` case when present; otherwise the first case for that kernel is used.",
        "`ooo speedup` compares each row against the matching in-order case with the same case/kernel/lanes/issue_width/workload/hardware settings.",
        "",
        "## Summary",
        "",
    ]

    if has_case_names and has_case_descriptions:
        lines.extend([
            "| case | description | kernel | mode | rename | lanes | issue width | cycles | baseline speedup | ooo speedup | issue util | load | store | fma | reduce | exp2 |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
    elif has_case_names:
        lines.extend([
            "| case | kernel | mode | rename | lanes | issue width | cycles | baseline speedup | ooo speedup | issue util | load | store | fma | reduce | exp2 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
    else:
        lines.extend([
            "| kernel | mode | rename | lanes | issue width | cycles | baseline speedup | ooo speedup | issue util | load | store | fma | reduce | exp2 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])

    for row in rows:
        row_cells = [
            _md_cell(row['kernel']),
            _md_cell(row['execution_mode']),
            str(row['register_renaming']),
            str(row['lanes']),
            str(row['issue_width']),
            str(row['cycles']),
            f"{row['speedup']:.2f}x",
            f"{row['ooo_speedup']:.2f}x" if row['ooo_speedup'] is not None else "-",
            f"{row['issue_util']:.1f}%",
            f"{row['load_util']:.1f}%",
            f"{row['store_util']:.1f}%",
            f"{row['fma_util']:.1f}%",
            f"{row['reduce_util']:.1f}%",
            f"{row['exp2_util']:.1f}%",
        ]
        if has_case_names and has_case_descriptions:
            row_cells.insert(0, _md_cell(row['description']))
            row_cells.insert(0, _md_cell(row['name']))
        elif has_case_names:
            row_cells.insert(0, _md_cell(row['name']))
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.extend([
        "",
        "## Instruction Mix",
        "",
    ])

    if has_case_names:
        if has_case_descriptions:
            lines.extend([
                "| case | description | kernel | mode | rename | lanes | issue width | instructions | micro-ops |",
                "|---|---|---|---|---|---:|---:|---:|---:|",
            ])
        else:
            lines.extend([
                "| case | kernel | mode | rename | lanes | issue width | instructions | micro-ops |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ])
    else:
        lines.extend([
            "| kernel | mode | rename | lanes | issue width | instructions | micro-ops |",
            "|---|---|---|---:|---:|---:|---:|",
        ])

    for row in rows:
        row_cells = [
            _md_cell(row['kernel']),
            _md_cell(row['execution_mode']),
            str(row['register_renaming']),
            str(row['lanes']),
            str(row['issue_width']),
            str(row['instruction_count']),
            str(row['uop_count']),
        ]
        if has_case_names and has_case_descriptions:
            row_cells.insert(0, _md_cell(row['description']))
            row_cells.insert(0, _md_cell(row['name']))
        elif has_case_names:
            row_cells.insert(0, _md_cell(row['name']))
        lines.append("| " + " | ".join(row_cells) + " |")

    if has_case_notes:
        lines.extend([
            "",
            "## Case Notes",
            "",
        ])
        seen_notes = set()
        for row in rows:
            if not row['name']:
                continue
            if row['name'] in seen_notes:
                continue
            seen_notes.add(row['name'])
            lines.append(f"- `{row['name']}`")
            if row['description']:
                lines.append(f"  - description: {row['description']}")
            if row['parameter_notes']:
                lines.append("  - parameters:")
                notes = row['parameter_notes']
                if isinstance(notes, dict):
                    for key, value in notes.items():
                        lines.append(f"    - {key}: {value}")
                else:
                    lines.append(f"    - {notes}")
            if row['dependency_chain']:
                lines.append(f"  - dependency chain: {row['dependency_chain']}")
            if row['latency_diagram']:
                lines.append("  - single-chain latency diagram:")
                lines.append("    ```text")
                for line in row['latency_diagram'].splitlines():
                    lines.append(f"    {line}")
                lines.append("    ```")

    return "\n".join(lines) + "\n"


def format_benchmark_quiet_summary(markdown_report: str, output_path: str) -> str:
    """Extract the benchmark summary table for compact --quiet stdout."""
    lines = markdown_report.splitlines()
    try:
        start = lines.index("## Summary")
    except ValueError:
        return f"Wrote benchmark results to {output_path}"

    summary_lines = [f"Wrote benchmark results to {output_path}", ""]
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.strip():
            summary_lines.append(line)
    return "\n".join(summary_lines)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RISC-V Vector Processor Softmax Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--execution-mode", 
        choices=["in-order", "out-of-order"], 
        default="in-order",
        help="Execution mode for the processor"
    )

    parser.add_argument(
        "--exp2-unit",
        action="store_true",
        help="If true implement softmax and silu with a dedicated exp2 unit; otherwise use FMA approximation paths"
    )

    parser.add_argument(
        "--kernel",
        choices=["softmax", "rmsnorm", "silu", "rope"],
        default="softmax",
        help="Kernel instruction stream to simulate"
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="The number of attention heads for softmax",
    )

    parser.add_argument(
        "--num-rows",
        type=int,
        default=8,
        help="The number of independent rows for rmsnorm, silu, and rope",
    )

    parser.add_argument(
        '--seq-chunk-bits',
        type=int,
        choices=[512, 1024, 2048, 4096, 8192],
        default=2048,
        help="The sequence chunk bits",
    )
    
    parser.add_argument(
        "--register-width",
        type=int,
        choices=[512, 1024, 2048, 4096],
        default=2048,
        help="Register width in bits"
    )

    parser.add_argument(
        "--all-compute-widths",
        type=int,
        choices=[128, 256, 512, 1024],
        default=512,
        help="Control all compute unit widths in bits",
    )
    
    parser.add_argument(
        "--reduce-compute-width",
        type=int,
        choices=[128, 256, 512, 1024],
        default=512,
        help="Reduce compute unit width in bits"
    )
    
    parser.add_argument(
        "--simple-elementwise-width",
        type=int,
        choices=[128, 256, 512, 1024],
        default=512,
        help="Simple elementwise compute unit width in bits"
    )
    
    parser.add_argument(
        "--complex-elementwise-width",
        type=int,
        choices=[128, 256, 512, 1024],
        default=512,
        help="Complex elementwise compute unit width in bits"
    )
    
    parser.add_argument(
        "--cache-bandwidth",
        type=int,
        choices=[32, 64, 128],
        default=64,
        help="Cache bandwidth in bytes per cycle"
    )
    
    parser.add_argument(
        "--chaining",
        action="store_true",
        default=True,
        help="Enable chaining"
    )

    rename_group = parser.add_mutually_exclusive_group()
    rename_group.add_argument(
        "--register-renaming",
        dest="register_renaming",
        action="store_true",
        default=True,
        help="Enable physical register renaming for independent heads or rows"
    )
    rename_group.add_argument(
        "--no-register-renaming",
        dest="register_renaming",
        action="store_false",
        help="Disable register renaming and serialize repeated logical registers within each context"
    )
    
    parser.add_argument(
        "--issue-queue-window",
        type=int,
        default=10,
        help="Oldest not-yet-completed uops visible to the issue queue"
    )

    parser.add_argument(
        "--ooo-scheduler-window-size",
        "--ooo-window-size",
        dest="ooo_scheduler_window_size",
        type=int,
        default=128,
        help="Not-yet-issued uops visible to the out-of-order scheduler inside the issue queue window"
    )

    parser.add_argument(
        "--issue-width",
        type=int,
        choices=[1, 2],
        default=2,
        help="Max μops issued per cycle (1=single-issue, 2=dual-issue)"
    )

    parser.add_argument(
        "--num-contexts",
        type=int,
        default=1,
        help="Number of hardware contexts (register groups), heads are round-robin assigned"
    )

    parser.add_argument(
        "--lane-stride-0",
        type=int,
        default=0,
        help="vlane0 stride in bytes"
    )

    parser.add_argument(
        "--lane-stride-1",
        type=int,
        default=0,
        help="vlane1 stride in bytes"
    )

    parser.add_argument(
        "--lane-stride-2",
        type=int,
        default=0,
        help="vlane2 stride in bytes"
    )

    parser.add_argument(
        "--lane-stride-3",
        type=int,
        default=0,
        help="vlane3 stride in bytes"
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=100000,
        help="Maximum simulation cycles before timeout"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run all kernels across 1/2/4 lanes and issue_width 1/2"
    )

    parser.add_argument(
        "--benchmark-config",
        help="CSV/YAML benchmark config. YAML supports defaults, metadata, and list-valued sweeps"
    )

    parser.add_argument(
        "--benchmark-output",
        default="plans/benchmark_results.md",
        help="Markdown output path for --benchmark"
    )

    parser.add_argument(
        "--benchmark-workers",
        type=int,
        default=0,
        help="Worker processes for benchmark cases; 0 means auto (min(case count, CPU count))"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print a compact summary without timelines or per-uop details"
    )

    return parser.parse_args()


def main():
    """Example usage of the softmax simulator"""
    # Parse command line arguments
    args = parse_arguments()
    if args.benchmark or args.benchmark_config:
        benchmark_md = run_benchmark_suite(args)
        benchmark_path = args.benchmark_output
        with open(benchmark_path, "w", encoding="utf-8") as f:
            f.write(benchmark_md)
        if not args.quiet:
            print(benchmark_md, end="")
            print(f"Wrote benchmark results to {benchmark_path}")
        else:
            print(format_benchmark_quiet_summary(benchmark_md, benchmark_path))
        return

    quiet = args.quiet
    results, processor, instructions = run_kernel_simulation(args, quiet=quiet)

    if not quiet:
        execution_mode = ExecutionMode.IN_ORDER if args.execution_mode == "in-order" else ExecutionMode.OUT_OF_ORDER
        if args.all_compute_widths is not None:
            reduce_width = args.all_compute_widths
            simple_width = args.all_compute_widths
            complex_width = args.all_compute_widths
        else:
            reduce_width = args.reduce_compute_width
            simple_width = args.simple_elementwise_width
            complex_width = args.complex_elementwise_width

        print("RISC-V Vector Processor Softmax Simulator")
        print("=" * 50)
        print("Configuration:")
        print(f"  Kernel: {args.kernel}")
        print(f"  Register width: {args.register_width} bits")
        print(f"  Reduce compute unit width: {reduce_width} bits")
        print(f"  Simple elementwise compute unit width: {simple_width} bits")
        print(f"  Complex elementwise compute unit width: {complex_width} bits")
        print(f"  Cache bandwidth: {args.cache_bandwidth} bytes/cycle")
        print(f"  Execution mode: {execution_mode.value}")
        print(f"  Register renaming: {'enabled' if args.register_renaming else 'disabled'}")
        print(f"  Issue queue window: {args.issue_queue_window} uops")
        print(f"  OOO scheduler window: {args.ooo_scheduler_window_size} uops")
        print(f"  Chaining: {'enabled' if args.chaining else 'disabled'}")
        print(f"  num_contexts: {args.num_contexts}")
        print(f"  lane strides: ({args.lane_stride_0}, {args.lane_stride_1}, {args.lane_stride_2}, {args.lane_stride_3})")
        print()

        print(f"Loaded {len(instructions)} instructions for {args.kernel} computation")
        print("Instructions:")
        for inst in instructions:
            deps_str = f"depends on {sorted(list(inst.dependencies))}" if inst.dependencies else "no dependencies"
            print(f"  {inst.id}: {inst.type.value} ({inst.data_size} bytes, ctx {inst.context_id}, "
                  f"vlane{inst.vlane_ctx}, {deps_str})")
        print()

        print("Instruction Timeline:")
        for inst_result in results['instructions']:
            issue_str = f"issue:{inst_result['issue_cycle']}" if inst_result['issue_cycle'] >= 0 else "issue:N/A"
            print(f"  Instruction {inst_result['id']} ({inst_result['type']}): "
                  f"{issue_str}, cycles {inst_result['start_cycle']}-{inst_result['complete_cycle']} "
                  f"(duration: {inst_result['execution_time']})")
        print()

        print(f"Generated {len(results['uops'])} micro-operations")
        print()

        print("Micro-operation (uop) Timeline:")
        for uop_result in results['uops']:
            inst_id = uop_result['instruction_id']
            uop_id = uop_result['uop_id']
            uop_type = uop_result['type']
            context_id = uop_result['context_id']
            vlane_ctx = uop_result['vlane_ctx']
            address = uop_result['address']
            start_cycle = uop_result['start_cycle']
            complete_cycle = uop_result['complete_cycle']
            execution_time = uop_result['execution_time']

            address_str = f", addr {address}" if address is not None else ""
            print(f"  uop {inst_id}.{uop_id} ({uop_type}): "
                  f"cycles {start_cycle}-{complete_cycle}, ctx {context_id}, vlane{vlane_ctx}{address_str} "
                  f"(duration: {execution_time})")
        print()

        processor.visualize_execution()
        print()
        processor.visualize_uop_execution()
        print()

    if quiet:
        print(format_compact_utilization(results, processor, len(instructions)))
        return

    print(f"Total execution time: {results['total_cycles']} cycles")
    if results['instructions']:
        throughput = len(instructions) / results['total_cycles']
        print(f"Instruction throughput: {throughput:.3f} instructions/cycle")
    print()

    processor.print_utilization()


if __name__ == "__main__":
    main()
