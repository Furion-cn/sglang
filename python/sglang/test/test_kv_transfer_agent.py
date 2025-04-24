import unittest
import torch
import numpy as np
from sglang.srt.managers.kv_transfer_agent import KVBuffer, KVBufferBlock


class TestKVBuffer(unittest.TestCase):
    def setUp(self):
        # Create a test cache tensor
        self.cache_size = 100
        self.cache = torch.zeros([self.cache_size, 2, 3, 4], dtype=torch.float32)
        self.kv_buffer = KVBuffer(self.cache)

    def test_basic_allocation_and_free(self):
        # Allocate a block of memory
        length = 10
        offset = self.kv_buffer.allocate(length)
        
        # Verify allocation
        self.assertEqual(offset, 0)
        self.assertEqual(self.kv_buffer.blocks[offset].length, length)  # length is actual used size
        self.assertEqual(self.kv_buffer.blocks[offset].capacity, length)  # capacity should match when no alignment
        self.assertEqual(len(self.kv_buffer.free_slots), 1)  # One remaining slot (the rest of the buffer)
        
        # Free the block
        self.kv_buffer.free(offset)
        self.assertEqual(len(self.kv_buffer.free_slots), 1)  # Now we have two free slots
        print(self.kv_buffer.free_slots)
        
        # Allocate again, should get the same offset
        new_offset = self.kv_buffer.allocate(length)
        self.assertEqual(new_offset, offset)

    def test_allocation_larger_than_available(self):
        # Allocate more than available
        with self.assertRaises(Exception):
            self.kv_buffer.allocate(self.cache_size + 1)

    def test_multiple_allocations(self):
        # Allocate multiple blocks
        offset1 = self.kv_buffer.allocate(10)
        offset2 = self.kv_buffer.allocate(20)
        offset3 = self.kv_buffer.allocate(30)
        
        # Verify allocations
        self.assertEqual(offset1, 0)
        self.assertEqual(offset2, 10)
        self.assertEqual(offset3, 30)
        
        # Verify blocks
        self.assertEqual(self.kv_buffer.blocks[offset1].capacity, 10)
        self.assertEqual(self.kv_buffer.blocks[offset2].capacity, 20)
        self.assertEqual(self.kv_buffer.blocks[offset3].capacity, 30)
        
        # Free middle block
        self.kv_buffer.free(offset2)
        
        # Allocate a block that fits in the freed space
        offset4 = self.kv_buffer.allocate(15)
        self.assertEqual(offset4, offset2)
        
        # Allocate a block that doesn't fit in the freed space
        offset5 = self.kv_buffer.allocate(25)
        self.assertEqual(offset5, 60)

    def test_fragmentation_and_cleaning(self):
        # Create a deliberate scenario with adjacent free blocks
        # Allocate consecutive blocks of the same size
        offsets = []
        block_size = 10
        num_blocks = 5
        for i in range(num_blocks):
            offsets.append(self.kv_buffer.allocate(block_size))
        
        # Verify blocks are allocated consecutively
        for i in range(1, num_blocks):
            self.assertEqual(offsets[i], offsets[i-1] + block_size)
        
        # Free adjacent blocks to test fragmentation cleaning
        self.kv_buffer.free(offsets[1])  # Free block at offset 10
        self.kv_buffer.free(offsets[2])  # Free block at offset 20 (adjacent to previous)
        self.kv_buffer.free(offsets[4])  # Free block at offset 40 (non-adjacent)
        
        # Check free slots before cleaning
        # Should have 4 free slots: the 3 freed blocks plus the remaining space after all allocations
        self.assertEqual(len(self.kv_buffer.free_slots), 2)
        
        # Record the state before cleaning
        free_slots_before = sorted(self.kv_buffer.free_slots.copy())
        blocks_before = {k: v for k, v in self.kv_buffer.blocks.items()}
        
        # Record the largest free block capacity before cleaning
        largest_before = max(blocks_before[offset].capacity for offset in free_slots_before)
        self.assertEqual(largest_before, 60)  # Should be 50, not 10
        
        # Explicitly call the clean fragment method
        self.kv_buffer._clean_fragment()
        
        # After cleaning, adjacent free blocks should be merged:
        # - offsets[1](10-19) and offsets[2](20-29) should merge
        # - offsets[4](40-49) and remaining space(50-99) should merge
        # So instead of 4 free slots, we should now have 2
        self.assertEqual(len(self.kv_buffer.free_slots), 2)
        
        # Verify total free space is preserved
        total_free_before = sum(blocks_before[offset].capacity for offset in free_slots_before)
        total_free_after = sum(self.kv_buffer.blocks[offset].capacity for offset in self.kv_buffer.free_slots)
        self.assertEqual(total_free_before, total_free_after)
        
        # Find the merged blocks and verify their sizes
        merged_blocks = []
        for offset in self.kv_buffer.free_slots:
            merged_blocks.append((offset, self.kv_buffer.blocks[offset].capacity))
        
        # Sort merged blocks by offset for consistent verification
        merged_blocks.sort()
        
        # Verify the first merged block (10-29) has capacity of 2*block_size
        self.assertEqual(merged_blocks[0][1], 2 * block_size)
        
        # Verify the second merged block (40-99) has capacity of block_size + remaining_space
        self.assertEqual(merged_blocks[1][1], block_size + self.cache_size - num_blocks * block_size)
        
        # Try to allocate a block larger than a single original block but fits in the first merged block
        larger_block_size = block_size + 5  # 15, which is between 10 and 20
        new_offset = self.kv_buffer.allocate(larger_block_size)
        
        # Verify it was allocated in the first merged block's space (10-29)
        self.assertEqual(new_offset, merged_blocks[0][0])
        
        # Verify the allocated block has correct length
        self.assertEqual(self.kv_buffer.blocks[new_offset].length, larger_block_size)

    def test_set_and_get_item(self):
        # Create a test tensor to set
        test_data = torch.ones([10, 2, 3, 4], dtype=torch.float32)
        
        # Set the item
        offset = self.kv_buffer.set_item(test_data)
        
        # Get the item back
        retrieved_data = self.kv_buffer.get_item(offset)
        
        # Verify data
        self.assertTrue(torch.all(torch.eq(retrieved_data, test_data)))
        
        # Free the block
        self.kv_buffer.free(offset)

    def test_data_ptr(self):
        # Allocate a block
        offset = self.kv_buffer.allocate(10)
        
        # Get pointer to the data
        ptr = self.kv_buffer.data_ptr(offset)
        
        # Should return a valid pointer (non-zero)
        self.assertNotEqual(ptr, 0)
        
        # Free the block
        self.kv_buffer.free(offset)

    def test_fragment_coalescence(self):
        """
        Test specifically for merging adjacent free blocks during fragmentation cleanup.
        This test creates a situation with guaranteed adjacent free blocks.
        """
        # Start with a clean buffer
        self.kv_buffer = KVBuffer(self.cache)
        
        # Allocate 5 blocks of size 10 each
        blocks = []
        for i in range(5):
            blocks.append(self.kv_buffer.allocate(10))
        
        # Free alternate blocks to create a fragmented buffer with adjacent free blocks
        self.kv_buffer.free(blocks[0])  # Free 0-9
        self.kv_buffer.free(blocks[1])  # Free 10-19, adjacent to previous
        self.kv_buffer.free(blocks[3])  # Free 30-39, not adjacent to previous free block
        
        # Verify initial state
        # Should have 4 free slots: blocks[0], blocks[1], blocks[3], and the remaining space after all allocations
        self.assertEqual(len(self.kv_buffer.free_slots), 3)
        
        # Store the total free space before cleaning
        total_free_before = sum(self.kv_buffer.blocks[offset].capacity for offset in self.kv_buffer.free_slots)
        
        # Find the largest free block before cleaning - should be the remaining space
        largest_before = max(self.kv_buffer.blocks[offset].capacity for offset in self.kv_buffer.free_slots)
        remaining_space = self.cache_size - (5 * 10)  # 5 blocks of size 10 = 50, remaining should be 50
        self.assertEqual(largest_before, remaining_space)
        
        # Run fragment cleaning
        self.kv_buffer._clean_fragment()
        
        # After cleaning:
        # - blocks[0](0-9) and blocks[1](10-19) should be merged
        # - blocks[3](30-39) remains separate (not adjacent to 50-99 due to blocks[4] at 40-49)
        # - remaining space(50-99) remains separate
        # So instead of 4 free slots, we should now have 3
        self.assertEqual(len(self.kv_buffer.free_slots), 3)
        
        # Verify that the total free space is preserved
        total_free_after = sum(self.kv_buffer.blocks[offset].capacity for offset in self.kv_buffer.free_slots)
        self.assertEqual(total_free_before, total_free_after)
        
        # Find the merged blocks and verify their sizes
        merged_blocks = []
        for offset in self.kv_buffer.free_slots:
            merged_blocks.append((offset, self.kv_buffer.blocks[offset].capacity))
            
        # Sort merged blocks by offset for consistent verification
        merged_blocks.sort()
        
        # Verify the first merged block (0-19) has capacity 20
        self.assertEqual(merged_blocks[0][1], 20)
        
        # Verify the other free blocks remain unchanged
        self.assertEqual(len(merged_blocks), 3)
        
        # Try to allocate a block of size 15, which wouldn't fit in any individual block before merging
        # but should fit in the first merged block now
        new_offset = self.kv_buffer.allocate(15)
        self.assertEqual(new_offset, merged_blocks[0][0])
        
        # Verify the allocated block has correct length
        self.assertEqual(self.kv_buffer.blocks[new_offset].length, 15)

    def test_stress_allocation_deallocation(self):
        """
        Stress test simulating production environment with frequent allocations and deallocations 
        of different sizes. Validates that memory fragmentation is handled properly.
        """
        # Create a larger buffer for stress testing
        large_cache = torch.zeros([1000, 2, 3, 4], dtype=torch.float32)
        buffer = KVBuffer(large_cache)
        
        # Tracking allocated blocks
        allocated_blocks = {}
        allocation_sizes = list(range(5, 51, 5))  # Sizes from 5 to 50 in steps of 5
        total_allocated = 0
        
        # Simulate 500 operations (allocations and deallocations)
        import random
        random.seed(42)  # For reproducibility
        
        for i in range(500):
            # Decide whether to allocate or deallocate
            if len(allocated_blocks) < 10 or random.random() < 0.7:  # 70% chance to allocate
                # Allocate a random size
                size = random.choice(allocation_sizes)
                try:
                    offset = buffer.allocate(size)
                    allocated_blocks[offset] = size
                    total_allocated += size
                except Exception:
                    # If allocation fails, trigger fragment cleaning and try again
                    buffer._clean_fragment()
                    try:
                        offset = buffer.allocate(size)
                        allocated_blocks[offset] = size
                        total_allocated += size
                    except Exception:
                        # Even after cleaning, allocation might fail if we're out of space
                        pass
            elif allocated_blocks:  # Deallocate if we have blocks
                # Choose a random block to free
                offset = random.choice(list(allocated_blocks.keys()))
                size = allocated_blocks.pop(offset)
                buffer.free(offset)
                total_allocated -= size
        
        # Verify final state
        # 1. Sum of all allocated blocks' sizes should match our tracking
        actual_allocated = large_cache.shape[0] - sum(buffer.blocks[offset].capacity for offset in buffer.free_slots)
        self.assertEqual(actual_allocated, total_allocated)
        
        # 2. No overlapping blocks
        all_offsets = sorted(list(buffer.blocks.keys()))
        for i in range(len(all_offsets) - 1):
            curr_offset = all_offsets[i]
            next_offset = all_offsets[i + 1]
            print(f"curr_offset: {curr_offset}, next_offset: {next_offset}, capacity: {buffer.blocks[curr_offset].capacity}")
            self.assertEqual(curr_offset + buffer.blocks[curr_offset].capacity, next_offset)
        
        # 3. Final cleanup should work without errors
        buffer._clean_fragment()
        
        # 4. After cleanup, we should have at most one free block per contiguous region
        free_offsets = sorted(buffer.free_slots)
        for i in range(len(free_offsets) - 1):
            curr_offset = free_offsets[i]
            next_offset = free_offsets[i + 1]
            # Verify non-adjacent free blocks
            self.assertNotEqual(curr_offset + buffer.blocks[curr_offset].capacity, next_offset)

    def test_block_size_allocation(self):
        """
        Test that the block size allocation strategy works correctly.
        This ensures that requested sizes are properly aligned to block sizes
        and that the best-fit allocation algorithm works as expected.
        """
        # Create a buffer with block size strategy enabled
        cache = torch.zeros([1000, 2, 3, 4], dtype=torch.float32)
        block_sizes = [10, 20, 50, 100, 200]
        buffer = KVBuffer(cache, block_sizes=block_sizes)
        
        # Verify block sizes are sorted
        self.assertEqual(buffer.block_sizes, sorted(block_sizes))
        
        # Test alignment to block size
        self.assertEqual(buffer._align_to_block_size(5), 10)   # Round up to 10
        self.assertEqual(buffer._align_to_block_size(10), 10)  # Exact match
        self.assertEqual(buffer._align_to_block_size(15), 20)  # Round up to 20
        self.assertEqual(buffer._align_to_block_size(51), 100) # Round up to 100
        self.assertEqual(buffer._align_to_block_size(201), 400) # Exceed max block size, round up to multiple of largest
        
        # Test allocation with block sizes
        # Allocate blocks of various sizes and verify they get properly aligned
        offset1 = buffer.allocate(5)  # Should allocate a block of capacity 10
        self.assertEqual(buffer.blocks[offset1].capacity, 10)
        self.assertEqual(buffer.blocks[offset1].length, 5)     # length should be the requested size
        
        offset2 = buffer.allocate(15)  # Should allocate a block of capacity 20
        self.assertEqual(buffer.blocks[offset2].capacity, 20)
        self.assertEqual(buffer.blocks[offset2].length, 15)    # length should be the requested size
        
        offset3 = buffer.allocate(51)  # Should allocate a block of capacity 100
        self.assertEqual(buffer.blocks[offset3].capacity, 100)
        self.assertEqual(buffer.blocks[offset3].length, 51)    # length should be the requested size
        
        # Test best-fit allocation
        # Free previous blocks
        buffer.free(offset1)
        buffer.free(offset2)
        buffer.free(offset3)
        
        # Create a situation with multiple free blocks of different sizes
        offset_big = buffer.allocate(200)    # Block of capacity 200
        offset_medium = buffer.allocate(100) # Block of capacity 100
        offset_small = buffer.allocate(50)   # Block of capacity 50
        
        # Free blocks in non-sequential order to create fragmentation
        buffer.free(offset_medium)  # Free the medium block
        
        # Now allocate a block that could fit in the medium block
        offset_new = buffer.allocate(75)  # Should use the medium block (capacity 100)
        self.assertEqual(offset_new, offset_medium)  # Should reuse the medium block's offset
        self.assertEqual(buffer.blocks[offset_new].capacity, 100)  # Capacity should be 100
        self.assertEqual(buffer.blocks[offset_new].length, 75)     # Length should be 75
        
        # Clean up
        buffer.free(offset_big)
        buffer.free(offset_small)
        buffer.free(offset_new)
        
        # Test without block size strategy
        buffer_no_blocks = KVBuffer(cache)
        self.assertIsNone(buffer_no_blocks.block_sizes)
        
        # Without block sizes, allocation should use exact requested sizes
        offset4 = buffer_no_blocks.allocate(17)
        self.assertEqual(buffer_no_blocks.blocks[offset4].length, 17)  # Should be exactly 17, not aligned

    def test_free_resets_length(self):
        # Create a test tensor to set
        test_data = torch.ones([10, 2, 3, 4], dtype=torch.float32)
        
        # Set the item which will allocate a block with the correct length
        offset = self.kv_buffer.set_item(test_data)
        
        # Verify the length is set correctly
        self.assertEqual(self.kv_buffer.blocks[offset].length, 10)
        
        # Record the capacity for later verification
        capacity = self.kv_buffer.blocks[offset].capacity
        
        # Free the block
        self.kv_buffer.free(offset)
        
        # Check if the block still exists (it might have been merged and deleted)
        if offset in self.kv_buffer.blocks:
            # Verify the length has been reset to 0
            self.assertEqual(self.kv_buffer.blocks[offset].length, 0)
        
        # Verify a block with this capacity appears in stats as a free block
        # This works even if the original block was merged
        stats = self.kv_buffer.stats()
        free_block_sizes = stats.available_block_sizes
        self.assertTrue(any(size >= capacity for size in free_block_sizes), 
                       f"No free block of at least size {capacity} found. Free sizes: {free_block_sizes}")

    def test_free_auto_merge_adjacent_blocks(self):
        """
        Test that the free method automatically merges adjacent blocks.
        This tests the new functionality added to immediately merge adjacent blocks during free.
        """
        # Start with a clean buffer
        self.kv_buffer = KVBuffer(self.cache)
        
        # Allocate 4 consecutive blocks of size 10
        offsets = []
        block_size = 10
        for i in range(4):
            offsets.append(self.kv_buffer.allocate(block_size))
        
        # Verify the blocks are allocated consecutively
        for i in range(1, 4):
            self.assertEqual(offsets[i], offsets[i-1] + block_size)
        
        # First test: free a block and its adjacent blocks in sequence
        # Free the second block
        self.kv_buffer.free(offsets[1])
        # Should have 2 free slots now: the freed block and the remaining space
        self.assertEqual(len(self.kv_buffer.free_slots), 2)
        
        # Free the third block, which is adjacent to the just freed second block
        # This should automatically merge with the second block
        self.kv_buffer.free(offsets[2])
        # Still 2 free slots: one merged block (covering offsets 1 and 2) and the remaining space
        self.assertEqual(len(self.kv_buffer.free_slots), 2)
        
        # Verify the merged block has a capacity of 2 * block_size
        merged_block_offset = offsets[1]  # The smaller offset of the merged blocks
        self.assertEqual(self.kv_buffer.blocks[merged_block_offset].capacity, 2 * block_size)
        
        # Free the first block, which is adjacent to the merged block
        # This should merge again
        self.kv_buffer.free(offsets[0])
        # Still 2 free slots: one larger merged block and the remaining space
        self.assertEqual(len(self.kv_buffer.free_slots), 2)
        
        # Verify the merged block now has a capacity of 3 * block_size
        # Now the merged block starts at offset 0
        self.assertEqual(self.kv_buffer.blocks[offsets[0]].capacity, 3 * block_size)
        
        # Second test: free blocks that would create a "sandwich" pattern
        # Reset with a new buffer
        self.kv_buffer = KVBuffer(self.cache)
        
        # Allocate 5 consecutive blocks
        offsets = []
        for i in range(5):
            offsets.append(self.kv_buffer.allocate(block_size))
        
        # Free blocks 1, 3, and then 2 (creating a situation where block 2 needs to merge with both sides)
        self.kv_buffer.free(offsets[1])
        self.kv_buffer.free(offsets[3])
        # Before freeing block 2, we have 3 free slots
        self.assertEqual(len(self.kv_buffer.free_slots), 3)
        
        # Free block 2, which should merge with both block 1 and block 3
        self.kv_buffer.free(offsets[2])
        # Now we should have 2 free slots: one large merged block and the remaining space
        self.assertEqual(len(self.kv_buffer.free_slots), 2)
        
        # The merged block should start at offset 1 and have capacity of 3 * block_size
        self.assertEqual(self.kv_buffer.blocks[offsets[1]].capacity, 3 * block_size)
        
        # Third test: Free blocks in reverse order to test merging logic
        # Reset with a new buffer
        self.kv_buffer = KVBuffer(self.cache)
        
        # Allocate 3 consecutive blocks
        offsets = []
        for i in range(3):
            offsets.append(self.kv_buffer.allocate(block_size))
        
        # Free blocks in reverse order: 2, 1, 0
        self.kv_buffer.free(offsets[2])  # Free 20-29
        # We now have one free blocks: 20-99
        self.assertEqual(len(self.kv_buffer.free_slots), 1)
        
        # Free the middle block
        self.kv_buffer.free(offsets[1])  # Free 10-19, should merge with both 20-29 and 30-99
        # Still 2 free slots: 0-9 (allocated) and 10-99 (free)
        self.assertEqual(len(self.kv_buffer.free_slots), 1)
        
        # Verify the free block is now 10-99
        merged_block_size = self.cache_size - block_size  # 100 - 10 = 90
        self.assertEqual(self.kv_buffer.blocks[offsets[1]].capacity, merged_block_size)
        
        # Free the first block
        self.kv_buffer.free(offsets[0])  # Free 0-9, should merge with 10-99
        # Now only 1 free slot covering the entire buffer: 0-99
        self.assertEqual(len(self.kv_buffer.free_slots), 1)
        
        # Verify the merged block now has a capacity equal to the entire cache
        self.assertEqual(self.kv_buffer.blocks[offsets[0]].capacity, self.cache_size)


if __name__ == '__main__':
    unittest.main()