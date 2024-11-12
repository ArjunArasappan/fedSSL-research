# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Partitioner class that works with Hugging Face Datasets."""

from abc import ABC, abstractmethod
from datasets import Dataset
import numpy as np
from typing import Optional

from flwr_datasets.partitioner import Partitioner

class Anchor_Partitioner(Partitioner):
    def __init__(self, partitions: int, num_anchors: int) -> None:
        super().__init__()
        
        self.num_anchors = num_anchors
        self.partitions = partitions
        
        self.calculated_indicies = False
        self.indicies = {}


    def load_partition(self, partition_id: int) -> Dataset:
        if self._dataset is None:
            raise ValueError("Dataset is not assigned.")
        
        
        if not self.calculated_indicies:
            
            shuffled_dataset = self._dataset.shuffle(seed=42)
            
            total_samples = len(shuffled_dataset)
            regular_partition_size = (total_samples - self.num_anchors) // self.partitions
            partition_sizes = [regular_partition_size] * self.partitions + [self.num_anchors]
            
            for i in range(len(partition_sizes)):
                if i >= (total_samples - self.num_anchors) % self.partitions:
                    break
                partition_sizes[i] += 1
                
            print("PARTITIONS: ", partition_sizes, sum(partition_sizes))
                
            cumulative_sizes = np.cumsum(partition_sizes)
            
            for id in range(len(partition_sizes)):
                
                start_idx = 0 if id == 0 else cumulative_sizes[id - 1]
                end_idx = cumulative_sizes[id]
                
                self.indicies[id] = (start_idx, end_idx)
                
            self.calculated_indicies = True
                
                
        start_idx, end_idx = self.indicies[partition_id]
        shuffled_dataset = self._dataset.shuffle(seed=42)
        return shuffled_dataset.select(range(start_idx, end_idx))


    @property
    def num_partitions(self) -> int:
        return self.partitions + 1
