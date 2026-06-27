#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Tensor;

namespace {

constexpr int kNumBuffers = 3;
constexpr int kMaxWorldSize = 16;

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct Layout {
  size_t slot_bytes;
  size_t control;
  size_t scratch;
  size_t flags;
  size_t done;
  size_t arrive_copy;
  size_t arrive_done;
  size_t status;
  size_t bytes;
};

struct alignas(16) SequenceControl {
  uint64_t next_sequence;
  uint64_t reserved;
};

struct alignas(16) SequenceTicket {
  uint64_t base;
  uint64_t count;
};

Layout make_layout(size_t elems, int world_size, size_t element_size) {
  TVM_FFI_ICHECK_GT(elems, 0);
  TVM_FFI_ICHECK_GT(world_size, 0);
  TVM_FFI_ICHECK_LE(world_size, kMaxWorldSize);
  TVM_FFI_ICHECK(element_size == 2 || element_size == 4);

  Layout layout{};
  layout.slot_bytes = align_up(elems * element_size, 128);
  size_t offset = 0;
  layout.control = offset;
  offset += sizeof(SequenceControl);
  offset = align_up(offset, 128);
  layout.scratch = offset;
  offset += kNumBuffers * static_cast<size_t>(world_size) * layout.slot_bytes;
  offset = align_up(offset, 128);
  layout.flags = offset;
  offset += kNumBuffers * static_cast<size_t>(world_size) * sizeof(int);
  offset = align_up(offset, 128);
  layout.done = offset;
  offset += kNumBuffers * static_cast<size_t>(world_size) * sizeof(int);
  offset = align_up(offset, 128);
  layout.arrive_copy = offset;
  offset += sizeof(int);
  offset = align_up(offset, 128);
  layout.arrive_done = offset;
  offset += sizeof(int);
  offset = align_up(offset, 128);
  layout.status = offset;
  offset += 4 * sizeof(int);
  layout.bytes = align_up(offset, 4096);
  return layout;
}

void check_cuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << operation << " failed: " << cudaGetErrorString(status);
}

struct KernelContext {
  char* local_base;
  const char* input;
  char* output;
  char* peer_bases[kMaxWorldSize];
  size_t elems;
  size_t element_size;
  size_t slot_bytes;
  size_t scratch;
  size_t flags;
  size_t done;
  size_t arrive_copy;
  size_t arrive_done;
  size_t status;
  int world_size;
  int rank;
  unsigned long long timeout_cycles;
};

__device__ bool wait_until_at_least(volatile int* value, int expected,
                                    unsigned long long timeout_cycles,
                                    int* status) {
  const unsigned long long start = clock64();
  while (*value < expected) {
    if (status[1] != 0) return false;
    if (clock64() - start > timeout_cycles) {
      atomicExch(status + 2, 1);
      return false;
    }
  }
  return true;
}

__device__ inline void copy_bytes(const char* source, char* destination,
                                  size_t bytes, size_t tid, size_t stride) {
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(source) |
                              reinterpret_cast<uintptr_t>(destination) | bytes;
  if ((alignment & (alignof(uint4) - 1)) == 0) {
    const uint4* source_vec = reinterpret_cast<const uint4*>(source);
    uint4* destination_vec = reinterpret_cast<uint4*>(destination);
    const size_t vector_count = bytes / sizeof(uint4);
    for (size_t index = tid; index < vector_count; index += stride) {
      destination_vec[index] = source_vec[index];
    }
    return;
  }
  for (size_t index = tid; index < bytes; index += stride) {
    destination[index] = source[index];
  }
}

__device__ bool wait_for_all(volatile int* values, int sequence,
                             KernelContext context, int timeout_stage) {
  int* status = reinterpret_cast<int*>(context.local_base + context.status);
  for (int source = 0; source < context.world_size; ++source) {
    if (!wait_until_at_least(values + source, sequence,
                             context.timeout_cycles, status)) {
      if (status[2] != 0) {
        atomicCAS(status + 3, 0,
                  timeout_stage * 100000 + sequence * 100 + source);
      }
      return false;
    }
  }
  return true;
}

__device__ void publish_sequence(KernelContext context, size_t offset,
                                 int sequence) {
  __threadfence_system();
  for (int destination = 0; destination < context.world_size; ++destination) {
    if (destination == context.rank) continue;
    *reinterpret_cast<volatile int*>(context.peer_bases[destination] + offset) =
        sequence;
  }
  __threadfence_system();
  *reinterpret_cast<volatile int*>(context.local_base + offset) = sequence;
}

__device__ void allgather_step(KernelContext context, int sequence) {
  const size_t tid =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const int buffer = sequence % kNumBuffers;
  int* status = reinterpret_cast<int*>(context.local_base + context.status);
  __shared__ int block_ready;

  volatile int* local_done = reinterpret_cast<volatile int*>(
      context.local_base + context.done +
      static_cast<size_t>(buffer) * context.world_size * sizeof(int));
  if (threadIdx.x == 0) {
    block_ready =
        sequence <= kNumBuffers ||
        wait_for_all(local_done, sequence - kNumBuffers, context, 1);
  }
  __syncthreads();
  if (!block_ready) return;

  const size_t payload_bytes = context.elems * context.element_size;
  const size_t target_offset =
      context.scratch +
      (static_cast<size_t>(buffer) * context.world_size + context.rank) *
          context.slot_bytes;
  for (int destination = 0; destination < context.world_size; ++destination) {
    copy_bytes(context.input,
               context.peer_bases[destination] + target_offset, payload_bytes,
               tid, stride);
  }

  __threadfence_system();
  __syncthreads();
  int* arrive_copy =
      reinterpret_cast<int*>(context.local_base + context.arrive_copy);
  volatile int* local_flags = reinterpret_cast<volatile int*>(
      context.local_base + context.flags +
      static_cast<size_t>(buffer) * context.world_size * sizeof(int));
  if (gridDim.x == 1) {
    if (threadIdx.x == 0) {
      const size_t flag_offset =
          context.flags +
          (static_cast<size_t>(buffer) * context.world_size + context.rank) *
              sizeof(int);
      publish_sequence(context, flag_offset, sequence);
    }
    __syncthreads();
  } else if (threadIdx.x == 0) {
    const int prior = atomicAdd(arrive_copy, 1);
    if (prior == gridDim.x - 1) {
      *arrive_copy = 0;
      const size_t flag_offset =
          context.flags +
          (static_cast<size_t>(buffer) * context.world_size + context.rank) *
              sizeof(int);
      publish_sequence(context, flag_offset, sequence);
    }
  }

  if (threadIdx.x == 0) {
    block_ready = wait_for_all(local_flags, sequence, context, 2);
  }
  __syncthreads();
  if (!block_ready) return;

  for (int source = 0; source < context.world_size; ++source) {
    const char* source_slot =
        context.local_base + context.scratch +
        (static_cast<size_t>(buffer) * context.world_size + source) *
            context.slot_bytes;
    copy_bytes(source_slot,
               context.output + static_cast<size_t>(source) * payload_bytes,
               payload_bytes, tid, stride);
  }

  __syncthreads();
  int* arrive_done =
      reinterpret_cast<int*>(context.local_base + context.arrive_done);
  if (gridDim.x == 1) {
    if (threadIdx.x == 0) {
      const size_t done_offset =
          context.done +
          (static_cast<size_t>(buffer) * context.world_size + context.rank) *
              sizeof(int);
      publish_sequence(context, done_offset, sequence);
    }
    __syncthreads();
    return;
  }
  if (threadIdx.x == 0) {
    const int prior = atomicAdd(arrive_done, 1);
    if (prior == gridDim.x - 1) {
      *arrive_done = 0;
      const size_t done_offset =
          context.done +
          (static_cast<size_t>(buffer) * context.world_size + context.rank) *
              sizeof(int);
      publish_sequence(context, done_offset, sequence);
    }
  }

  if (threadIdx.x == 0) {
    block_ready = wait_until_at_least(
        local_done + context.rank, sequence, context.timeout_cycles, status);
    if (!block_ready && status[2] != 0) {
      atomicCAS(status + 3, 0, 300000 + sequence * 100 + context.rank);
    }
  }
  __syncthreads();
}

__global__ void allgather_kernel(KernelContext context,
                                 const SequenceTicket* ticket) {
  const SequenceTicket reservation = *ticket;
  int* status = reinterpret_cast<int*>(context.local_base + context.status);
  if (reservation.base == 0 || reservation.count != 1 ||
      reservation.base > static_cast<uint64_t>(INT32_MAX)) {
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicExch(status + 1, 1);
    return;
  }
  allgather_step(context, static_cast<int>(reservation.base));
}

__global__ void initialize_control_kernel(SequenceControl* control) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    control->next_sequence = 1;
    control->reserved = 0;
  }
}

__global__ void reserve_sequences_kernel(SequenceControl* control,
                                         SequenceTicket* ticket,
                                         uint64_t count) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ticket->base =
        atomicAdd(reinterpret_cast<unsigned long long*>(&control->next_sequence),
                  static_cast<unsigned long long>(count));
    ticket->count = count;
  }
}

int64_t get_workspace_bytes(int64_t max_elems, int64_t world_size,
                            int64_t element_size) {
  return static_cast<int64_t>(
      make_layout(static_cast<size_t>(max_elems),
                  static_cast<int>(world_size),
                  static_cast<size_t>(element_size))
          .bytes);
}

void initialize_workspace(int64_t local_ptr, int64_t max_elems,
                          int64_t world_size, int64_t element_size,
                          int64_t stream_ptr) {
  TVM_FFI_ICHECK_NE(local_ptr, 0);
  Layout layout = make_layout(static_cast<size_t>(max_elems),
                              static_cast<int>(world_size),
                              static_cast<size_t>(element_size));
  char* local_base =
      reinterpret_cast<char*>(static_cast<uintptr_t>(local_ptr));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  check_cuda(cudaMemsetAsync(local_base, 0, layout.bytes, stream),
             "cudaMemsetAsync(all-gather workspace)");
  initialize_control_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<SequenceControl*>(local_base + layout.control));
  check_cuda(cudaGetLastError(), "initialize_control_kernel");
}

void reserve_sequences(int64_t local_ptr, Tensor ticket, int64_t count,
                       int64_t stream_ptr) {
  TVM_FFI_ICHECK_NE(local_ptr, 0);
  TVM_FFI_ICHECK_GT(count, 0);
  CHECK_INPUT_AND_TYPE(ticket, dl_uint64);
  TVM_FFI_ICHECK_GE(ticket.numel(), 2);
  char* local_base =
      reinterpret_cast<char*>(static_cast<uintptr_t>(local_ptr));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  reserve_sequences_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<SequenceControl*>(local_base),
      static_cast<SequenceTicket*>(ticket.data_ptr()),
      static_cast<uint64_t>(count));
  check_cuda(cudaGetLastError(), "reserve_sequences_kernel");
}

void launch_allgather(int64_t local_ptr, Array<int64_t> peer_ptrs,
                      Tensor input, Tensor output, int64_t elems,
                      int64_t max_elems, int64_t world_size, int64_t rank,
                      int64_t blocks, int64_t threads, int64_t dtype,
                      Tensor ticket, int64_t stream_ptr) {
  TVM_FFI_ICHECK_NE(local_ptr, 0);
  TVM_FFI_ICHECK_GT(elems, 0);
  TVM_FFI_ICHECK_LE(elems, max_elems);
  TVM_FFI_ICHECK_EQ(peer_ptrs.size(), static_cast<size_t>(world_size));
  TVM_FFI_ICHECK_GE(rank, 0);
  TVM_FFI_ICHECK_LT(rank, world_size);
  TVM_FFI_ICHECK_GT(blocks, 0);
  TVM_FFI_ICHECK_GT(threads, 0);
  TVM_FFI_ICHECK_LE(threads, 1024);
  TVM_FFI_ICHECK_EQ(input.numel(), elems);
  TVM_FFI_ICHECK_EQ(output.numel(), elems * world_size);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id);
  CHECK_INPUT_AND_TYPE(ticket, dl_uint64);
  TVM_FFI_ICHECK_GE(ticket.numel(), 2);
  TVM_FFI_ICHECK_EQ(ticket.device().device_id, input.device().device_id);

  size_t element_size = 0;
  if (dtype == 0) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(input.dtype()), float16_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), float16_code);
    element_size = 2;
  } else if (dtype == 1) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(input.dtype()), bfloat16_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), bfloat16_code);
    element_size = 2;
  } else if (dtype == 2) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(input.dtype()), float32_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), float32_code);
    element_size = 4;
  } else {
    TVM_FFI_ICHECK(false) << "unsupported dtype code " << dtype;
  }

  Layout layout = make_layout(static_cast<size_t>(max_elems),
                              static_cast<int>(world_size), element_size);
  KernelContext context{};
  context.local_base =
      reinterpret_cast<char*>(static_cast<uintptr_t>(local_ptr));
  context.input = static_cast<const char*>(input.data_ptr());
  context.output = static_cast<char*>(output.data_ptr());
  for (int peer = 0; peer < world_size; ++peer) {
    TVM_FFI_ICHECK_NE(peer_ptrs[peer], 0);
    context.peer_bases[peer] = reinterpret_cast<char*>(
        static_cast<uintptr_t>(peer_ptrs[peer]));
  }
  TVM_FFI_ICHECK_EQ(context.peer_bases[rank], context.local_base);
  context.elems = static_cast<size_t>(elems);
  context.element_size = element_size;
  context.slot_bytes = layout.slot_bytes;
  context.scratch = layout.scratch;
  context.flags = layout.flags;
  context.done = layout.done;
  context.arrive_copy = layout.arrive_copy;
  context.arrive_done = layout.arrive_done;
  context.status = layout.status;
  context.world_size = static_cast<int>(world_size);
  context.rank = static_cast<int>(rank);
  context.timeout_cycles = 30000000000ULL;

  ffi::CUDADeviceGuard guard(input.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  allgather_kernel<<<static_cast<int>(blocks), static_cast<int>(threads), 0,
                     stream>>>(
      context, static_cast<const SequenceTicket*>(ticket.data_ptr()));
  check_cuda(cudaGetLastError(), "allgather_kernel");
}

Array<int64_t> get_status(int64_t local_ptr, int64_t max_elems,
                          int64_t world_size, int64_t element_size) {
  TVM_FFI_ICHECK_NE(local_ptr, 0);
  Layout layout = make_layout(static_cast<size_t>(max_elems),
                              static_cast<int>(world_size),
                              static_cast<size_t>(element_size));
  int host_status[4]{};
  char* local_base =
      reinterpret_cast<char*>(static_cast<uintptr_t>(local_ptr));
  check_cuda(cudaMemcpy(host_status, local_base + layout.status,
                        sizeof(host_status), cudaMemcpyDeviceToHost),
             "cudaMemcpy(all-gather status)");
  return Array(std::vector<int64_t>{host_status[0], host_status[1],
                                    host_status[2], host_status[3]});
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_workspace_bytes, get_workspace_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(initialize_workspace, initialize_workspace);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reserve_sequences, reserve_sequences);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_allgather, launch_allgather);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_status, get_status);
