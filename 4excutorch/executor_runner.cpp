/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run ExecuTorch model files that only use operators that
 * are covered by the portable kernels, with possible delegate to the
 * test_backend_compiler_lib.
 *
 * It sets all input tensor data to ones, and assumes that the outputs are
 * all fp32 tensors.
 */

#include <iostream>
#include <memory>
#include <gflags/gflags.h>
#include <vector>
#include <cstring> // For memcpy
#include <unistd.h>
#include <sstream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <errno.h>
#include <cstdlib>
#include <sys/time.h>
#include <string>
#include <sstream>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/extension/runner_util/inputs_portable.h>
#include <executorch/runtime/core/evalue.h>

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

struct r_tm{
	float *cost;
};

long long get_timestamp(void) {
    long long tmp;
    struct timeval tv;

    gettimeofday(&tv, NULL);
    tmp = tv.tv_sec;
    tmp = tmp * 1000 * 1000;
    tmp = tmp + tv.tv_usec;
    return tmp;
}


int main(int argc, char** argv) {
  runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  

  while (true) {
    volatile struct r_tm rtm;
    
    uint8_t result_new[8] = {0}; // Initialize all elements to 0
    result_new[0] = 0x55;
    result_new[1] = 0xaa;
    result_new[3] = 0x00;
    rtm.cost = (float *)&result_new[4];
    // Create a loader to get the data of the program file. There are other
    // DataLoaders that use mmap() or point to data that's already in memory, and
    // users can create their own DataLoaders to load from arbitrary sources.
    std::cout << "Step 1: Creating data loader" << std::endl;
    const char* model_path = FLAGS_model_path.c_str();
    Result<FileDataLoader> loader = FileDataLoader::from(model_path);
    if (!loader.ok()) {
      ET_LOG(Error, "FileDataLoader::from() failed: 0x%" PRIx32, (uint32_t)loader.error());
      sleep(1);
      continue;
    }
    std::cout << "Data loader created successfully" << std::endl;

    // Parse the program file. This is immutable, and can also be reused between
    // multiple execution invocations across multiple threads.
    std::cout << "Step 2: Parsing the program file" << std::endl;
    Result<Program> program = Program::load(&loader.get());
    if (!program.ok()) {
      ET_LOG(Error, "Failed to parse model file %s", model_path);
      sleep(1);
      continue;
    }
    ET_LOG(Info, "Model file %s is loaded.", model_path);
    std::cout << "Program file parsed successfully" << std::endl;

    // Use the first method in the program.
    std::cout << "Step 3: Using the first method in the program" << std::endl;
    const char* method_name = nullptr;
    {
      const auto method_name_result = program->get_method_name(0);
      if (!method_name_result.ok()) {
        ET_LOG(Error, "Program has no methods");
        sleep(1);
        continue;
      }
      method_name = *method_name_result;
    }
    ET_LOG(Info, "Using method %s", method_name);
    std::cout << "Method " << method_name << " selected" << std::endl;

    // MethodMeta describes the memory requirements of the method.
    std::cout << "Step 4: Getting MethodMeta" << std::endl;
    Result<MethodMeta> method_meta = program->method_meta(method_name);
    if (!method_meta.ok()) {
      ET_LOG(Error, "Failed to get method_meta for %s: 0x%" PRIx32, method_name, (uint32_t)method_meta.error());
      sleep(1);
      continue;
    }
    std::cout << "MethodMeta obtained successfully" << std::endl;

    // The runtime does not use malloc/new; it allocates all memory using the
    // MemoryManger provided by the client. Clients are responsible for allocating
    // the memory ahead of time, or providing MemoryAllocator subclasses that can
    // do it dynamically.
    std::cout << "Step 5: Setting up MemoryAllocator and planned buffers" << std::endl;
    MemoryAllocator method_allocator{
        MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

    // The memory-planned buffers will back the mutable tensors used by the
    // method. The sizes of these buffers were determined ahead of time during the
    // memory-planning pasees.
    std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
    std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
    size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
      size_t buffer_size = static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
      ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
      planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
      planned_spans.push_back({planned_buffers.back().get(), buffer_size});
    }
    HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});
    std::cout << "MemoryAllocator and planned buffers setup complete" << std::endl;

    // Assemble all of the allocators into the MemoryManager that the Executor will use.
    std::cout << "Step 6: Assembling all allocators into MemoryManager" << std::endl;
    MemoryManager memory_manager(&method_allocator, &planned_memory);

    // Load the method from the program, using the provided allocators. Running
    // the method can mutate the memory-planned buffers, so the method should only
    // be used by a single thread at at time, but it can be reused.
    std::cout << "Step 7: Loading the method from the program" << std::endl;
    Result<Method> method = program->load_method(method_name, &memory_manager);
    if (!method.ok()) {
      ET_LOG(Error, "Loading of method %s failed with status 0x%" PRIx32, method_name, (uint32_t)method.error());
      sleep(1);
      continue;
    }
    ET_LOG(Info, "Method loaded.");
    std::cout << "Method loaded successfully" << std::endl;

    // variable owns the allocated memory and must live past the last call to
    // `execute()`.
    std::cout << "Step 8: Allocating input tensors" << std::endl;
    auto inputs = util::prepare_input_tensors(*method);
    if (!inputs.ok()) {
      ET_LOG(Error, "Could not prepare inputs: 0x%" PRIx32, (uint32_t)inputs.error());
      sleep(1);
      continue;
    }
    ET_LOG(Info, "Inputs prepared.");
    std::cout << "Input tensors allocated and prepared" << std::endl;

    // Run the model.
    std::cout << "Step 9: Running the model" << std::endl;
    long long start_time = get_timestamp();
    Error status = method->execute();
    if (status != Error::Ok) {
      ET_LOG(Error, "Execution of method %s failed with status 0x%" PRIx32, method_name, (uint32_t)status);
      sleep(1);
      continue;
    }
    long long end_time = get_timestamp();
    long long cost_time = end_time - start_time;
    *rtm.cost = (float)(cost_time / 1000.0);
    ET_LOG(Info, "Model executed successfully.");
    std::cout << "Model executed successfully" << std::endl;

    // Print the outputs.
    std::cout << "Step 10: Printing the outputs" << std::endl;
    std::vector<EValue> outputs(method->outputs_size());
    ET_LOG(Info, "%zu outputs: ", outputs.size());
    status = method->get_outputs(outputs.data(), outputs.size());
    if (status != Error::Ok) {
    ET_LOG(Error, "Failed to get outputs with status 0x%" PRIx32, (uint32_t)status);
    sleep(1);
    continue;
    }
    std::cout << torch::executor::util::evalue_edge_items(100);
    for (int i = 0; i < outputs.size(); ++i) {
    std::cout << "Output " << i << ": " << outputs[i] << std::endl;
    }
    std::cout << "Outputs printed successfully" << std::endl;

    // 提取浮点数并发送到串口
    std::cout << "Step 11: Extracting floats and sending to UART" << std::endl;
    float result[2] = {0}; // 用于存储模型输出的两个浮点数

    // 假设 outputs[0] 包含 tensor 数据，并且 tensor 包含两个浮点数
    if (outputs[0].isTensor()) {
    const exec_aten::Tensor& tensor = outputs[0].toTensor();
    auto data_ptr = tensor.data_ptr<float>(); // 获取数据指针
    result[0] = data_ptr[0]; // 赋值给 result[0]
    result[1] = data_ptr[1]; // 赋值给 result[1]
    } else {
    std::cerr << "Output 0 is not a tensor." << std::endl;
    }

    // 打印提取的浮点数
    std::cout << "Extracted floats: [" << result[0] << ", " << result[1] << "]" << std::endl;



    int ret=1;

    if (ret >= 0) {
        result_new[2] = 0x00; // Operation successful
        if (result[0] > result[1]) {
            std::cout << "verify-result[0]=" << result[0] << std::endl;
            std::cout << "verify-result[1]=" << result[1] << std::endl;
            result_new[3] = 0x00;
        } else {
            result_new[3] = 0x01;
        }
            // 发送数据到串口
        std::cout << "Sending data to UART: ";
        write(fd, result_new, 8); // Send 8 bytes
        std::cout << "Floats sent to UART successfully" << std::endl;
    } else if (ret < 0) {
        result_new[2] = 0x01; // Operation failed
        write(fd, result_new, 4); // Send 4 bytes
    }
    
    







    // 延时一段时间，避免CPU占用率过高
    sleep(1); // 延时1秒，可以根据需要调整
  }

  return 0;
}
