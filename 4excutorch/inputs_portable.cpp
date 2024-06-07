#include <executorch/extension/runner_util/inputs.h>

#include <algorithm>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/log.h>
#include <iostream>
#include <random>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <errno.h>
#include <cstdlib>
#include <sys/time.h>
#include <string>
#include <sstream>

int fd;


namespace torch {
namespace executor {
namespace util {
namespace internal {

// Globals
namespace {
uint8_t result_new[8] = {0}; // 55 AA  f null   f:0 su 1 fail 2 data
}  // namespace

union r_dat {
    float fdat;
    uint8_t udat[4];
};
struct r_tm {
    float *cost;
};

// 获取时间戳函数
long long get_timestamp(void) {
    long long tmp;
    struct timeval tv;

    gettimeofday(&tv, NULL);
    tmp = tv.tv_sec;
    tmp = tmp * 1000 * 1000;
    tmp = tmp + tv.tv_usec;
    return tmp;
}

// 设置串口属性
int set_interface_attribs(int fd, int speed) {
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        return -1;
    }

    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_oflag &= ~OPOST;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        return -1;
    }

    return 0;
}

// 初始化串口
bool setup() {
    // 打开串口
    fd = open("/dev/ttyS7", O_RDWR | O_NOCTTY);
    if (fd == -1) {
        perror("open_port: Unable to open /dev/ttyS7");
        return false;
    }
    if (set_interface_attribs(fd, B115200) == -1) {
        close(fd);
        return false;
    }
    sleep(1);
    return true;
}

int read_serial_data(float* test_data, int max_len) {
    std::stringstream ss;
    volatile struct r_tm rtm;
    volatile union r_dat rdat;
    uint8_t dat;
    int len = 0, index = 0, datalen = 0;
    int status = 0;
    bool flag = false;
    int ret;
    result_new[0] = 0x55;
    result_new[1] = 0xaa;
    result_new[3] = 0x00;
    rtm.cost = (float *)&result_new[4];

    while (true) {
        if ((ret = read(fd, &dat, 1)) < 0) {
            len = 0;
            index = 0;
            datalen = 0;
            status = 0;
            return -1; // 读取错误
        }
        switch (status) {
            case 0:
                if (dat == 0xaa) {
                    status = 1;
                }
                break;
            case 1:
                if (dat == 0x55) {
                    status = 2;
                    index = 0;
                } else {
                    status = 0;
                }
                break;
            case 2:
                if (index == 0) {
                    datalen = dat;
                    index = 1;
                } else {
                    datalen |= dat << 8;
                    status = 3;
                    len = 0;
                    index = 0;
                }
                break;
            case 3:
                rdat.udat[index++] = dat;
                if (index >= 4) {
                    index = 0;
                    test_data[len++] = rdat.fdat;
                    
                }
                if (len >= datalen) {
                    flag = true;
                    status = 0;
                } else if (len >= max_len) {
                    status = 0;
                }
                break;
            default:
                status = 0;
                break;
        }

        if (flag) {
            flag = false;
            long long start_time = get_timestamp();
            long long end_time = get_timestamp();
            long long cost_time = end_time - start_time;
            *rtm.cost = (float)(cost_time / 1000.0);
            tcflush(fd, TCOFLUSH); // 清空发送缓冲区
            return len; // 返回读取的数据长度
        } else if (ret < 0) {
            result_new[2] = 0x01;
            tcflush(fd, TCOFLUSH); // 清空发送缓冲区
            write(fd, result_new, 4);
            return -1; // 读取错误
        }
    }
}

namespace {

/**
 * Sets all elements of a tensor to 1.
 */

// 从 test_data 数组中填充张量
Error fill_with_test_data(torch::executor::Tensor tensor, float* test_data, int data_len) {
    // 检查张量的大小是否与 test_data 数组的大小匹配
    if (tensor.numel() != data_len) {
        std::cerr << "Tensor size does not match test_data size." << std::endl;
        return Error::InvalidArgument;
    }

#define FILL_CASE(T, n)                                \
  case (torch::executor::ScalarType::n):                         \
    std::copy(                                         \
        test_data,                                     \
        test_data + data_len,                          \
        tensor.mutable_data_ptr<T>());                 \
    break;

    switch (tensor.scalar_type()) {
        ET_FORALL_REAL_TYPES_AND(Bool, FILL_CASE)
        default:
            std::cerr << "Unsupported scalar type " << (int)tensor.scalar_type() << std::endl;
            return Error::InvalidArgument;
    }
    return Error::Ok;
}
} // namespace

Error fill_and_set_input(
    Method& method,
    TensorInfo& tensor_meta,
    size_t input_index,
    void* data_ptr) {
    // 初始化串口
    if (!setup()) {
        std::cerr << "Failed to setup serial port." << std::endl;
        return Error::InvalidArgument;
    }

    float test_data[1250] = {0}; // 假设最大数据长度为1250

    // 持续读取数据直到成功
    int data_len;
    do {
        data_len = read_serial_data(test_data, 1250);
        if (data_len <= 0) {
            std::cerr << "Error reading data from serial port. Retrying..." << std::endl;
        }
    } while (data_len <= 0);

    TensorImpl impl = TensorImpl(
        tensor_meta.scalar_type(),
        /*dim=*/tensor_meta.sizes().size(),
        // These const pointers will not be modified because we never resize this
        // short-lived TensorImpl. It only exists so that set_input() can verify
        // that the shape is correct; the Method manages its own sizes and
        // dim_order arrays for the input.
        const_cast<TensorImpl::SizesType*>(tensor_meta.sizes().data()),
        data_ptr,
        const_cast<TensorImpl::DimOrderType*>(tensor_meta.dim_order().data()));
    Tensor t(&impl);
    ET_CHECK_OK_OR_RETURN_ERROR(fill_with_test_data(t, test_data, data_len));
    return method.set_input(t, input_index);
}

} // namespace internal
} // namespace util
} // namespace executor
} // namespace torch