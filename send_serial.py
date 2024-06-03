import serial
import numpy as np
import argparse

def read_floats_from_file(filename, size):
    """
    从文件中读取浮点数，每行一个浮点数。
    
    :param filename: 包含浮点数的文件名。
    :param size: 要读取的浮点数的数量。
    :return: 包含浮点数的numpy数组。
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = np.array([float(line.strip()) for line in lines[:size]], dtype=np.float32)
    return data

def send_data_to_serial(data, port, baudrate):
    """
    通过串口发送数据。

    :param data: 要发送的数据，应该是一个numpy数组。
    :param port: 串口端口名。
    :param baudrate: 串口波特率。
    """
    ser = serial.Serial(port, baudrate)
    
    try:
        # 确保数据是字节形式
        byte_data = data.tobytes()
        
        # 发送数据
        bytes_written = ser.write(byte_data)
        
        # 检查发送的数据量是否正确
        if bytes_written == len(byte_data):
            print("Data sent successfully.")
        else:
            print(f"Error: Only {bytes_written} bytes out of {len(byte_data)} were sent.")
        
    except Exception as e:
        print(f"Error sending data: {e}")
    finally:
        ser.close()

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Send data from a file to a serial port.")
    parser.add_argument('--com', type=str, default='com4', help='Serial port (default: com4)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--size', type=int, default=1250, help='Number of floats to read (default: 1250)')
    parser.add_argument('filename', type=str, help='The name of the file to read data from')

    # 解析命令行参数
    args = parser.parse_args()

    # 从文件读取数据
    float_data = read_floats_from_file(args.filename, args.size)

    # 发送数据到串口
    send_data_to_serial(float_data, args.com, args.baudrate)

