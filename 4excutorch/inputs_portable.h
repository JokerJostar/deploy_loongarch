#ifndef INPUTS_PORTABLE_H
#define INPUTS_PORTABLE_H

#include <unistd.h> // For close
#include <termios.h> // For termios

extern int fd; // 声明外部文件描述符

// 设置串口属性
int set_interface_attribs(int fd, int speed);

// 初始化串口
bool setup();

#endif // INPUTS_PORTABLE_H