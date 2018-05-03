// 判断两浮点数是否相等，不要用 a == b
double tolerance = 1e-1;
bool isEqual = if(fabs(a-b) < tolerance) ? true : false;

// 判断整数是否为奇数，不要用 x%2 != 0，因为x可能为负数
bool isOdd = if(x%2 == 1) ? true : false;

// 在处理字符串问题如统计每个字符出现的次数时，用char值作为数组下标时要考虑到char可能为负数，因此先强制转换为unsigned char再用作下标
