// 判断两浮点数是否相等，不要用 a == b
double tolerance = 1e-1;
bool isEqual = if(fabs(a-b) < tolerance) ? true : false;

// 判断整数是否为奇数，不要用 x%2 != 0，因为x可能为负数
bool isOdd = if(x % 2 == 1) ? true : false;
// 或者可以通过位运算进行判别
(x & 1) ? printf("odd") : printf("even");

