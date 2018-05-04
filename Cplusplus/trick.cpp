// 判断两浮点数是否相等，不要用 a == b
double tolerance = 1e-1;
bool isEqual = if(fabs(a-b) < tolerance) ? true : false;

// 判断整数是否为奇数，不要用 x%2 != 0，因为x可能为负数
bool isOdd = if(x%2 == 1) ? true : false;
// 或者可以通过位运算进行判别，不能用x&1==0判别整数是否为偶数，可以用!(x&1)
if(x&1) ? printf("odd"); : printf("even");

