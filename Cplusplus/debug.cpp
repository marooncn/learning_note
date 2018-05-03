int maxLen(const std::string& str) {
    int max = -1;  // int max = 0
    for(auto element : str) 
       if(element.size() > max)
          max = element.size(); 
    return max;
 }
 // 上面函数的返回值一直为-1。因为max = -1与无符号整型element.size()比较时，自动转换为无符号类型，即2^16-1 (int设为16位)
 // signed与unsigned比较、运算时，signed转换为unsined。
 unsigned u = 10;
 int i = -42;
 std::cout<< u+i << std::endl;  // 输出2^16-32
 
