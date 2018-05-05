// 判断字符c是否在字符串s中
bool isInString(char c, string s) {
   for(string::size_type i=0; i<s.size(); i++) 
     if(c == s[i])
        return true;
   return false;
}

// 将字符c转换为字符串s，类似地，可以将int转换为string
string char2String(char c) {
    stringstream ss;
    string s;
    ss << c;
    ss >> s;
    return s;
}
// 向定义好的string里添加char可以直接用append,即 string.append(size_t n, char c)


// 在处理字符串问题如统计每个字符出现的次数时，用char值作为数组下标时要考虑到char可能为负数，因此先强制转换为unsigned char再用作下标
