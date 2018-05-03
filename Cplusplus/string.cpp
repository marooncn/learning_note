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

// 得到字符串中的最大回文数
#include <string>
using std::string;

void getPalindrome(const string &s, int a, int b, int &start, int &end) {
    string::size_type len = s.size();
    while(a>=0 && b<len && s[a]==s[b]) {
        if(b-a > end-start) {  // 更新
            start = a;
            end = b;
        }  
        a--,b++;
    }
}

class Solution {
public:
    string longestPalindrome(string s) {
        int start = 0, end = 0;
        for(int i=0; i<s.size()-1; i++) {
            getPalindrome(s, i, i, start, end);  // 回文数个数为奇数，如 a b c
            getPalindrome(s, i, i+1, start, end);  // 回文数个数为偶数，如 a b b c
        }
        return s.substr(start, end-start+1);     
    }
};
