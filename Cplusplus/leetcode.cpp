22. Generate Parentheses
#include <vector>
#include <string>

using std::string; using std::vector;
//魔性的递归啊
class Solution {
public:
    vector<string> vec;
    void addParenthesis(string str, int ln, int rn)  
        {
            if (ln == 0 && rn == 0) vec.push_back(str);
            if (ln > 0) addParenthesis(str+"(", ln-1, rn+1);
            if (rn > 0) addParenthesis(str+")", ln, rn-1); }
    vector<string> generateParenthesis(int n) {
        addParenthesis("", n, 0);
        return vec;
    }
};


21. Merge Two Sorted Lists
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1 == NULL) return l2;
        if(l2 == NULL) return l1;
        ListNode dummy(0), *tail = &dummy;  //这种表示在链表问题中很常用，见第2题
        for(; l1!=NULL && l2!=NULL; tail=tail->next) {
            if(l1->val <= l2->val) {tail->next = l1; l1 = l1->next;}
            else {tail->next = l2; l2 = l2->next;}
        }
        tail->next = (l1==NULL ? l2 : l1);
        return dummy.next;
    }
};

递归解法：
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1 == NULL) return l2;
        if(l2 == NULL) return l1;
        if(l1->val <= l2->val) {l1->next = mergeTwoLists(l1->next, l2); return l1;}
        else {l2->next = mergeTwoLists(l1, l2->next); return l2;}
    }
};

20. Valid Parentheses
#include <string>
#include <stack>
using std::string;
using std::stack;
/* 用堆栈来实现括号匹配 */
class Solution {
public:
    bool isValid(string s) {
         stack<char> st;
         for(auto i : s) 
             if(!st.empty() && ((i==')' && st.top()=='(') || (i==']' && st.top()=='[') || (i=='}' && st.top()=='{')))
              // st.empty()判断栈是否为空，st.top()返回栈顶元素，st.pop()删除栈顶元素，st.push(item)将新元素压入栈顶
                 st.pop();
             else
                 st.push(i);       
        return st.empty();
    }
};


1. Two Sum
Solution 1
#include <iostream>
#include <vector>
using namespace std;
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int>::iterator it = nums.begin();
        for (int i = 0; (it+i) != nums.end(); i++) {
            int j=1;
            for (auto it2 = it+i; (it2+j) != nums.end(); j++)
                if (*(it+i) + *(it2+j) == target) {
                    indices.push_back(i);
                    indices.push_back(j); 
                    break; }
        }
        return indices;
    }
private:
    vector<int> indices;
};

Solution 2
#include <vector>
using std::vector;
#include <unordered_map>
using std::unordered_map;

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int, int> record;
        for (int i = 0; i != nums.size(); ++i) {
            auto found = record.find(nums[i]);
            if (found != record.end())
                return {found->second, i};
            record.emplace(target - nums[i], i);
        }
        return {-1, -1};
    }
};


2. Add Two Numbers
Solution 1
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {    
        ListNode header(0);
        ListNode* p = new ListNode(0);
        header.next = p; 
        while(l1 || l2) {
            if(l1) {
                p->val += l1->val;
                l1 = l1->next;
            }
            if(l2) {
                p->val += l2->val;
                l2 = l2->next;
            }
            if(l1 || l2 || p->val/10)
                p->next = new ListNode(p->val/10);
            p->val = p->val%10;
            if(p->next)
                p = p->next;
        }
        return header.next;
    }
};


Solution2
#include <cstddef>
#include <cstdlib>

class Solution {
public:
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        ListNode dummy(0), *tail = &dummy;
        for (div_t sum{0, 0}; sum.quot || l1 || l2; tail = tail->next) {
        // div为cstdlib库中规定的函数，div_t out = div(a,b)，out.quot为a/b，out.rem为a%b
            if (l1) { sum.quot += l1->val; l1 = l1->next; }
            if (l2) { sum.quot += l2->val; l2 = l2->next; }
            sum = div(sum.quot, 10);
            tail->next = new ListNode(sum.rem);
        }
        return dummy.next;
    }
};

Main
using namespace std;
ListNode *create_linkedlist(std::initializer_list<int> lst)
{
    auto iter = lst.begin();
    ListNode *head = lst.size() ? new ListNode(*iter++) : NULL;
    for (ListNode *cur=head; iter != lst.end(); cur=cur->next)
        cur->next = new ListNode(*iter++);
    return head;
}

int main()
{
    Solution s;
    ListNode *l1 = create_linkedlist({2,4,3});
    ListNode *l2 = create_linkedlist({5,6,4});
    ListNode *ret = s.addTwoNumbers(l1, l2);
    for (ListNode *cur = ret; cur; cur = cur->next)
        cout << cur->val << "->";
    cout << "\b\b  " << endl;

    return 0;
}


3. Longest Substring Without Repeating Characters
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
using std::string;
using std::vector;

bool isInString(char c, string s) {
   for(string::size_type i=0; i<s.size(); i++) 
     if(c == s[i])
        return true;
   return false;
}

// Also int2String
string char2String(char c) {
    stringstream ss;
    string s;
    ss << c;
    ss >> s;
    return s;
}

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
    vector<string> str;
    for (string::size_type i=0; i<s.size(); i++) {
       str.push_back(char2String(s[i]));
       for(string::size_type j=i+1; j<s.size(); j++) 
       if(!isInString(s[j], str[i]))
          str[i]+=s[j]; // or str[i].append(1, s[j]);
       else
          break;
    }
    int max = 0;  // Can't use int max = -1
    for(auto element : str) 
       if(element.size() > max)
          max = element.size(); 
    return max;
    
    }
};


5. Longest Palindromic Substring
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


6. ZigZag Conversion#include <string>
using std::string;

class Solution {
public:
    string convert(string s, int numRows) {
        if(numRows<2 || s.empty()) return s; 
        string result;
        vector<string> row(numRows);
        for(int i = 0; i < s.size(); i++) {
            int j = i%(numRows-1), k = i/(numRows-1);
            if((k & 1) == 0)// k为偶数，余数即为行数, &的运算优先级低于==,因此k&1的括号是必须的
                row[j] += s[i];
            else  // k为奇数，余数即为倒行数
                row[numRows-1-j].append(1, s[i]); 
       /*   if((k & 0x1) == 1) // k为奇数，余数即为倒行数 
                 row[numRows-1-j].append(1, s[i]);
               
            else  // k为偶数，余数即为行数
                 row[j] += s[i]; */
        }
        for(int i = 0; i < numRows; i++) 
            result.append(row[i]);
        return result; 
        }
};

7. Reverse Integer
#include <cmath>
#define MAX pow(2,31)
class Solution {
public:
    int reverse(int x) {
        int flag = 1;
        long result = 0;
        if(x < 0)
           { flag = -1; x = -x; }
        while(x > 0) {
            result = 10*result + x%10; 
            x /= 10; }
        if (result > MAX-1 || (flag && result > MAX))  // 注意这里判断溢出的方法
            return 0;
        return result*flag;
        
    }
};


8. String to Integer (atoi)
#include <string>
using std::string;
#include <cctype>
#include <cmath>
#define INT_MAX pow(2,31)
using std::size_t;

class Solution {
public:
    int myAtoi(string str) {
        string::size_type len = str.size();
        char tmp;
        for(int i=0; i<len/2; i++) {
            tmp = str[len-i-1];
            str[len-i-1] = str[i];
            str[i] = tmp;
        }
        while(str[len-1] == ' ') {
            len -= 1;
            if(len < 1) 
                return 0; }
        int flag = 1;
        if(str[len-1] == '-' && len > 1) {
            flag = -1;
            len -= 1; }
        else if(str[len-1] == '+' && len > 1) {
            flag = 1;
            len -= 1;
        }
        else if(!isdigit(str[len-1])) 
            return 0;
        size_t result = 0;   
        while(isdigit(str[len-1])) {
            result = 10*result + static_cast<int>(str[(len--)-1]-48);  // '0'--48;...;'9'--57
            if(flag > 0 && result > INT_MAX-1)  // 若把此判断加到循环外边，遇到非常大的整数导致溢出结果就会错误。
                return INT_MAX-1;
            else if(flag < 0 && result > INT_MAX)
                return flag*INT_MAX;
            }
        return flag*result; 
    }
};

9. Palindrome Number
class Solution {
public:
    bool isPalindrome(int x) {
        int result = 0;
        int tmp = x;
        if(tmp < 0)
            return false;
        for(; tmp > 0; tmp = tmp/10) {
            result = 10*result + tmp%10;
        }
        if(result == x)
            return true;
        else
            return false;
    }
};

11. Container With Most Water
#include <vector>
#include <Algorithm>
using std::vector;
using std::max; using std::min; 

class Solution {
public:
    int maxArea(vector<int>& height) {
        /* 暴力搜索法,时间复杂度O（n^2）：超时 
        vector<int>::size_type len = height.size();
        int maxVol = 0;
        for(int i=0; i<len; i++)
            for(int j=i+1; j<len; j++) {
                int vol = (j-i)*((height[i]>height[j]) ? height[j] : height[i]);
                if(vol > maxVol)
                    maxVol = vol;              
            } 
        
        return maxVol;     */
        // 以两端垂线围成的面积为初始容量，要想容量更大，需要向内寻找更长的边，因此逐渐向内移动较短的一边。这种方法的时间复杂度为O（n）
        int maxVol = 0;
        for(auto beg=height.begin(), end = height.end()-1; end > beg; ) {
             maxVol = max(maxVol, min(*beg, *end)*static_cast<int>(end-beg));
            if(*beg < *end)
                beg++;
            else
                end--;
        }
        return maxVol;
        /* 代码可以写到更简
        int maxVol = 0;
        for(auto beg=height.begin(), end = height.end()-1; end > beg; *beg<*end? beg++ : end--) 
             maxVol = max(maxVol, min(*beg, *end)*static_cast<int>(end-beg));
        return maxVol; */
    }
};

12. Integer to Roman
#include <map>
#include <string>
using std::string;
using std::map;

class Solution {
public:
    string intToRoman(int num) {
        map<int, string> mp = {{1, "I"},{4, "IV"}, {5, "V"}, {9, "IX"}, {10, "X"}, {40, "XL"}, {50, "L"}, 
                               {90, "XC"}, {100, "C"}, {400, "CD"}, {500, "D"}, {900, "CM"}, {1000, "M"}};
        
        string result;
        while(num > 0)
            for(auto beg = mp.rbegin(); beg != mp.rend(); beg++) {  // 逆向迭代
                if(num >= beg->first) {
                    result += beg->second;  // string定义后（默认值null）可直接通过+赋值
                    num -= beg->first;
                    break;
                }   
           }
        
     /*OR  for(auto beg = mp.rbegin(); beg != mp.rend(); beg++)  {
            while(num >= beg->first) {
                result += beg-second;
                num -= beg->first;
            }
        } */
        
       return result;     
    }
}; 

13. Roman to Integer
#include <map>
#include <string>
using std::map;
using std::string;

bool findElement(map<string, int> m, string str) {
    for(const auto &w : m)
        if(w.first == str) 
            return true;
    return false;
}

class Solution {
public:
    int romanToInt(string s) {
        int result = 0;
        map<string, int> mp = {{"I", 1},  {"IV", 4},  {"V", 5}, {"IX", 9}, {"X", 10}, {"XL", 40},{"L", 50}, 
                               {"XC", 90}, {"C", 100}, {"CD", 400}, {"D", 500}, {"CM", 900}, {"M", 1000}};
        auto len = s.size();
        for(int i = 0; i < len; i++) {
            string tmp;
            if(i+1 < len && findElement(mp, tmp = tmp+s[i]+s[i+1])) 
                // 或 tmp.append(s, i, 2)
                // 但写成 tmp = s[i]+s[i+1]会转换成数字，tmp += s[i]+s[i+1]会得到其它字符
                i++;
            else 
                tmp = s[i]; 
            
            result += mp[tmp];    
        }
        return result;

    }
};
/*
#include <string>
using std::string;

class Solution {
public:
    int romanToInt(string s) {
        int res = 0;
        for (auto iter = s.rbegin(); iter != s.rend(); ++iter)
            switch (*iter)
            {
                case 'I': res += res >= 5 ? -1 : 1; break;
                case 'V': res += 5; break;
                case 'X': res += 10 * (res >= 50 ? -1 : 1); break;
                case 'L': res += 50; break;
                case 'C': res += 100 * (res >= 500 ? -1 : 1); break;
                case 'D': res += 500; break;
                case 'M': res += 1000; break;
            }
        return res;
    }
};  */


14. Longest Common Prefix
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        std::size_t len = strs.size();
        if(len == 0)
            return "";
        string prefix;
        for(int i=0; i<strs[0].size(); i++) {
            for(int j=1; j<len; j++)
                if(strs[j].size() <= i || strs[j][i] != strs[0][i])
                    return prefix;
            prefix += strs[0][i];
        }         
        return prefix;
    }
};

15. 3Sum
#include <vector>
#include <algorithm>
using std::vector;
using std::sort;

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        if(nums.size() < 3)
            return result;
        sort(nums.begin(), nums.end());
        for(auto i=nums.begin(),last=nums.end(); i<last-2; i++) {
            auto j = i+1;
            if(i>nums.begin() && *i==*(i-1)) continue;
            auto k=last-1;
            while(j<k) {
                if(*i+*j+*k < 0) {
                    ++j;
                    while(*j==*(j-1) && j<k) j++;
                } else if(*i+*j+*k > 0) {
                    k--;
                    while(*k==*(k+1) && j<k) k--;
                } else {
                    result.push_back({*i, *j, *k});
                    j++; k--;
                    while(*j==*(j-1) && *k==*(k+1) && j<k) ++j;
                }
                }
            }
        return result;
    }
};  

16. 3Sum Closest
#include <vector>
#include <algorithm>
#include <cmath>

using std::vector;
using std::sort;
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        assert(nums.size() > 2);
        sort(nums.begin(), nums.end());
        int result, sum;
        for(auto i=nums.begin(); i<nums.end()-2; i++) {
            auto j = i+1;
            auto k = nums.end()-1;
            if(i==nums.begin()) result=*i+*j+*k;
            while(j < k) {
                sum = *i+*j+*k;
                if(abs(sum-target) < abs(result-target))
                    result = *i+*j+*k; 
                if(sum < target) 
                    j++;
                else if(sum > target)
                    k--;
                else
                    return result;
            }   
        }
        return result; 
    }
}; 

17. Letter Combinations of a Phone Number
#include <string>
#include <vector>
#include <map>

using std::string;
using std::vector;

class Solution {
public:
    vector<string> letterCombinations(string digits) {
        map<char,string> mapping = {{'2',"abc"}, {'3',"def"}, {'4', "ghi"}, {'5', "jkl"}, 
                                {'6', "mno"}, {'7', "pqrs"}, {'8', "tuv"},{'9', "wxyz"}};
        vector<string> result;
        if(digits=="") return result;  // 返回 []
        result = {""};  //这一步是必须的，否则第三层循环进不去
        for(auto digit:digits) {  //第一层循环，逐个输入的数字
            vector<string> tmp;
            for(auto chr:mapping[digit])   //第二层循环，逐个数字对应的字符
                for(const auto &pre:result)   //第三层循环，逐个叠加到前层
                    tmp.push_back(pre+chr);
            result = tmp; }   
        return result;
    }
};


18. 4Sum
#include <vector>
#include <algorithm>

using std::sort;

class Solution {
public:  
    vector<vector<int> > fourSum(vector<int> &num, int target) {  
        int n = num.size();  
        vector<vector<int> > ret;  
        if(n < 4) return ret;  
        vector<int> ivec(num);  
        sort(ivec.begin(), ivec.end());  
        unordered_map<int, vector<pair<int,int> > > pairs;  
        pairs.reserve(n*n);  // 重组存储，使得pairs可以保存n*n个元素且不必rehash
          
        //将n个数字的两两组合求和存进map 相同和的对在map中按照链表存储，即存在vector中 j从i+1开始  
        for(int i = 0; i < n; i++)  
        {  
            for(int j = i+1; j < n; j++)  
            {  
                pairs[ivec[i] + ivec[j]].push_back(make_pair(i,j));  
            }  
        }  
          
        for(int i = 0; i < n - 3; i++)  
        {  
            if(i > 0 && ivec[i] == ivec[i-1]) continue;  
            for(int j = i+1; j < n - 2; j++)  
            {  
                if(j > i+1 && ivec[j] == ivec[j-1]) continue;  
                if(pairs.find(target - ivec[i] - ivec[j]) != pairs.end())  
                {  
                    vector<pair<int, int> > sum2 = pairs[target - ivec[i] - ivec[j]];  
                    bool isFstpush = true;  
                    //将符合要求的不重复且下标递增的pair存进答案  
                    for(int k = 0; k < sum2.size(); k++)  
                    {  
                        if(sum2[k].first <= j) continue;  
                        //第一次存储的重复数字，或者不是重复数字，判断第三位是否相等  
                        if(isFstpush || (ret.back())[2] != ivec[sum2[k].first])  
                        {  
                           vector<int> tmp{ivec[i], ivec[j], ivec[sum2[k].first], ivec[sum2[k].second]};   
                           ret.push_back(tmp);  
                           isFstpush = false;  
                        }  
                    }  
                }  
            }  
        }  
          
        return ret;  
    }  
      
}; 

19. Remove Nth Node From End of List
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
 /* 删除链表结点， node1->next = node3; 或者 node1->next = node2->next; 或者 node1->next = node1->next->next;
    更好的做法是   ListNode **del = &node2;
                *del = (*del)->next; // 直接替换，只有两个指针参与 */
   
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *iter=head, **del=&head; //iter用于迭代，del用于删除，两者距离始终为n，当iter指向空指针时，*del所指结点即为删除的结点
        for(int i=0; i<n; i++,iter=iter->next);
        for(; iter!=NULL; iter=iter->next, del = &((*del)->next));
        *del = (*del)->next;  
        return head;
    }
};
