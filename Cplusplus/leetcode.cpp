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
