// 求二叉树最大高度
// 时间复杂度O(n)，空间复杂度O(logn)
class Solution {
public:
  int maxDepth(TreeNode *root) {
    if (root == nullptr) return 0;
    return max(maxDepth(root->left), maxDepth(root->right)) + 1;
  }
};

// 判断二叉树是否对称
// 递归版，时间复杂度O(n)，空间复杂度O(logn)
class Solution {
public:
  bool isSymmetric(TreeNode *root) {
    return root ? isSymmetric(root->left, root->right) : true;
  }
  bool isSymmetric(TreeNode *left, TreeNode *right) {
    if (!left && !right) return true; // 终止条件
    if (!left || !right) return false; // 终止条件
    return left->val == right->val 
    && isSymmetric(left->left, right->right)
    && isSymmetric(left->right, right->left);  // 三方合并
  }
};

