using CodeSandbox.Leetcode.Medium;

namespace CodeSandbox.Leetcode.Easy;

public class Solution {
    // merge two sorted linkedLists 
    
    public RangeBitwiseAnd1.Solution.ListNode MergeTwoLists(RangeBitwiseAnd1.Solution.ListNode node1, RangeBitwiseAnd1.Solution.ListNode node2)
    {
        if (node1 == null)
            return node2;
        if (node2 == null)
            return node1;
        if (node2.val <= node1.val)
            return new RangeBitwiseAnd1.Solution.ListNode(node2.val, MergeTwoLists(node1, node2.next));
        return new RangeBitwiseAnd1.Solution.ListNode(node1.val, MergeTwoLists(node1.next, node2));
    }
}

