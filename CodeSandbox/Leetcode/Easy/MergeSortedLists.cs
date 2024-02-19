namespace CodeSandbox.Leetcode.Easy;

public class Solution {
    // merge two sorted linkedLists 
    
    public ListNode MergeTwoLists(ListNode node1, ListNode node2)
    {
        if (node1 == null)
            return node2;
        if (node2 == null)
            return node1;
        if (node2.val <= node1.val)
            return new ListNode(node2.val, MergeTwoLists(node1, node2.next));
        return new ListNode(node1.val, MergeTwoLists(node1.next, node2));
    }
}

public class ListNode
{
    public int val { get; set; }
    public ListNode next { get; set; }

    public ListNode(int val = 0, ListNode next = null)
    {
        this.val = val;
        this.next = next;
    }
  }