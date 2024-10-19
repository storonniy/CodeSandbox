using CodeSandbox.Leetcode.Medium;

namespace CodeSandbox.Leetcode.Easy;

public class RemoveDuplicatesSortedList
{
    public RangeBitwiseAnd1.Solution.ListNode DeleteDuplicates(RangeBitwiseAnd1.Solution.ListNode head)
    {
        var current = head;
        while (current != null && current.next != null)
        {
            if (current.val == current.next.val)
                current.next = current.next.next;
            else
                current = current.next;
        }

        return head;
    }
}