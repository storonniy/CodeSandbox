namespace CodeSandbox.Leetcode.Easy;

public class RemoveDuplicatesSortedList
{
    public ListNode DeleteDuplicates(ListNode head)
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