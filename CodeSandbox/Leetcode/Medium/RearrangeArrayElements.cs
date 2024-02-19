namespace CodeSandbox.Leetcode.Medium;

public class RearrangeArrayElements
{
    public int[] RearrangeArray(int[] nums)
    {
        var positive = new int[nums.Length / 2];
        var p = 0;
        var negative = new int[nums.Length / 2];
        var n = 0;
        foreach (var x in nums)
            if (x > 0)
                positive[p++] = x;
            else
                negative[n++] = x;
        p = 0;
        for (var i = 0; i < nums.Length - 1; i += 2)
        {
            nums[i] = positive[p];
            nums[i + 1] = negative[p];
            p++;
        }

        return nums;
    }
}