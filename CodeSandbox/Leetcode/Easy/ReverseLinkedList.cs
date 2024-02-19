namespace CodeSandbox.Leetcode.Easy;

public class ReverseLinkedList
{

    public static long LargestPerimeter(int[] nums)
    {
        Array.Sort(nums);
        var sum = nums.Select(x => (long)x).Sum();
        long max = -1;
        for (var i = nums.Length - 1; i >= 0; i--)
        {
            sum -= nums[i];
            if (nums[i] < sum && nums[i] + sum > max)
                max = nums[i] + sum;
        }

        return max;
    }
}