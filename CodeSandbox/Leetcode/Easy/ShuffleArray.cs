namespace CodeSandbox.Leetcode.Easy;

public class ShuffleArray
{
    // https://leetcode.com/problems/shuffle-the-array/
    public int[] Shuffle(int[] nums, int n)
    {
        var result = new int[nums.Length];
        var pointer = 0;
        for (var i = 0; i < n; i ++)
        {
            result[pointer] = nums[i];
            result[pointer + 1] = nums[n + i];
            pointer += 2;
        }

        return result;
    }
}